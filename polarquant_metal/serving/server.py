"""OpenAI-compatible FastAPI server for PolarQuant Metal + speculative decoding.

Wraps mlx_lm.stream_generate() directly — no site-packages patching needed.
PolarQuant KV cache is created per-request via make_fused_cache().

KV cache bit-width is selected adaptively per-request based on macOS memory
pressure (vm.memory_pressure). A background monitor thread maintains a tier
with hysteresis; make_cache() reads the current tier at request time.

Tier mapping:
    normal   → FP16 (no compression)
    warn     → 4-bit K, 4-bit V
    critical → 3-bit K, 4-bit V  (V less sensitive, stays at 4-bit)

Models incompatible with PolarQuant (e.g. Phi-4-Mini) always use FP16.

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (streaming + non-streaming)
    GET  /health
    GET  /memory_tier           ← current pressure tier + active bits

Usage:
    python -m polarquant_metal.serving.server --model <model_id> --port 8082
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional

import mlx_lm
import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..integration import make_fused_cache, patch_sdpa
from ..memory_monitor import start_monitor, get_controller, is_compatible_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "polarquant"
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.model_id: str = ""
        self.draft_model_id: str = ""
        self.boundary_layers: int = 2
        self.num_draft_tokens: int = 4
        self.ready: bool = False
        self._pq_compatible: bool = True  # False for Phi-4-Mini etc.

    def load(
        self,
        model_id: str,
        draft_model_id: Optional[str] = None,
        boundary_layers: int = 2,
        num_draft_tokens: int = 4,
    ) -> None:
        logger.info("Loading main model: %s", model_id)
        patch_sdpa()
        self.model, self.tokenizer = mlx_lm.load(model_id)
        self.model_id = model_id
        self.boundary_layers = boundary_layers
        self.num_draft_tokens = num_draft_tokens
        self._pq_compatible = is_compatible_model(model_id)

        if not self._pq_compatible:
            logger.warning(
                "Model '%s' is not compatible with PolarQuant KV compression. "
                "All requests will use FP16 cache regardless of memory pressure.",
                model_id,
            )

        if draft_model_id:
            logger.info("Loading draft model: %s", draft_model_id)
            self.draft_model, _ = mlx_lm.load(draft_model_id)
            self.draft_model_id = draft_model_id

        # Warm up: single forward pass to force weight loading into GPU memory.
        logger.info("Warming up model weights...")
        from mlx_lm.models.cache import make_prompt_cache
        warm_tokens = mx.array([[1, 2, 3]])
        warm_cache = make_prompt_cache(self.model)
        _ = self.model(warm_tokens, cache=warm_cache)
        mx.metal.clear_cache()

        # Start memory pressure monitor singleton (no-op if already running).
        start_monitor()
        logger.info("Models ready. Memory pressure monitor started.")
        self.ready = True

    def make_cache(self):
        """Create a per-request KV cache at the current memory pressure tier.

        Falls back to FP16 (no compression) if:
        - Memory pressure is 'normal', OR
        - Model is incompatible with PolarQuant (e.g. Phi-4-Mini).

        Returns a cache object compatible with mlx_lm.stream_generate(prompt_cache=...).
        """
        from mlx_lm.models.cache import make_prompt_cache

        if not self._pq_compatible:
            return make_prompt_cache(self.model)

        tier = get_controller().tier
        if tier.bits is None:
            return make_prompt_cache(self.model)

        return make_fused_cache(
            self.model,
            bits=tier.bits,
            bits_v=tier.bits_v,
            boundary_layers=self.boundary_layers,
        )

    @property
    def current_tier(self) -> str:
        if not self._pq_compatible:
            return "fp16-forced"
        return get_controller().tier_name


# Module-level singleton state
_state = ModelState()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(state: Optional[ModelState] = None) -> FastAPI:
    """Create the FastAPI app. Pass a pre-loaded ModelState for testing."""
    global _state
    if state is not None:
        _state = state

    app = FastAPI(title="PolarQuant Metal Server", version="0.1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok" if _state.ready else "loading",
            "model": _state.model_id,
            "memory_tier": _state.current_tier,
        }

    @app.get("/memory_tier")
    def memory_tier():
        tier = get_controller().tier
        return {
            "tier": get_controller().tier_name,
            "bits_k": tier.bits,
            "bits_v": tier.bits_v,
            "pq_compatible": _state._pq_compatible,
        }

    @app.get("/v1/models")
    def list_models():
        models = []
        if _state.model_id:
            models.append({
                "id": _state.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "polarquant-metal",
            })
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if not _state.ready:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prompt = _state.tokenizer.apply_chat_template(
            [m.dict() for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        if request.stream:
            return StreamingResponse(
                _stream_response(prompt, request),
                media_type="text/event-stream",
            )
        return await _non_stream_response(prompt, request)

    return app


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _build_gen_kwargs(request: ChatCompletionRequest) -> tuple:
    """Build (main_cache, gen_kwargs) for a request."""
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache

    sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
    main_cache = _state.make_cache()

    if _state.draft_model is not None:
        draft_cache = make_prompt_cache(_state.draft_model)
        combined_cache = main_cache + draft_cache
        kwargs = dict(
            max_tokens=request.max_tokens,
            sampler=sampler,
            prompt_cache=combined_cache,
            draft_model=_state.draft_model,
            num_draft_tokens=_state.num_draft_tokens,
        )
        return main_cache, kwargs

    kwargs = dict(
        max_tokens=request.max_tokens,
        sampler=sampler,
        prompt_cache=main_cache,
    )
    return main_cache, kwargs


async def _stream_response(
    prompt: str,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Yield SSE chunks in OpenAI delta format."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_name = _state.model_id
    created = int(time.time())

    yield _sse_chunk(request_id, model_name, created, delta={"role": "assistant"})

    main_cache, gen_kwargs = _build_gen_kwargs(request)
    stop_seqs = request.stop or []
    finish_reason = None
    loop = asyncio.get_event_loop()

    def _generate():
        return list(mlx_lm.stream_generate(
            _state.model,
            _state.tokenizer,
            prompt=prompt,
            **gen_kwargs,
        ))

    responses = await loop.run_in_executor(None, _generate)

    for resp in responses:
        if resp.text:
            if any(s in resp.text for s in stop_seqs):
                finish_reason = "stop"
                break
            yield _sse_chunk(request_id, model_name, created, delta={"content": resp.text})
        if resp.finish_reason:
            finish_reason = resp.finish_reason
            break
        await asyncio.sleep(0)

    yield _sse_chunk(request_id, model_name, created, delta={}, finish_reason=finish_reason or "stop")
    yield "data: [DONE]\n\n"

    if responses:
        last = responses[-1]
        logger.info(
            "request=%s prompt_tokens=%d gen_tokens=%d tps=%.1f tier=%s spec=%s",
            request_id, last.prompt_tokens, last.generation_tokens,
            last.generation_tps, _state.current_tier,
            "yes" if _state.draft_model else "no",
        )


async def _non_stream_response(
    prompt: str,
    request: ChatCompletionRequest,
) -> JSONResponse:
    main_cache, gen_kwargs = _build_gen_kwargs(request)
    loop = asyncio.get_event_loop()

    def _generate():
        pieces = []
        finish = None
        last = None
        for resp in mlx_lm.stream_generate(
            _state.model,
            _state.tokenizer,
            prompt=prompt,
            **gen_kwargs,
        ):
            if resp.text:
                pieces.append(resp.text)
            if resp.finish_reason:
                finish = resp.finish_reason
            last = resp
        return "".join(pieces), finish, last

    content, finish_reason, last_resp = await loop.run_in_executor(None, _generate)

    prompt_tokens = last_resp.prompt_tokens if last_resp else 0
    gen_tokens = last_resp.generation_tokens if last_resp else 0
    gen_tps = last_resp.generation_tps if last_resp else 0.0

    tier = get_controller().tier

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _state.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason or "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": gen_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
        },
        "x_polarquant": {
            "generation_tps": round(gen_tps, 1),
            "memory_tier": _state.current_tier,
            "bits_k": tier.bits,
            "bits_v": tier.bits_v,
            "speculative": _state.draft_model is not None,
        },
    })


def _sse_chunk(
    request_id: str,
    model: str,
    created: int,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(
    model_id: str,
    *,
    draft_model_id: Optional[str] = None,
    port: int = 8082,
    host: str = "0.0.0.0",
    boundary_layers: int = 2,
    num_draft_tokens: int = 4,
) -> None:
    """Load models and start the server. Blocking call."""
    import uvicorn
    _state.load(
        model_id=model_id,
        draft_model_id=draft_model_id,
        boundary_layers=boundary_layers,
        num_draft_tokens=num_draft_tokens,
    )
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="PolarQuant Metal Server")
    parser.add_argument("--model", required=True, help="Main model path or HF repo ID")
    parser.add_argument("--draft-model", default=None, help="Draft model for speculative decoding")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--boundary-layers", type=int, default=2,
                        help="Number of boundary layers between linear and standard attention (Qwen3.5 hybrid)")
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--hysteresis", type=float, default=5.0,
                        help="Seconds of sustained improvement before tier upgrade")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Memory pressure poll interval in seconds")
    args = parser.parse_args()

    # Start monitor before model load so it's ready by the time serve() returns.
    start_monitor(hysteresis_s=args.hysteresis, poll_interval_s=args.poll_interval)

    serve(
        model_id=args.model,
        draft_model_id=args.draft_model,
        port=args.port,
        host=args.host,
        boundary_layers=args.boundary_layers,
        num_draft_tokens=args.num_draft_tokens,
    )
