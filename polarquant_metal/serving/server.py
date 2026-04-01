"""OpenAI-compatible FastAPI server for 72B + PolarQuant + speculative decoding.

Wraps mlx_lm.stream_generate() directly — no site-packages patching needed.
PolarQuant cache is created per-request via make_fused_cache().

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions  (streaming + non-streaming)
    GET  /health

Usage:
    python scripts/serve_72b.py --port 8082
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional

import mlx_lm
import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..integration import make_fused_cache, patch_sdpa

logger = logging.getLogger(__name__)


# --- Request / Response models ---

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "polarquant-72b"
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[list[str]] = None


# --- Server state ---

class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.model_id: str = ""
        self.draft_model_id: str = ""
        self.pq_bits: int = 3
        self.boundary_layers: int = 2
        self.num_draft_tokens: int = 4
        self.ready: bool = False

    def load(
        self,
        model_id: str,
        draft_model_id: Optional[str] = None,
        pq_bits: int = 3,
        boundary_layers: int = 2,
        num_draft_tokens: int = 4,
    ):
        logger.info(f"Loading main model: {model_id}")
        patch_sdpa()
        self.model, self.tokenizer = mlx_lm.load(model_id)
        self.model_id = model_id
        self.pq_bits = pq_bits
        self.boundary_layers = boundary_layers
        self.num_draft_tokens = num_draft_tokens

        if draft_model_id:
            logger.info(f"Loading draft model: {draft_model_id}")
            self.draft_model, _ = mlx_lm.load(draft_model_id)
            self.draft_model_id = draft_model_id

        # Warm up: single forward pass to force weight loading into GPU memory
        logger.info("Warming up model weights...")
        warm_tokens = mx.array([[1, 2, 3]])
        from mlx_lm.models.cache import make_prompt_cache
        warm_cache = make_prompt_cache(self.model)
        _ = self.model(warm_tokens, cache=warm_cache)
        mx.metal.clear_cache()

        logger.info("Models ready.")
        self.ready = True

    def make_cache(self):
        return make_fused_cache(
            self.model,
            bits=self.pq_bits,
            boundary_layers=self.boundary_layers,
        )


_state = ModelState()


def create_app(state: ModelState = None) -> FastAPI:
    """Create the FastAPI app. Pass a pre-loaded ModelState for testing."""
    global _state
    if state is not None:
        _state = state

    app = FastAPI(title="PolarQuant 72B Server", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok" if _state.ready else "loading", "model": _state.model_id}

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


def _build_gen_kwargs(request: ChatCompletionRequest) -> tuple[list, dict]:
    """Build prompt_cache and generation kwargs for a request.

    generate_step accepts temp/top_p directly.
    speculative_generate_step requires a sampler callable — stream_generate
    strips temp/top_p and raises when draft_model is set. We always pass
    sampler= to be safe for both paths.
    """
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
    main_cache = _state.make_cache()

    if _state.draft_model is not None:
        from mlx_lm.models.cache import make_prompt_cache
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
    """Stream SSE chunks in OpenAI delta format."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_name = _state.model_id
    created = int(time.time())

    yield _sse_chunk(request_id, model_name, created, delta={"role": "assistant"})

    main_cache, gen_kwargs = _build_gen_kwargs(request)
    stop_seqs = request.stop or []
    buffer = ""
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
            buffer += resp.text
            if any(s in buffer for s in stop_seqs):
                finish_reason = "stop"
                break
            yield _sse_chunk(request_id, model_name, created, delta={"content": resp.text})
        if resp.finish_reason:
            finish_reason = resp.finish_reason
            break
        await asyncio.sleep(0)

    yield _sse_chunk(request_id, model_name, created, delta={}, finish_reason=finish_reason or "stop")
    yield "data: [DONE]\n\n"

    # Log stats from last response
    if responses:
        last = responses[-1]
        logger.info(
            f"request={request_id} "
            f"prompt_tokens={last.prompt_tokens} "
            f"gen_tokens={last.generation_tokens} "
            f"tps={last.generation_tps:.1f} "
            f"pq_bits={_state.pq_bits} "
            f"spec={'yes' if _state.draft_model else 'no'}"
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
            "pq_bits": _state.pq_bits,
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


def serve(
    model_id: str,
    *,
    draft_model_id: Optional[str] = None,
    port: int = 8082,
    host: str = "0.0.0.0",
    pq_bits: int = 3,
    boundary_layers: int = 2,
    num_draft_tokens: int = 4,
):
    """Load models and start the server. Blocking call."""
    import uvicorn
    _state.load(
        model_id=model_id,
        draft_model_id=draft_model_id,
        pq_bits=pq_bits,
        boundary_layers=boundary_layers,
        num_draft_tokens=num_draft_tokens,
    )
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
