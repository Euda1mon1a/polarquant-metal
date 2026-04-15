#!/usr/bin/env python3
"""
Experiment 10 — Lazy Prefill: defer FP16→PQ quantization to first decode step.

Hypothesis: By holding all prefill tokens in FP16 and bulk-quantizing at
the first L_q=1 decode step, chunked prefill becomes O(N) instead of O(N²),
making 64K+ contexts practical.

Current behavior (before fix):
  - FP16 storage until min_fused_context (512 tokens)
  - After that: each 512-token prefill chunk dequantizes ALL prior PQ tokens
  - 64K prefill: >112 min (killed), vs ~2 min for FP16

Expected after fix:
  - FP16 storage through entire prefill (no mid-prefill quantize trigger)
  - First decode step: _bulk_quantize() fires once
  - 64K prefill: ~2-3 min (FP16 SDPA throughout), decode tps unchanged

Run on Mini (Qwen3.5-35B, M4 Pro 64GB):

    cd /tmp/polarquant-metal
    source .venv/bin/activate   # or .venv-mini/bin/activate
    python benchmarks/exp10_lazy_prefill.py

Outputs prefill time and decode tps at each context length.
"""

import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx_lm

_total_ram = mx.device_info()["memory_size"]
mx.set_memory_limit(int(_total_ram * 0.75))
mx.set_cache_limit(int(_total_ram * 0.20))
mx.set_wired_limit(int(_total_ram * 0.70))

MODEL_ID = "mlx-community/Qwen2.5-35B-Instruct-4bit"
GEN_TOKENS = 128
PREFILL_STEP_SIZE = 512
CONTEXT_LENGTHS = [8_192, 32_768, 65_536]

_PARA = (
    "The development of large language models represents a significant milestone "
    "in artificial intelligence research. These models, trained on vast corpora "
    "of text, demonstrate emergent capabilities that were not explicitly programmed. "
    "Researchers continue to investigate the relationship between scale, data quality, "
    "and downstream task performance. Memory bandwidth, not raw compute, increasingly "
    "constrains throughput at long sequence lengths during autoregressive decoding. "
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    repeats = max(1, target_tokens // len(tokenizer.encode(_PARA)) + 2)
    body = _PARA * repeats
    tokens = tokenizer.encode(body)
    if len(tokens) > target_tokens - 20:
        tokens = tokens[:target_tokens - 20]
        body = tokenizer.decode(tokens)
    return body + "\n\nSummarize the key insight in one sentence:"


def run_one(model, tokenizer, prompt, cache, label):
    t_prefill_start = time.perf_counter()
    last = None
    first_token_time = None

    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=GEN_TOKENS,
        prompt_cache=cache,
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        last = resp

    t_end = time.perf_counter()
    prefill_s = (first_token_time - t_prefill_start) if first_token_time else 0
    tps = last.generation_tps if last else 0.0
    prompt_tok = last.prompt_tokens if last else 0

    from polarquant_metal.turboquant_cache import TurboQuantKVCache
    tq_caches = [c for c in cache if isinstance(c, TurboQuantKVCache)]
    kv_mb = sum(c.memory_bytes() for c in tq_caches) / 1024 / 1024

    print(
        f"  [{label}] ctx={prompt_tok} prefill={prefill_s:.1f}s "
        f"tps={tps:.1f} kv={kv_mb:.0f}MB"
    )
    sys.stdout.flush()
    return prefill_s, tps, kv_mb


def main():
    print("=" * 72)
    print(f"Exp 10 — Lazy Prefill  ({MODEL_ID})")
    print(f"Contexts: {[f'{c//1024}K' for c in CONTEXT_LENGTHS]}")
    print(f"Gen tokens: {GEN_TOKENS}  |  Prefill chunk: {PREFILL_STEP_SIZE}")
    print("=" * 72)
    sys.stdout.flush()

    from polarquant_metal.integration import patch_sdpa, make_fused_cache
    patch_sdpa()

    print("Loading model...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")
    sys.stdout.flush()

    from mlx_lm.models.cache import make_prompt_cache

    results = []
    for ctx_len in CONTEXT_LENGTHS:
        print(f"--- {ctx_len // 1024}K context ---")
        sys.stdout.flush()
        prompt = build_prompt(tokenizer, ctx_len)

        # FP16 baseline
        cache_fp16 = make_prompt_cache(model)
        prefill_fp16, tps_fp16, _ = run_one(model, tokenizer, prompt, cache_fp16, "FP16")
        del cache_fp16
        mx.clear_cache()
        gc.collect()

        # PQ-simd with lazy prefill fix
        cache_pq = make_fused_cache(model, bits=4, bits_v=4, use_simd=True)
        prefill_pq, tps_pq, kv_mb = run_one(model, tokenizer, prompt, cache_pq, "PQ-lazy")
        del cache_pq
        mx.clear_cache()
        gc.collect()

        results.append((ctx_len, prefill_fp16, tps_fp16, prefill_pq, tps_pq, kv_mb))

    print("\n" + "=" * 72)
    print("Summary")
    print(f"{'ctx':>6}  {'fp16_pre':>9}  {'fp16_tps':>9}  {'pq_pre':>8}  {'pq_tps':>8}  {'kv_mb':>7}  {'pre_ratio':>9}")
    print("-" * 72)
    for ctx, pre_fp16, tps_fp16, pre_pq, tps_pq, kv_mb in results:
        ratio = pre_pq / pre_fp16 if pre_fp16 > 0 else 0
        print(
            f"{ctx//1024:>5}K  {pre_fp16:>8.1f}s  {tps_fp16:>9.1f}  "
            f"{pre_pq:>7.1f}s  {tps_pq:>8.1f}  {kv_mb:>7.0f}  {ratio:>8.1f}x"
        )
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
