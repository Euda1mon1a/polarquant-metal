#!/usr/bin/env python3
"""
Long-context benchmark: Gemma 4 31B-IT-4bit  —  FP16 vs PolarQuant original vs simdgroup.

Gemma 4 is a dense model (all standard attention layers), making it the primary
target for PolarQuant. At 32K context the KV cache is the dominant bandwidth
bottleneck — exactly where compression pays off.

Three paths compared:
  FP16     — standard mlx-lm KVCache, no compression
  PQ-orig  — PolarQuant 4-bit K/V, scalar kernels (use_simd=False)
  PQ-simd  — PolarQuant 4-bit K/V, simdgroup kernels (use_simd=True)

Memory watchdog monitors pressure every 3s and kills the process before
a Metal driver panic can occur (learned from Qwen 35B experiments).

Usage:
    cd /tmp/polarquant-metal
    nohup ~/.mlx-server-env/bin/python3 benchmarks/bench_gemma4_longctx.py \
        > /tmp/bench_gemma4.log 2>&1 &
    tail -f /tmp/bench_gemma4.log
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx_vlm as mlx_lm  # Gemma 4 is in mlx-vlm, not mlx-lm

from polarquant_metal.integration import make_fused_cache, patch_sdpa
from polarquant_metal.turboquant_cache import TurboQuantKVCache
from benchmarks.watchdog import start_watchdog

MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"
GEN_TOKENS = 64
CONTEXT_LENGTHS = [8_192, 16_384, 32_768]

# Chunked prefill: process the prompt in steps to reduce peak GPU memory.
# Without this, Gemma 4 31B at 32K context wires ~54GB on a 64GB machine.
# With 512-token chunks, peak activation memory drops dramatically.
PREFILL_STEP_SIZE = 512

_PARA = (
    "The development of large language models represents a significant milestone "
    "in artificial intelligence research. These models, trained on vast corpora "
    "of text, demonstrate emergent capabilities that were not explicitly programmed. "
    "Researchers continue to investigate the relationship between scale, data quality, "
    "and downstream task performance. Architectural innovations such as attention "
    "mechanisms and positional encodings have proven critical to enabling long-context "
    "reasoning. Hardware accelerators optimized for matrix operations form the "
    "computational substrate for both training and inference workloads. "
    "Memory bandwidth, not raw compute, increasingly constrains throughput at "
    "long sequence lengths during autoregressive decoding. "
)


def build_prompt(processor, target_tokens: int) -> str:
    # mlx-vlm returns a Processor; get the underlying text tokenizer
    tok = getattr(processor, 'tokenizer', processor)
    repeats = max(1, target_tokens // len(tok.encode(_PARA)) + 2)
    body = _PARA * repeats
    tokens = tok.encode(body)
    if len(tokens) > target_tokens - 20:
        tokens = tokens[:target_tokens - 20]
        body = tok.decode(tokens)
    return body + "\n\nSummarize the key insight in one sentence:"


def run_path(model, processor, prompt, cache, label):
    """Run generation, return (tps, gen_tokens, kv_mb, fp16_equiv_mb)."""
    t0 = time.perf_counter()
    last = None
    for resp in mlx_lm.stream_generate(
        model, processor, prompt=prompt,
        max_tokens=GEN_TOKENS,
        prompt_cache=cache,           # passed via **kwargs → generate_step
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        last = resp
    elapsed = time.perf_counter() - t0

    tps = last.generation_tps if last else 0.0
    gen_tok = last.generation_tokens if last else 0
    prompt_tok = last.prompt_tokens if last else 0

    tq_caches = [c for c in cache if isinstance(c, TurboQuantKVCache)]
    kv_bytes = sum(c.memory_bytes() for c in tq_caches)
    kv_mb = kv_bytes / 1024 / 1024

    fp16_mb = 0.0
    if tq_caches:
        ref = tq_caches[0]
        if ref._k_packed is not None:
            nh = ref._k_packed.shape[1]
            fp16_mb = (2 * nh * ref.offset * ref._head_dim * 2 / 1024 / 1024
                       * len(tq_caches))

    if kv_mb > 0:
        print(f"  [{label}] ctx={prompt_tok} gen={gen_tok} "
              f"tps={tps:.1f} kv={kv_mb:.0f}MB fp16eq={fp16_mb:.0f}MB "
              f"cmpr={fp16_mb/kv_mb:.1f}x")
    else:
        print(f"  [{label}] ctx={prompt_tok} gen={gen_tok} tps={tps:.1f} kv=FP16")
    sys.stdout.flush()

    mx.clear_cache()
    return tps, gen_tok, kv_mb, fp16_mb


def main():
    # Start watchdog before any Metal allocation
    watchdog = start_watchdog(os.getpid(), log_file="/tmp/bench_gemma4_watchdog.log")

    print("=" * 72)
    print(f"PolarQuant Long-Context E2E: {MODEL_ID}")
    print(f"Paths: FP16 | PQ-orig (scalar) | PQ-simd")
    print(f"Context lengths: {[f'{c//1024}K' for c in CONTEXT_LENGTHS]}")
    print(f"Gen tokens per run: {GEN_TOKENS}")
    print(f"Watchdog: active (pressure kill + {52}GB wired limit)")
    print("=" * 72)
    sys.stdout.flush()

    print("Loading model...")
    t0 = time.perf_counter()
    patch_sdpa()
    model, processor = mlx_lm.load(MODEL_ID)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s\n")
    sys.stdout.flush()

    from mlx_vlm.models.cache import make_prompt_cache

    print(f"{'ctx':>7}  {'FP16':>8}  {'PQ-orig':>9}  {'PQ-simd':>9}  "
          f"{'orig/fp16':>10}  {'simd/fp16':>10}  {'kv_mb':>7}  {'cmpr':>6}")
    print("-" * 76)
    sys.stdout.flush()

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n--- {ctx_len//1024}K context ---")
        sys.stdout.flush()

        prompt = build_prompt(processor, ctx_len)

        cache_fp16 = make_prompt_cache(model)
        tps_fp16, _, _, _ = run_path(model, processor, prompt, cache_fp16, "FP16")

        cache_orig = make_fused_cache(model, bits=4, bits_v=4, use_simd=False)
        tps_orig, _, kv_mb, fp16_mb = run_path(model, processor, prompt, cache_orig, "PQ-orig")

        cache_simd = make_fused_cache(model, bits=4, bits_v=4, use_simd=True)
        tps_simd, _, _, _ = run_path(model, processor, prompt, cache_simd, "PQ-simd")

        cmpr = fp16_mb / kv_mb if kv_mb > 0 else 0
        print(
            f"{ctx_len//1024}K  {tps_fp16:>8.1f}  {tps_orig:>9.1f}  {tps_simd:>9.1f}  "
            f"{tps_orig/tps_fp16:>9.2f}x  {tps_simd/tps_fp16:>9.2f}x  "
            f"{kv_mb:>7.0f}  {cmpr:>5.1f}x"
        )
        sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
