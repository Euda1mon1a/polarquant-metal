#!/usr/bin/env python3
"""
Long-context benchmark: Qwen3.5-35B-A3B-4bit  —  FP16 vs PolarQuant original vs simdgroup.

Tests at 32K, 64K, 128K context to find the crossover point where PolarQuant
KV cache compression outweighs kernel overhead.

Three paths compared:
  FP16     — standard mlx-lm KVCache, no compression
  PQ-orig  — PolarQuant 4-bit K/V, scalar kernels (pre-simdgroup, original result)
  PQ-simd  — PolarQuant 4-bit K/V, simdgroup kernels (current)

Usage:
    cd /tmp/polarquant-metal
    nohup ~/.mlx-server-env/bin/python3 benchmarks/bench_35b_longctx.py \
        > /tmp/bench_longctx.log 2>&1 &
    tail -f /tmp/bench_longctx.log
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx_lm

from polarquant_metal.integration import make_fused_cache, patch_sdpa
from polarquant_metal.turboquant_cache import TurboQuantKVCache

MODEL_ID  = "mlx-community/Qwen3.5-35B-A3B-4bit"
GEN_TOKENS = 64
CONTEXT_LENGTHS = [32_768, 65_536, 131_072]

# Build filler from a public-domain-style passage to avoid trivial repetition artifacts.
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


def build_prompt(tokenizer, target_tokens: int) -> str:
    repeats = max(1, target_tokens // len(tokenizer.encode(_PARA)) + 2)
    body = _PARA * repeats
    tokens = tokenizer.encode(body)
    if len(tokens) > target_tokens - 20:
        tokens = tokens[:target_tokens - 20]
        body = tokenizer.decode(tokens)
    return body + "\n\nSummarize the key insight in one sentence:"


def run_path(model, tokenizer, prompt, cache, label):
    """Run generation, return (tps, gen_tokens, kv_mb, fp16_equiv_mb)."""
    t0 = time.perf_counter()
    last = None
    for resp in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=GEN_TOKENS,
        prompt_cache=cache,
    ):
        last = resp
    elapsed = time.perf_counter() - t0

    tps = last.generation_tps if last else 0.0
    gen_tok = last.generation_tokens if last else 0
    prompt_tok = last.prompt_tokens if last else 0

    tq_caches = [c for c in cache if isinstance(c, TurboQuantKVCache)]
    kv_bytes = sum(c.memory_bytes() for c in tq_caches)
    kv_mb = kv_bytes / 1024 / 1024

    # FP16 equivalent for the quantized layers
    fp16_mb = 0.0
    if tq_caches:
        ref = tq_caches[0]
        if ref._k_packed is not None:
            nh = ref._k_packed.shape[1]
            fp16_mb = (2 * nh * ref.offset * ref._head_dim * 2 / 1024 / 1024
                       * len(tq_caches))

    print(f"  [{label}] ctx={prompt_tok} gen={gen_tok} "
          f"tps={tps:.1f} kv={kv_mb:.0f}MB fp16eq={fp16_mb:.0f}MB "
          f"cmpr={fp16_mb/kv_mb:.1f}x" if kv_mb > 0 else
          f"  [{label}] ctx={prompt_tok} gen={gen_tok} tps={tps:.1f} kv=FP16")
    sys.stdout.flush()

    mx.clear_cache()
    return tps, gen_tok, kv_mb, fp16_mb


def main():
    print("=" * 72)
    print(f"PolarQuant Long-Context E2E: {MODEL_ID}")
    print(f"Paths: FP16 | PQ-orig (scalar) | PQ-simd")
    print(f"Context lengths: {[f'{c//1024}K' for c in CONTEXT_LENGTHS]}")
    print(f"Gen tokens per run: {GEN_TOKENS}")
    print("=" * 72)
    sys.stdout.flush()

    print("Loading model...")
    t0 = time.perf_counter()
    patch_sdpa()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s\n")
    sys.stdout.flush()

    from mlx_lm.models.cache import make_prompt_cache

    print(f"{'ctx':>7}  {'FP16':>8}  {'PQ-orig':>9}  {'PQ-simd':>9}  "
          f"{'orig/fp16':>10}  {'simd/fp16':>10}  {'kv_mb':>7}  {'cmpr':>6}")
    print("-" * 76)
    sys.stdout.flush()

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n--- {ctx_len//1024}K context ---")
        sys.stdout.flush()

        prompt = build_prompt(tokenizer, ctx_len)
        actual = len(tokenizer.encode(prompt))

        cache_fp16 = make_prompt_cache(model)
        tps_fp16, _, _, _ = run_path(model, tokenizer, prompt, cache_fp16, "FP16")

        cache_orig = make_fused_cache(model, bits=4, bits_v=4, use_simd=False)
        tps_orig, _, kv_mb, fp16_mb = run_path(model, tokenizer, prompt, cache_orig, "PQ-orig")

        cache_simd = make_fused_cache(model, bits=4, bits_v=4, use_simd=True)
        tps_simd, _, _, _ = run_path(model, tokenizer, prompt, cache_simd, "PQ-simd")

        cmpr = fp16_mb / kv_mb if kv_mb > 0 else 0
        print(
            f"{actual:>7}  {tps_fp16:>8.1f}  {tps_orig:>9.1f}  {tps_simd:>9.1f}  "
            f"{tps_orig/tps_fp16:>9.2f}x  {tps_simd/tps_fp16:>9.2f}x  "
            f"{kv_mb:>7.0f}  {cmpr:>5.1f}x"
        )
        sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
