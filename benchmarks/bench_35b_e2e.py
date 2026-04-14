#!/usr/bin/env python3
"""
End-to-end benchmark: Qwen3.5-35B-A3B-4bit  —  FP16 vs PolarQuant KV cache.

Loads the model once, runs generation at several seeded context lengths,
and reports tokens/sec and KV cache memory for each path.

Usage:
    cd /tmp/polarquant-metal
    ~/.mlx-server-env/bin/python3 benchmarks/bench_35b_e2e.py

Model is assumed to be cached at the default HuggingFace cache location.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx_lm

from polarquant_metal.integration import make_fused_cache, patch_sdpa
from polarquant_metal.turboquant_cache import TurboQuantKVCache

MODEL_ID = "mlx-community/Qwen3.5-35B-A3B-4bit"
GEN_TOKENS = 64
# Context lengths to test — seed prompt is padded to approximately these lengths
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384]

# Filler paragraph used to pad prompts to target context lengths
_FILLER = (
    "The history of computing spans several decades of innovation, "
    "from vacuum tubes to transistors to integrated circuits. "
    "Each generation of hardware enabled new categories of software "
    "and new approaches to problem solving. "
) * 20  # ~600 tokens per repeat


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt that tokenizes to approximately target_tokens."""
    question = "\n\nBased on the above context, briefly explain the key insight:"
    # Estimate tokens: filler is ~600 per repeat, scale up
    repeats = max(1, target_tokens // 600)
    body = _FILLER * repeats
    prompt = body + question
    tokens = tokenizer.encode(prompt)
    # Trim or pad to get close to target
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens - 10]
        prompt = tokenizer.decode(tokens)
    return prompt


def run_generate(model, tokenizer, prompt: str, cache) -> tuple[float, int, float]:
    """Run generation and return (tps, gen_tokens, cache_mb).

    Returns:
        tps: generation tokens per second
        gen_tokens: number of tokens generated
        cache_mb: KV cache memory in MB (from memory_bytes() or 0)
    """
    tps = 0.0
    gen_tokens = 0
    last = None
    for resp in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=GEN_TOKENS,
        prompt_cache=cache,
    ):
        last = resp
    if last:
        tps = last.generation_tps
        gen_tokens = last.generation_tokens

    cache_bytes = sum(
        c.memory_bytes() for c in cache if isinstance(c, TurboQuantKVCache)
    )
    cache_mb = cache_bytes / 1024 / 1024
    return tps, gen_tokens, cache_mb


def main():
    print("=" * 72)
    print(f"PolarQuant E2E: {MODEL_ID}")
    print(f"Gen tokens: {GEN_TOKENS}  |  Context lengths: {CONTEXT_LENGTHS}")
    print("=" * 72)

    print("Loading model...")
    patch_sdpa()
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

    from mlx_lm.models.cache import make_prompt_cache

    header = f"{'Path':<14}  {'L_ctx':>6}  {'tok/s':>7}  {'kv_mb':>7}  {'fp16_mb':>8}  {'cmpr':>6}"
    print(header)
    print("-" * len(header))

    for ctx_len in CONTEXT_LENGTHS:
        prompt = build_prompt(tokenizer, ctx_len)
        actual_ctx = len(tokenizer.encode(prompt))

        # FP16 baseline
        cache_fp16 = make_prompt_cache(model)
        tps_fp16, _, _ = run_generate(model, tokenizer, prompt, cache_fp16)
        mx.metal.clear_cache()

        # PolarQuant 4-bit
        cache_pq4 = make_fused_cache(model, bits=4, bits_v=4)
        tps_pq4, _, mb_pq4 = run_generate(model, tokenizer, prompt, cache_pq4)
        # Estimate equivalent FP16 size: heads * offset * D * 2 * 2 (K+V) bytes
        # We get this from one reference TQ cache
        ref = next((c for c in cache_pq4 if isinstance(c, TurboQuantKVCache)), None)
        if ref and ref._head_dim:
            B, nh = 1, ref._k_packed.shape[1] if ref._k_packed is not None else 0
            fp16_mb = 2 * nh * ref.offset * ref._head_dim * 2 / 1024 / 1024 if nh else 0
            fp16_mb *= sum(1 for c in cache_pq4 if isinstance(c, TurboQuantKVCache))
        else:
            fp16_mb = 0
        mx.metal.clear_cache()

        # PolarQuant 3-bit K / 4-bit V
        cache_pq3 = make_fused_cache(model, bits=3, bits_v=4)
        tps_pq3, _, mb_pq3 = run_generate(model, tokenizer, prompt, cache_pq3)
        mx.metal.clear_cache()

        cmpr4 = fp16_mb / mb_pq4 if mb_pq4 > 0 else 0
        cmpr3 = fp16_mb / mb_pq3 if mb_pq3 > 0 else 0

        print(f"{'FP16':<14}  {actual_ctx:>6}  {tps_fp16:>7.1f}  {'—':>7}  {'—':>8}  {'1.0x':>6}")
        print(f"{'PQ4 (4K/4V)':<14}  {actual_ctx:>6}  {tps_pq4:>7.1f}  {mb_pq4:>7.1f}  {fp16_mb:>8.1f}  {cmpr4:>5.1f}x")
        print(f"{'PQ3 (3K/4V)':<14}  {actual_ctx:>6}  {tps_pq3:>7.1f}  {mb_pq3:>7.1f}  {fp16_mb:>8.1f}  {cmpr3:>5.1f}x")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
