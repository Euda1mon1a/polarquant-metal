#!/usr/bin/env python3
"""
Single-path Gemma 4 31B benchmark — FP16 or PQ-simd.

Run each mode as a separate process to avoid Metal buffer accumulation:

    # FP16 baseline (16K and 32K — 8K already known: 4.2 tps)
    nohup ~/.mlx-server-env/bin/python3 benchmarks/bench_gemma4_clean.py \\
        --mode fp16 --contexts 16384 32768 > /tmp/bench_gemma4_fp16.log 2>&1 &

    # PQ-simd (all three)
    nohup ~/.mlx-server-env/bin/python3 benchmarks/bench_gemma4_clean.py \\
        --mode simd --contexts 8192 16384 32768 > /tmp/bench_gemma4_simd.log 2>&1 &

Requires all heavy services disabled via launchctl bootout before running.
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx_vlm as mlx_lm

from benchmarks.watchdog import start_watchdog

MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"
GEN_TOKENS = 128          # more tokens → more stable tps measurement
PREFILL_STEP_SIZE = 512   # chunked prefill to cap peak activation memory

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
    tok = getattr(processor, "tokenizer", processor)
    repeats = max(1, target_tokens // len(tok.encode(_PARA)) + 2)
    body = _PARA * repeats
    tokens = tok.encode(body)
    if len(tokens) > target_tokens - 20:
        tokens = tokens[: target_tokens - 20]
        body = tok.decode(tokens)
    return body + "\n\nSummarize the key insight in one sentence:"


def run_path(model, processor, prompt, cache, label: str):
    """Run generation and return (tps, prompt_tokens, gen_tokens, kv_mb, fp16_mb)."""
    from polarquant_metal.turboquant_cache import TurboQuantKVCache

    t0 = time.perf_counter()
    last = None
    for resp in mlx_lm.stream_generate(
        model,
        processor,
        prompt=prompt,
        max_tokens=GEN_TOKENS,
        prompt_cache=cache,
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        last = resp
    elapsed = time.perf_counter() - t0  # noqa: F841

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
            fp16_mb = (
                2 * nh * ref.offset * ref._head_dim * 2 / 1024 / 1024 * len(tq_caches)
            )

    if kv_mb > 0:
        cmpr = fp16_mb / kv_mb if kv_mb > 0 else 0
        print(
            f"  [{label}] ctx={prompt_tok} gen={gen_tok} tps={tps:.1f} "
            f"kv={kv_mb:.0f}MB fp16eq={fp16_mb:.0f}MB cmpr={cmpr:.1f}x"
        )
    else:
        print(f"  [{label}] ctx={prompt_tok} gen={gen_tok} tps={tps:.1f} kv=FP16")
    sys.stdout.flush()

    return tps, prompt_tok, gen_tok, kv_mb, fp16_mb


def main():
    parser = argparse.ArgumentParser(description="Single-path Gemma 4 benchmark")
    parser.add_argument("--mode", choices=["fp16", "simd"], required=True)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[8_192, 16_384, 32_768],
        help="Context lengths to test",
    )
    args = parser.parse_args()

    label = "FP16" if args.mode == "fp16" else "PQ-simd"
    watchdog_log = f"/tmp/bench_gemma4_watchdog_{args.mode}.log"

    start_watchdog(os.getpid(), log_file=watchdog_log)

    print("=" * 72)
    print(f"Gemma 4 31B — {label} path")
    print(f"Model: {MODEL_ID}")
    print(f"Contexts: {[f'{c//1024}K' for c in args.contexts]}")
    print(f"Gen tokens: {GEN_TOKENS}  |  Prefill chunk: {PREFILL_STEP_SIZE}")
    print(f"Watchdog: {watchdog_log}")
    print("=" * 72)
    sys.stdout.flush()

    if args.mode == "simd":
        from polarquant_metal.integration import make_fused_cache, patch_sdpa
        patch_sdpa()

    print("Loading model...")
    t0 = time.perf_counter()
    model, processor = mlx_lm.load(MODEL_ID)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")
    sys.stdout.flush()

    from mlx_vlm.models.cache import make_prompt_cache

    results = []
    for ctx_len in args.contexts:
        print(f"--- {ctx_len // 1024}K context ---")
        sys.stdout.flush()

        prompt = build_prompt(processor, ctx_len)

        if args.mode == "fp16":
            cache = make_prompt_cache(model)
        else:
            cache = make_fused_cache(model, bits=4, bits_v=4, use_simd=True)

        tps, prompt_tok, gen_tok, kv_mb, fp16_mb = run_path(
            model, processor, prompt, cache, label
        )
        results.append((ctx_len, tps, kv_mb, fp16_mb))

        # Release cache and flush Metal allocator before next run
        del cache
        mx.clear_cache()
        gc.collect()
        time.sleep(3)

    print("\n" + "=" * 72)
    print(f"Summary — {label}")
    print(f"{'ctx':>6}  {'tps':>8}  {'kv_mb':>7}  {'cmpr':>6}")
    print("-" * 36)
    for ctx, tps, kv_mb, fp16_mb in results:
        cmpr = f"{fp16_mb/kv_mb:.1f}x" if kv_mb > 0 else "FP16"
        kv_str = f"{kv_mb:.0f}MB" if kv_mb > 0 else "—"
        print(f"{ctx//1024:>5}K  {tps:>8.1f}  {kv_str:>7}  {cmpr:>6}")
    print("=" * 72)
    print("Done.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
