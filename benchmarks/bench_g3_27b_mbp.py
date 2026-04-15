#!/usr/bin/env python3
"""
Gemma 3 27B-IT benchmark for M5 Max MBP — FP16 vs PQ-simd.

Why Gemma 3 27B at 64K:
  - Full attention (no sliding window) — all 62 layers' KV scales with seq_len
  - At 64K context, KV is ~42% of memory bandwidth (vs ~14% for Gemma 4 31B at 32K)
  - KV > weights crossover at ~95K tokens — 64K is clearly mixed/KV-trending
  - 128GB M5 Max has full headroom for FP16 at all context lengths

Run each mode as a separate process (no Metal buffer carryover):

    cd /tmp/polarquant-metal

    # FP16 baseline
    nohup .venv-mbp/bin/python3 benchmarks/bench_g3_27b_mbp.py \\
        --mode fp16 > /tmp/bench_g3_fp16.log 2>&1 &

    # PQ-simd (4-bit KV)
    nohup .venv-mbp/bin/python3 benchmarks/bench_g3_27b_mbp.py \\
        --mode simd > /tmp/bench_g3_simd.log 2>&1 &

    tail -f /tmp/bench_g3_simd.log
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx_lm

# Hard-cap Metal wired memory before model load to prevent IOGPU kernel panics.
# (MLX 0.31+ APIs — ref: IOGPU.kext "completeMemory() prepare count underflow" bug)
# On M5 Max 128GB: ~70%/15%/65% → 89.6 / 19.2 / 83.2 GB.
# set_wired_limit raises if the value exceeds max_recommended_working_set_size,
# so this also acts as a sanity-check that MLX/macOS agree on available RAM.
_total_ram = mx.device_info()["memory_size"]
mx.set_memory_limit(int(_total_ram * 0.70))
mx.set_cache_limit(int(_total_ram * 0.15))
mx.set_wired_limit(int(_total_ram * 0.65))

from benchmarks.watchdog import start_watchdog

MODEL_ID = "mlx-community/gemma-3-27b-it-4bit"
GEN_TOKENS = 256          # long generation = stable tps deep in KV-dominant regime
PREFILL_STEP_SIZE = 512   # chunked prefill
CONTEXT_LENGTHS = [32_768, 65_536, 131_072]

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
        tokens = tokens[: target_tokens - 20]
        body = tokenizer.decode(tokens)
    return body + "\n\nSummarize the key insight in one sentence:"


def run_path(model, tokenizer, prompt, cache, label: str):
    """Run generation, return (tps, prompt_tokens, gen_tokens, kv_mb, fp16_mb)."""
    from polarquant_metal.turboquant_cache import TurboQuantKVCache

    last = None
    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=GEN_TOKENS,
        prompt_cache=cache,
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        last = resp

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
                2 * nh * ref.offset * ref._head_dim * 2 / 1024 / 1024
                * len(tq_caches)
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
    parser = argparse.ArgumentParser(description="Gemma 3 27B benchmark — MBP")
    parser.add_argument("--mode", choices=["fp16", "simd"], required=True)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
    )
    args = parser.parse_args()

    label = "FP16" if args.mode == "fp16" else "PQ-simd"
    watchdog_log = f"/tmp/bench_g3_watchdog_{args.mode}.log"

    # Wired threshold 100GB — M5 Max 128GB has plenty of headroom
    import benchmarks.watchdog as wd
    wd.KILL_ON_GPU_GB = 100.0
    start_watchdog(os.getpid(), log_file=watchdog_log)

    print("=" * 72)
    print(f"Gemma 3 27B — {label} — M5 Max MBP")
    print(f"Model: {MODEL_ID}")
    print(f"Contexts: {[f'{c//1024}K' for c in args.contexts]}")
    print(f"Gen tokens: {GEN_TOKENS}  |  Prefill chunk: {PREFILL_STEP_SIZE}")
    print(f"Watchdog: {watchdog_log} (kill at 100GB wired)")
    print("=" * 72)
    sys.stdout.flush()

    if args.mode == "simd":
        from polarquant_metal.integration import patch_sdpa
        patch_sdpa()

    print("Loading model...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")
    sys.stdout.flush()

    from mlx_lm.models.cache import make_prompt_cache

    results = []
    for ctx_len in args.contexts:
        print(f"--- {ctx_len // 1024}K context ---")
        sys.stdout.flush()

        prompt = build_prompt(tokenizer, ctx_len)

        if args.mode == "fp16":
            cache = make_prompt_cache(model)
        else:
            from polarquant_metal.integration import make_fused_cache
            cache = make_fused_cache(model, bits=4, bits_v=4, use_simd=True)

        tps, prompt_tok, gen_tok, kv_mb, fp16_mb = run_path(
            model, tokenizer, prompt, cache, label
        )
        results.append((ctx_len, tps, kv_mb, fp16_mb))

        del cache
        mx.clear_cache()
        gc.collect()
        time.sleep(3)

    print("\n" + "=" * 72)
    print(f"Summary — {label}  ({MODEL_ID})")
    print(f"{'ctx':>7}  {'tps':>8}  {'kv_mb':>9}  {'cmpr':>6}")
    print("-" * 38)
    for ctx, tps, kv_mb, fp16_mb in results:
        cmpr = f"{fp16_mb/kv_mb:.1f}x" if kv_mb > 0 else "FP16"
        kv_str = f"{kv_mb:.0f}MB" if kv_mb > 0 else "—"
        print(f"{ctx//1024:>6}K  {tps:>8.1f}  {kv_str:>9}  {cmpr:>6}")
    print("=" * 72)
    print("Done.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
