"""Benchmark speculative decoding + PolarQuant vs baselines on MBP 128GB.

Measures:
    - tok/s for main model only (no spec decoding)
    - tok/s with speculative decoding (7B draft)
    - tok/s with speculative + PolarQuant
    - Speculative acceptance rate

Usage:
    python benchmarks/bench_spec_decoding.py
    python benchmarks/bench_spec_decoding.py --model mlx-community/Qwen2.5-72B-Instruct-4bit
"""

import argparse
import time

MAIN_MODEL = "mlx-community/Qwen2.5-72B-Instruct-4bit"
DRAFT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

TEST_PROMPT = (
    "You are a board-certified internist. A 58-year-old male presents with "
    "progressive dyspnea on exertion for 3 months, bilateral ankle edema, "
    "orthopnea, and a new S3 gallop. BNP is 850 pg/mL. "
    "Provide a step-by-step differential diagnosis and management plan."
)

N_TOKENS = 256
N_RUNS = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MAIN_MODEL)
    p.add_argument("--draft-model", default=DRAFT_MODEL)
    p.add_argument("--n-tokens", type=int, default=N_TOKENS)
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip non-speculative baseline (saves time)")
    return p.parse_args()


def run_generation(model, tokenizer, draft_model, cache, n_tokens, label):
    """Run one generation pass and return (tok/s, n_tokens_generated)."""
    gen_kwargs = dict(max_tokens=n_tokens, temp=0.0, prompt_cache=cache)
    if draft_model:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["num_draft_tokens"] = 4

    import mlx_lm
    t0 = time.monotonic()
    n_gen = 0
    last_resp = None
    for resp in mlx_lm.stream_generate(
        model, tokenizer, prompt=TEST_PROMPT, **gen_kwargs
    ):
        n_gen += 1
        last_resp = resp
        if resp.finish_reason:
            break
    elapsed = time.monotonic() - t0
    tps = last_resp.generation_tps if last_resp else (n_gen / elapsed)
    print(f"  {label}: {tps:.1f} tok/s  ({n_gen} tokens, {elapsed:.1f}s)")
    return tps, n_gen


def main():
    args = parse_args()
    import mlx_lm
    from polarquant_metal.integration import make_fused_cache, patch_sdpa

    print(f"\n{'='*60}")
    print(f"PolarQuant Speculative Decoding Benchmark")
    print(f"Main model:   {args.model}")
    print(f"Draft model:  {args.draft_model}")
    print(f"PQ bits:      {args.bits}")
    print(f"Tokens:       {args.n_tokens} × {args.n_runs} runs")
    print(f"{'='*60}\n")

    print("Loading main model...")
    patch_sdpa()
    model, tokenizer = mlx_lm.load(args.model)

    print("Loading draft model...")
    draft_model, _ = mlx_lm.load(args.draft_model)

    results = {
        "baseline": [],
        "spec_only": [],
        "spec_pq": [],
    }

    for run in range(args.n_runs):
        print(f"\n--- Run {run+1}/{args.n_runs} ---")

        if not args.skip_baseline:
            from mlx_lm import cache as mlx_cache
            baseline_cache = mlx_cache.make_prompt_cache(model)
            tps, _ = run_generation(model, tokenizer, None, baseline_cache,
                                    args.n_tokens, "baseline (no spec, FP16)")
            results["baseline"].append(tps)

        from mlx_lm import cache as mlx_cache
        spec_cache = mlx_cache.make_prompt_cache(model)
        draft_cache = mlx_cache.make_prompt_cache(draft_model)
        combined = spec_cache + draft_cache
        tps, _ = run_generation(model, tokenizer, draft_model, combined,
                                args.n_tokens, "spec decoding (FP16 KV)")
        results["spec_only"].append(tps)

        pq_cache = make_fused_cache(model, bits=args.bits, boundary_layers=2)
        draft_cache2 = mlx_cache.make_prompt_cache(draft_model)
        combined_pq = pq_cache + draft_cache2
        tps, _ = run_generation(model, tokenizer, draft_model, combined_pq,
                                args.n_tokens, f"spec + PolarQuant ({args.bits}-bit KV)")
        results["spec_pq"].append(tps)

    print(f"\n{'='*60}")
    print("SUMMARY (median tok/s)")
    print(f"{'='*60}")

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2

    if results["baseline"]:
        base = median(results["baseline"])
        print(f"Baseline (FP16, no spec):          {base:.1f} tok/s")
    else:
        base = None

    spec = median(results["spec_only"])
    pq = median(results["spec_pq"])
    print(f"Spec decoding (FP16 KV):           {spec:.1f} tok/s", end="")
    if base:
        print(f"  ({spec/base:.2f}x vs baseline)", end="")
    print()
    print(f"Spec + PolarQuant ({args.bits}-bit KV):      {pq:.1f} tok/s", end="")
    if base:
        print(f"  ({pq/base:.2f}x vs baseline)", end="")
    print(f"  ({pq/spec:.2f}x vs spec-only)")
    print()


if __name__ == "__main__":
    main()
