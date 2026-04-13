"""Long-context sweep benchmark for PolarQuant on Apple Silicon.

Sweeps across multiple context lengths to find the crossover point where
PolarQuant's sparse SV kernel outperforms FP16 KV cache. The sparse path
activates at L_kv > 2048; below that threshold PQ adds overhead without benefit.

Usage:
    # Dry-run with 7B to verify mechanics (fast):
    python benchmarks/bench_long_context_sweep.py \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --draft-model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --context-lengths 2048 --n-tokens 32 --n-runs 1

    # Full 72B sweep (plugged in):
    python benchmarks/bench_long_context_sweep.py \\
        --context-lengths 2048,4096,8192,16384 \\
        --n-tokens 128 --n-runs 1 \\
        --csv benchmarks/results/long_context_sweep_72b.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MAIN_MODEL = "mlx-community/Qwen2.5-72B-Instruct-4bit"
DRAFT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384]

# Seed text repeated to build synthetic long prompts
_SEED_TEXT = (
    "You are a board-certified internist. A 58-year-old male presents with "
    "progressive dyspnea on exertion for 3 months, bilateral ankle edema, "
    "orthopnea, and a new S3 gallop. BNP is 850 pg/mL. Troponin is borderline "
    "at 0.04 ng/mL. CMP within normal limits. Chest X-ray shows cardiomegaly "
    "with pulmonary vascular congestion and Kerley B lines. Echocardiogram "
    "reveals EF of 30 percent, dilated left ventricle, and global hypokinesis. "
    "Provide a step-by-step differential diagnosis and management plan. "
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PolarQuant long-context sweep benchmark")
    p.add_argument("--model", default=MAIN_MODEL)
    p.add_argument("--draft-model", default=DRAFT_MODEL)
    p.add_argument(
        "--context-lengths",
        default=",".join(str(x) for x in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated list of prompt token lengths to sweep",
    )
    p.add_argument("--n-tokens", type=int, default=128,
                   help="Tokens to generate per run")
    p.add_argument("--n-runs", type=int, default=1,
                   help="Runs per configuration (use 1 for thermal safety, 2-3 for accuracy)")
    p.add_argument("--bits", type=int, default=3,
                   help="PolarQuant bits for KV compression")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip FP16-only baseline (saves ~40 percent of run time)")
    p.add_argument("--csv", default=None,
                   help="Optional path to write results CSV")
    return p.parse_args()


def build_prompt(tokenizer, target_len: int) -> str:
    """Build a synthetic prompt of exactly target_len tokens by repeating seed text."""
    base = tokenizer.encode(_SEED_TEXT)
    repeats = (target_len // len(base)) + 2
    tokens = tokenizer.encode((_SEED_TEXT + " ") * repeats)[:target_len]
    return tokenizer.decode(tokens)


def run_one(model, tokenizer, draft_model, cache, prompt: str, n_tokens: int) -> float:
    """Single generation pass; returns generation tok/s."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    gen_kwargs = dict(
        max_tokens=n_tokens,
        sampler=make_sampler(temp=0.0),
        prompt_cache=cache,
    )
    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["num_draft_tokens"] = 4

    last = None
    n = 0
    t0 = time.monotonic()
    for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, **gen_kwargs):
        n += 1
        last = resp
        if resp.finish_reason:
            break
    elapsed = time.monotonic() - t0
    return last.generation_tps if last else (n / elapsed)


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def main() -> None:
    args = parse_args()
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]

    import mlx_lm
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from polarquant_metal.integration import make_fused_cache, patch_sdpa

    print(f"\n{'='*65}")
    print("PolarQuant Long-Context Sweep Benchmark")
    print(f"Main model:      {args.model}")
    print(f"Draft model:     {args.draft_model}")
    print(f"PQ bits:         {args.bits}")
    print(f"Context lengths: {context_lengths}")
    print(f"Generate:        {args.n_tokens} tokens × {args.n_runs} run(s)")
    print(f"{'='*65}\n")

    print("Loading main model...")
    patch_sdpa()
    model, tokenizer = mlx_lm.load(args.model)

    print("Loading draft model...")
    draft_model, _ = mlx_lm.load(args.draft_model)

    rows: list[dict] = []

    for ctx_len in context_lengths:
        prompt = build_prompt(tokenizer, ctx_len)
        actual_len = len(tokenizer.encode(prompt))
        print(f"\n{'─'*65}")
        print(f"Context: {ctx_len} tokens (actual: {actual_len})")
        print(f"{'─'*65}")

        baseline_runs: list[float] = []
        spec_runs: list[float] = []
        pq_runs: list[float] = []

        for run_i in range(args.n_runs):
            suffix = f" (run {run_i+1}/{args.n_runs})" if args.n_runs > 1 else ""

            if not args.skip_baseline:
                cache = make_prompt_cache(model)
                tps = run_one(model, tokenizer, None, cache, prompt, args.n_tokens)
                baseline_runs.append(tps)
                print(f"  Baseline FP16{suffix}: {tps:.1f} tok/s")
                mx.clear_cache()

            cache = make_prompt_cache(model) + make_prompt_cache(draft_model)
            tps = run_one(model, tokenizer, draft_model, cache, prompt, args.n_tokens)
            spec_runs.append(tps)
            print(f"  Spec+FP16{suffix}:     {tps:.1f} tok/s")
            mx.clear_cache()

            cache = make_fused_cache(model, bits=args.bits, boundary_layers=2) + make_prompt_cache(draft_model)
            tps = run_one(model, tokenizer, draft_model, cache, prompt, args.n_tokens)
            pq_runs.append(tps)
            print(f"  Spec+PQ{args.bits}bit{suffix}:  {tps:.1f} tok/s")
            mx.clear_cache()

        row: dict = {"context_tokens": ctx_len}
        if baseline_runs:
            row["baseline_tps"] = round(median(baseline_runs), 1)
        row["spec_fp16_tps"] = round(median(spec_runs), 1)
        row["spec_pq_tps"] = round(median(pq_runs), 1)
        row["pq_vs_spec"] = round(median(pq_runs) / median(spec_runs), 3)
        if baseline_runs:
            row["pq_vs_baseline"] = round(median(pq_runs) / median(baseline_runs), 3)
        rows.append(row)

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Context':>9} | {'Baseline':>8} | {'Spec+FP16':>9} | {'Spec+PQ':>8} | {'PQ/spec':>8}")
    print(f"{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")
    for r in rows:
        base_str = f"{r['baseline_tps']:8.1f}" if "baseline_tps" in r else "    skip"
        crossover = "  ← ✓" if r["pq_vs_spec"] >= 1.0 else ""
        print(
            f"{r['context_tokens']:>9} | {base_str} | "
            f"{r['spec_fp16_tps']:9.1f} | {r['spec_pq_tps']:8.1f} | "
            f"{r['pq_vs_spec']:7.2f}x{crossover}"
        )
    print()

    # CSV output
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
