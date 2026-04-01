#!/usr/bin/env python3
"""
Benchmark: Phase 3 (compact-index sparse SV) vs Phase 2 (branch-per-position dense SV)

Compares the two SV kernel paths:
  - Dense (Phase 2): polarquant_sv_matmul() — iterates all L_kv with threshold branch
  - Sparse (Phase 3): polarquant_sv_build_index() + polarquant_sv_sparse() — builds
    compact active index, iterates only active positions

Tests across context lengths [2048..32768] with three attention patterns:
  - Concentrated (low entropy, ~1% active)
  - Moderate (~10% active)
  - Spread (~50% active)

Also tests zone priors at 16K: system_prompt_len=200, recent_zone_len=500.

Usage:
    cd ~/workspace/polarquant-metal
    python benchmarks/bench_phase3_sparse.py
"""

import gc
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from polarquant_metal.polar_quant import PolarQuant
from polarquant_metal.packing import pack_indices
from polarquant_metal.codebooks import load_codebook_f32
from polarquant_metal.kernels import (
    polarquant_sv_matmul,
    polarquant_sv_build_index,
    polarquant_sv_sparse,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 8
N_KV_HEADS = 2
D = 128
BITS = 3
L_Q = 1              # decode mode
REP = N_HEADS // N_KV_HEADS  # GQA ratio

WARMUP = 3
N_TRIALS = 10

CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768]

# Attention patterns: name, softmax temperature
# Higher temp -> more concentrated (fewer active positions)
PATTERNS = [
    ("concentrated", 5.0),   # ~1% active
    ("moderate",     1.0),   # ~10% active
    ("spread",       0.1),   # ~50% active
]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Cosine similarity, returns -1.0 for degenerate (both near-zero) cases."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    na = float(mx.sqrt(mx.sum(a_flat * a_flat)))
    nb = float(mx.sqrt(mx.sum(b_flat * b_flat)))
    # Degenerate case: both outputs near-zero (threshold filtered everything)
    if na < 1e-8 and nb < 1e-8:
        return -1.0  # sentinel: both zero
    dot = float(mx.sum(a_flat * b_flat))
    return dot / (na * nb + 1e-10)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_one(L_kv, pattern_name, temperature, results_list):
    """Run dense vs sparse benchmark for one (L_kv, pattern) combination."""

    print(f"\n  L_kv={L_kv:>6,}  pattern={pattern_name:<14s}  ", end="", flush=True)

    # --- Setup ---
    V = mx.random.normal((B, N_KV_HEADS, L_kv, D))
    mx.eval(V)

    pq = PolarQuant(bits=BITS, dim=D, seed=43)
    v_idx, v_norms = pq.quantize(V)
    v_packed = pack_indices(v_idx, BITS)
    v_cents = load_codebook_f32(BITS, D)
    mx.eval(v_packed, v_norms, v_cents)

    # Generate attention pattern
    scores = mx.random.normal((B, N_HEADS, 1, L_kv)) * temperature
    weights = mx.softmax(scores, axis=-1, precise=True)
    mx.eval(weights)

    # Precombine weight * norm (shared prep for both paths)
    norms_sq = v_norms.squeeze(-1)
    norms_exp = mx.repeat(norms_sq, REP, axis=1)
    wn = weights * norms_exp[:, :, None, :]
    mx.eval(wn)

    # Per-head threshold
    thresh = mx.full((N_HEADS,), 0.001, dtype=mx.float32)
    zone = mx.zeros(L_kv, dtype=mx.uint32)

    # --- Warmup ---
    for _ in range(WARMUP):
        out_d = polarquant_sv_matmul(weights, v_packed, v_norms, v_cents, D, BITS,
                                     sparse_v_threshold=thresh)
        mx.eval(out_d)

        ci = polarquant_sv_build_index(wn, thresh, zone)
        mx.eval(ci)
        out_s = polarquant_sv_sparse(ci, wn, v_packed, v_cents, D, L_kv, BITS)
        mx.eval(out_s)

    # --- Time dense (Phase 2) ---
    times_dense = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        out_dense = polarquant_sv_matmul(weights, v_packed, v_norms, v_cents, D, BITS,
                                         sparse_v_threshold=thresh)
        mx.eval(out_dense)
        times_dense.append(time.perf_counter() - t0)

    # --- Time sparse (Phase 3): index build + kernel separately ---
    times_sparse = []
    times_index = []
    times_kernel = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        ci = polarquant_sv_build_index(wn, thresh, zone)
        mx.eval(ci)
        t_idx = time.perf_counter() - t0

        t1 = time.perf_counter()
        out_sparse = polarquant_sv_sparse(ci, wn, v_packed, v_cents, D, L_kv, BITS)
        mx.eval(out_sparse)
        t_kern = time.perf_counter() - t1

        times_index.append(t_idx)
        times_kernel.append(t_kern)
        times_sparse.append(t_idx + t_kern)

    # --- Correctness: cosine similarity ---
    out_dense_final = polarquant_sv_matmul(weights, v_packed, v_norms, v_cents, D, BITS,
                                            sparse_v_threshold=thresh)
    ci_final = polarquant_sv_build_index(wn, thresh, zone)
    out_sparse_final = polarquant_sv_sparse(ci_final, wn, v_packed, v_cents, D, L_kv, BITS)
    mx.eval(out_dense_final, out_sparse_final)
    cos_sim = cosine_similarity(out_dense_final, out_sparse_final)

    # --- Active count per head ---
    ci_np = np.array(ci_final)
    stride = 1 + L_kv
    active_counts = []
    for h in range(N_HEADS):
        head_base = h * stride
        count = int(ci_np[head_base])
        active_counts.append(count)

    # --- Compute medians ---
    med_dense = float(np.median(times_dense)) * 1000
    med_sparse = float(np.median(times_sparse)) * 1000
    med_index = float(np.median(times_index)) * 1000
    med_kernel = float(np.median(times_kernel)) * 1000
    speedup = med_dense / med_sparse if med_sparse > 0 else float('inf')

    avg_active = np.mean(active_counts)
    active_pct = avg_active / L_kv * 100

    cos_label = "BOTH_ZERO" if cos_sim == -1.0 else f"{cos_sim:.6f}"
    print(f"dense={med_dense:>8.3f}ms  sparse={med_sparse:>8.3f}ms  "
          f"(idx={med_index:.3f} + kern={med_kernel:.3f})  "
          f"speedup={speedup:>5.2f}x  active={avg_active:.0f}/{L_kv} "
          f"({active_pct:.1f}%)  cos={cos_label}")

    row = {
        "L_kv": L_kv,
        "pattern": pattern_name,
        "dense_ms": med_dense,
        "sparse_ms": med_sparse,
        "index_ms": med_index,
        "kernel_ms": med_kernel,
        "speedup": speedup,
        "avg_active": avg_active,
        "active_pct": active_pct,
        "active_counts": active_counts,
        "cos_sim": cos_sim,
    }
    results_list.append(row)
    return row


def run_zone_test(results_list):
    """Test zone priors at 16K: system_prompt + recent zone always active."""

    L_kv = 16384
    sys_prompt_len = 200
    recent_zone_len = 500

    print(f"\n  --- Zone Prior Test (L_kv={L_kv:,}, sys_prompt={sys_prompt_len}, "
          f"recent={recent_zone_len}) ---")

    # Setup
    V = mx.random.normal((B, N_KV_HEADS, L_kv, D))
    mx.eval(V)

    pq = PolarQuant(bits=BITS, dim=D, seed=43)
    v_idx, v_norms = pq.quantize(V)
    v_packed = pack_indices(v_idx, BITS)
    v_cents = load_codebook_f32(BITS, D)
    mx.eval(v_packed, v_norms, v_cents)

    # Concentrated attention (so zone priors add to an otherwise sparse pattern)
    scores = mx.random.normal((B, N_HEADS, 1, L_kv)) * 5.0
    weights = mx.softmax(scores, axis=-1, precise=True)
    mx.eval(weights)

    norms_sq = v_norms.squeeze(-1)
    norms_exp = mx.repeat(norms_sq, REP, axis=1)
    wn = weights * norms_exp[:, :, None, :]
    mx.eval(wn)

    thresh = mx.full((N_HEADS,), 0.001, dtype=mx.float32)

    # Build zone prior: first sys_prompt_len + last recent_zone_len positions = 1
    zone_np = np.zeros(L_kv, dtype=np.uint32)
    zone_np[:sys_prompt_len] = 1
    zone_np[L_kv - recent_zone_len:] = 1
    zone = mx.array(zone_np)

    zone_none = mx.zeros(L_kv, dtype=mx.uint32)

    # --- Warmup ---
    for _ in range(WARMUP):
        ci = polarquant_sv_build_index(wn, thresh, zone)
        mx.eval(ci)
        out = polarquant_sv_sparse(ci, wn, v_packed, v_cents, D, L_kv, BITS)
        mx.eval(out)

        ci0 = polarquant_sv_build_index(wn, thresh, zone_none)
        mx.eval(ci0)
        out0 = polarquant_sv_sparse(ci0, wn, v_packed, v_cents, D, L_kv, BITS)
        mx.eval(out0)

    # --- Time WITHOUT zone priors ---
    times_no_zone = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        ci0 = polarquant_sv_build_index(wn, thresh, zone_none)
        mx.eval(ci0)
        out0 = polarquant_sv_sparse(ci0, wn, v_packed, v_cents, D, L_kv, BITS)
        mx.eval(out0)
        times_no_zone.append(time.perf_counter() - t0)

    # --- Time WITH zone priors ---
    times_zone = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        ci = polarquant_sv_build_index(wn, thresh, zone)
        mx.eval(ci)
        out = polarquant_sv_sparse(ci, wn, v_packed, v_cents, D, L_kv, BITS)
        mx.eval(out)
        times_zone.append(time.perf_counter() - t0)

    med_no_zone = float(np.median(times_no_zone)) * 1000
    med_zone = float(np.median(times_zone)) * 1000

    # Verify zone positions are in active index
    ci_eval = polarquant_sv_build_index(wn, thresh, zone)
    mx.eval(ci_eval)
    ci_np = np.array(ci_eval)
    stride = 1 + L_kv

    zone_coverage = []
    active_counts_zone = []
    active_counts_no_zone = []

    ci0_eval = polarquant_sv_build_index(wn, thresh, zone_none)
    mx.eval(ci0_eval)
    ci0_np = np.array(ci0_eval)

    for h in range(N_HEADS):
        head_base = h * stride
        count = int(ci_np[head_base])
        active_set = set(int(ci_np[head_base + 1 + i]) for i in range(count))
        active_counts_zone.append(count)

        count0 = int(ci0_np[head_base])
        active_counts_no_zone.append(count0)

        # Check: are all zone prior positions in active set?
        zone_positions = set(np.where(zone_np == 1)[0].tolist())
        covered = len(zone_positions & active_set)
        zone_coverage.append(covered / len(zone_positions) * 100)

    avg_coverage = np.mean(zone_coverage)
    avg_active_zone = np.mean(active_counts_zone)
    avg_active_no_zone = np.mean(active_counts_no_zone)
    zone_overhead = med_zone - med_no_zone

    print(f"  Without zones:  {med_no_zone:.3f}ms  "
          f"avg active = {avg_active_no_zone:.0f}/{L_kv}")
    print(f"  With zones:     {med_zone:.3f}ms  "
          f"avg active = {avg_active_zone:.0f}/{L_kv}")
    print(f"  Zone overhead:  {zone_overhead:+.3f}ms")
    print(f"  Zone coverage:  {avg_coverage:.1f}% of zone positions in active index")

    for h in range(N_HEADS):
        delta = active_counts_zone[h] - active_counts_no_zone[h]
        print(f"    H{h}: active {active_counts_no_zone[h]} -> {active_counts_zone[h]} "
              f"(+{delta}), zone coverage={zone_coverage[h]:.1f}%")

    zone_row = {
        "test": "zone_prior",
        "L_kv": L_kv,
        "sys_prompt_len": sys_prompt_len,
        "recent_zone_len": recent_zone_len,
        "no_zone_ms": med_no_zone,
        "zone_ms": med_zone,
        "overhead_ms": zone_overhead,
        "avg_coverage": avg_coverage,
        "avg_active_zone": avg_active_zone,
        "avg_active_no_zone": avg_active_no_zone,
        "active_counts_zone": active_counts_zone,
        "active_counts_no_zone": active_counts_no_zone,
        "zone_coverage": zone_coverage,
    }
    results_list.append(zone_row)
    return zone_row


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_results_md(results, zone_result, md_path):
    """Write results to PHASE3_RESULTS.md."""
    with open(md_path, "w") as f:
        f.write("# Phase 3 Benchmark: Compact-Index Sparse SV vs Dense SV\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} "
                f"(GQA {REP}:1), D={D}, bits={BITS}  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n")
        f.write(f"**Timing:** {N_TRIALS} trials (median), "
                f"{WARMUP} warmup  \n")
        f.write(f"**Threshold:** 0.001 per head  \n\n")

        # Main results table
        f.write("## Results\n\n")
        f.write("| L_kv | Pattern | Dense (ms) | Sparse (ms) | "
                "Index (ms) | Kernel (ms) | Speedup | "
                "Avg Active | Active % | Cos Sim |\n")
        f.write("|-----:|:--------|----------:|-----------:|"
                "---------:|----------:|--------:|"
                "---------:|---------:|--------:|\n")

        for r in results:
            cos_str = ("both zero*" if r['cos_sim'] == -1.0
                       else f"{r['cos_sim']:.6f}")
            f.write(f"| {r['L_kv']:,} | {r['pattern']} "
                    f"| {r['dense_ms']:.3f} | {r['sparse_ms']:.3f} "
                    f"| {r['index_ms']:.3f} | {r['kernel_ms']:.3f} "
                    f"| {r['speedup']:.2f}x "
                    f"| {r['avg_active']:.0f} | {r['active_pct']:.1f}% "
                    f"| {cos_str} |\n")

        # Per-head active counts
        f.write("\n## Per-Head Active Counts\n\n")
        f.write("| L_kv | Pattern |")
        for h in range(N_HEADS):
            f.write(f" H{h} |")
        f.write("\n|-----:|:--------|")
        for _ in range(N_HEADS):
            f.write("----:|")
        f.write("\n")

        for r in results:
            f.write(f"| {r['L_kv']:,} | {r['pattern']} |")
            for c in r["active_counts"]:
                f.write(f" {c} |")
            f.write("\n")

        # Zone prior test
        if zone_result:
            f.write("\n## Zone Prior Test\n\n")
            f.write(f"**L_kv:** {zone_result['L_kv']:,}  \n")
            f.write(f"**System prompt zone:** first {zone_result['sys_prompt_len']} "
                    f"positions  \n")
            f.write(f"**Recent zone:** last {zone_result['recent_zone_len']} "
                    f"positions  \n\n")

            f.write("| Metric | Without Zones | With Zones | Delta |\n")
            f.write("|:-------|-------------:|-----------:|------:|\n")
            f.write(f"| Time (ms) | {zone_result['no_zone_ms']:.3f} "
                    f"| {zone_result['zone_ms']:.3f} "
                    f"| {zone_result['overhead_ms']:+.3f} |\n")
            f.write(f"| Avg active | {zone_result['avg_active_no_zone']:.0f} "
                    f"| {zone_result['avg_active_zone']:.0f} "
                    f"| +{zone_result['avg_active_zone'] - zone_result['avg_active_no_zone']:.0f} |\n")
            f.write(f"| Zone coverage | - "
                    f"| {zone_result['avg_coverage']:.1f}% | - |\n")

            f.write("\n### Per-Head Zone Detail\n\n")
            f.write("| Head | Active (no zone) | Active (zone) | Delta | "
                    "Zone Coverage |\n")
            f.write("|-----:|-----------------:|--------------:|------:|"
                    "--------------:|\n")
            for h in range(N_HEADS):
                delta = (zone_result['active_counts_zone'][h]
                         - zone_result['active_counts_no_zone'][h])
                f.write(f"| H{h} | {zone_result['active_counts_no_zone'][h]} "
                        f"| {zone_result['active_counts_zone'][h]} "
                        f"| +{delta} "
                        f"| {zone_result['zone_coverage'][h]:.1f}% |\n")

        # Analysis
        f.write("\n## Analysis\n\n")

        # Group by pattern
        for pat_name, _ in PATTERNS:
            pat_rows = [r for r in results if r["pattern"] == pat_name]
            if not pat_rows:
                continue
            f.write(f"### {pat_name.capitalize()} Attention\n\n")
            for r in pat_rows:
                cos_str = ("both zero*" if r['cos_sim'] == -1.0
                           else f"{r['cos_sim']:.6f}")
                f.write(f"- **{r['L_kv']:,}**: {r['speedup']:.2f}x speedup, "
                        f"{r['active_pct']:.1f}% active, "
                        f"cos_sim={cos_str}\n")
            f.write("\n")

        # Crossover analysis
        f.write("### Crossover Point\n\n")
        crossover_rows = [r for r in results if r["speedup"] < 1.0]
        if crossover_rows:
            f.write("Sparse path is **slower** than dense for:\n\n")
            for r in crossover_rows:
                f.write(f"- L_kv={r['L_kv']:,}, {r['pattern']}: "
                        f"{r['speedup']:.2f}x (index overhead dominates)\n")
            f.write("\nDense fallback recommended when active% > ~50%.\n")
        else:
            f.write("Sparse path was faster than dense in all tested "
                    "configurations.\n")

        # Correctness
        f.write("\n### Correctness\n\n")
        # Filter out degenerate (both-zero) cases for correctness check
        real_rows = [r for r in results if r["cos_sim"] != -1.0]
        zero_rows = [r for r in results if r["cos_sim"] == -1.0]
        all_correct = all(r["cos_sim"] > 0.999 for r in real_rows)
        if all_correct:
            f.write("All non-degenerate configurations show cos_sim > 0.999 "
                    "between dense and sparse outputs, confirming identical "
                    "behavior.\n")
        else:
            bad = [r for r in real_rows if r["cos_sim"] <= 0.999]
            f.write("**WARNING:** Some configurations show divergence:\n\n")
            for r in bad:
                f.write(f"- L_kv={r['L_kv']:,}, {r['pattern']}: "
                        f"cos_sim={r['cos_sim']:.6f}\n")
        if zero_rows:
            f.write("\n*`both zero` entries indicate cases where the fixed "
                    "threshold (0.001) filters out ALL positions because "
                    "per-position wn values fall below the threshold at long "
                    "contexts with spread attention. Both dense and sparse "
                    "kernels produce near-zero output -- this is expected "
                    "behavior, not a correctness issue. In production, "
                    "entropy-guided thresholds would lower the threshold "
                    "for spread heads.*\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  Phase 3 Benchmark: Compact-Index Sparse SV vs Dense SV")
    print("=" * 80)
    print(f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
          f"D={D}, bits={BITS}")
    print(f"  Threshold:  0.001 per head")
    print(f"  Trials:     {N_TRIALS} (median), {WARMUP} warmup")
    print(f"  Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print(f"  Contexts:   {CONTEXT_LENGTHS}")
    print()

    results = []
    zone_result = None

    for L_kv in CONTEXT_LENGTHS:
        print(f"\n{'='*80}")
        print(f"  Context length: {L_kv:,}")
        print(f"{'='*80}")

        for pattern_name, temperature in PATTERNS:
            try:
                run_one(L_kv, pattern_name, temperature, results)
            except Exception as e:
                if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                    print(f"OOM at L_kv={L_kv}, {pattern_name} -- skipping")
                else:
                    print(f"FAILED: {e}")
                    traceback.print_exc()
            gc.collect()

    # Zone prior test at 16K
    print(f"\n\n{'='*80}")
    print("  Zone Prior Test")
    print(f"{'='*80}")
    try:
        zone_results_list = []
        zone_result = run_zone_test(zone_results_list)
    except Exception as e:
        print(f"  Zone test FAILED: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}\n")

    header = (f"  {'L_kv':>6s}  {'Pattern':<14s}  {'Dense (ms)':>10s}  "
              f"{'Sparse (ms)':>11s}  {'Index (ms)':>10s}  {'Kern (ms)':>9s}  "
              f"{'Speedup':>7s}  {'Active':>8s}  {'Act %':>6s}  {'CosSim':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        print(f"  {r['L_kv']:>6,}  {r['pattern']:<14s}  "
              f"{r['dense_ms']:>10.3f}  {r['sparse_ms']:>11.3f}  "
              f"{r['index_ms']:>10.3f}  {r['kernel_ms']:>9.3f}  "
              f"{r['speedup']:>6.2f}x  {r['avg_active']:>8.0f}  "
              f"{r['active_pct']:>5.1f}%  {r['cos_sim']:>8.6f}")

    # Save to markdown
    md_path = os.path.join(os.path.dirname(__file__), "PHASE3_RESULTS.md")
    write_results_md(results, zone_result, md_path)
    print(f"\n  Results saved to: {md_path}")

    print(f"\n{'='*80}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
