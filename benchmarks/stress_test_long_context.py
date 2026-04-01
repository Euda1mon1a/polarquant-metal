#!/usr/bin/env python3
"""
PolarQuant Metal -- Long-Context Stress Test (16K-32K decode)

Tests fused Metal kernel performance at decode (L_q=1) for context lengths
far beyond the existing 8K benchmark ceiling:

    [4096, 8192, 12288, 16384, 24576, 32768]

Metrics per context length:
  - FP16 baseline time (ms)
  - Naive dequantize-then-matmul time (ms)
  - Fused Metal kernel time (ms)
  - Speedup ratios (fused vs FP16, fused vs naive)
  - KV cache memory: FP16 vs 3-bit PolarQuant (MB)
  - Numerical accuracy: cosine similarity between fused and FP16 outputs
  - Per-operation breakdown: QK kernel, SV kernel, softmax, rotations

Also tests sparse_v_threshold impact at 16K context.

Usage:
    cd ~/workspace/polarquant-metal
    python benchmarks/stress_test_long_context.py
"""

import gc
import math
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
from polarquant_metal.kernels import polarquant_qk_matmul, polarquant_sv_matmul
from polarquant_metal.cache import FusedPolarQuantKVCache

# ---------------------------------------------------------------------------
# Configuration -- Qwen3.5-35B-A3B dimensions
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 32        # query heads
N_KV_HEADS = 8      # GQA kv heads
D = 128              # head_dim
BITS = 3             # 3-bit PolarQuant (production config)
N_TRIALS = 10        # median of N trials
WARMUP = 3           # warmup iterations before timing

CONTEXT_LENGTHS = [4096, 8192, 12288, 16384, 24576, 32768]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    return float(dot / (norm_a * norm_b + 1e-10))


def estimate_memory_mb(B, n_kv_heads, L_kv, D, bits):
    """Estimate KV cache memory for FP16 vs PolarQuant (both K and V)."""
    fp16_bytes = 2 * B * n_kv_heads * L_kv * D * 2  # keys + values, 2 bytes each
    vals_per_int = 32 // bits
    D_packed = (D + vals_per_int - 1) // vals_per_int
    # packed indices (uint32) + norms (float32) for both K and V
    pq_bytes = 2 * B * n_kv_heads * L_kv * (D_packed * 4 + 4)
    return fp16_bytes / (1024 * 1024), pq_bytes / (1024 * 1024)


def time_op(fn, warmup_n=WARMUP, trials_n=N_TRIALS):
    """Time an operation with warmup, return median seconds."""
    for _ in range(warmup_n):
        r = fn()
        mx.eval(r)
    times = []
    for _ in range(trials_n):
        t0 = time.perf_counter()
        r = fn()
        mx.eval(r)
        times.append(time.perf_counter() - t0)
    return np.median(times)


# ---------------------------------------------------------------------------
# Benchmark: FP16 baseline
# ---------------------------------------------------------------------------

def bench_fp16(queries, keys_raw, values_raw):
    """FP16 attention baseline (no quantization). Returns (time_s, output)."""
    scale = 1.0 / math.sqrt(D)
    rep = N_HEADS // N_KV_HEADS

    if rep > 1:
        keys_exp = mx.repeat(keys_raw, rep, axis=1)
        values_exp = mx.repeat(values_raw, rep, axis=1)
    else:
        keys_exp, values_exp = keys_raw, values_raw

    # Warmup
    for _ in range(WARMUP):
        s = (queries @ mx.swapaxes(keys_exp, -2, -1)) * scale
        w = mx.softmax(s, axis=-1, precise=True)
        o = w @ values_exp
        mx.eval(o)

    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        s = (queries @ mx.swapaxes(keys_exp, -2, -1)) * scale
        w = mx.softmax(s, axis=-1, precise=True)
        o = w @ values_exp
        mx.eval(o)
        times.append(time.perf_counter() - t0)

    return np.median(times), o


# ---------------------------------------------------------------------------
# Benchmark: Naive dequantize-then-matmul
# ---------------------------------------------------------------------------

def bench_naive(queries, keys_raw, values_raw):
    """Naive PolarQuant path. Returns (time_s, output)."""
    scale = 1.0 / math.sqrt(D)
    rep = N_HEADS // N_KV_HEADS

    pq_key = PolarQuant(bits=BITS, dim=D, seed=42)
    pq_val = PolarQuant(bits=BITS, dim=D, seed=43)
    k_indices, k_norms = pq_key.quantize(keys_raw)
    v_indices, v_norms = pq_val.quantize(values_raw)

    # Warmup
    for _ in range(WARMUP):
        kd = pq_key.dequantize(k_indices, k_norms)
        vd = pq_val.dequantize(v_indices, v_norms)
        if rep > 1:
            kd = mx.repeat(kd, rep, axis=1)
            vd = mx.repeat(vd, rep, axis=1)
        s = (queries @ mx.swapaxes(kd, -2, -1)) * scale
        w = mx.softmax(s, axis=-1, precise=True)
        o = w @ vd
        mx.eval(o)

    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        kd = pq_key.dequantize(k_indices, k_norms)
        vd = pq_val.dequantize(v_indices, v_norms)
        if rep > 1:
            kd = mx.repeat(kd, rep, axis=1)
            vd = mx.repeat(vd, rep, axis=1)
        s = (queries @ mx.swapaxes(kd, -2, -1)) * scale
        w = mx.softmax(s, axis=-1, precise=True)
        o = w @ vd
        mx.eval(o)
        times.append(time.perf_counter() - t0)

    return np.median(times), o


# ---------------------------------------------------------------------------
# Benchmark: Fused Metal kernel
# ---------------------------------------------------------------------------

def bench_fused(queries, keys_raw, values_raw, sparse_v_threshold=0.0):
    """Fused Metal kernel path. Returns (time_s, output)."""
    cache = FusedPolarQuantKVCache(bits=BITS, head_dim=D, key_seed=42, value_seed=43)
    cache.update_and_fetch(keys_raw, values_raw)

    # Warmup
    for _ in range(WARMUP):
        o = cache.fused_attention(queries)
        mx.eval(o)

    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        o = cache.fused_attention(queries)
        mx.eval(o)
        times.append(time.perf_counter() - t0)

    return np.median(times), o


# ---------------------------------------------------------------------------
# Per-operation breakdown using fused primitives directly
# ---------------------------------------------------------------------------

def profile_ops(queries, keys_raw, values_raw, sparse_v_threshold=0.0):
    """Profile individual fused attention operations. Returns dict of times."""
    scale = 1.0 / math.sqrt(D)

    pq_key = PolarQuant(bits=BITS, dim=D, seed=42)
    pq_val = PolarQuant(bits=BITS, dim=D, seed=43)

    k_idx, k_norms = pq_key.quantize(keys_raw)
    v_idx, v_norms = pq_val.quantize(values_raw)
    k_packed = pack_indices(k_idx, BITS)
    v_packed = pack_indices(v_idx, BITS)

    key_centroids = load_codebook_f32(BITS, D)
    val_centroids = load_codebook_f32(BITS, D)
    key_rotation_t = pq_key.rotation_t
    val_rotation = pq_val.rotation

    mx.eval(queries, k_packed, k_norms, v_packed, v_norms,
            key_centroids, val_centroids, key_rotation_t, val_rotation)

    # Pre-compute intermediates
    q_rot = queries @ key_rotation_t
    mx.eval(q_rot)
    scores = polarquant_qk_matmul(q_rot, k_packed, k_norms, key_centroids, scale, BITS)
    mx.eval(scores)
    weights = mx.softmax(scores, axis=-1, precise=True)
    mx.eval(weights)
    out_rot = polarquant_sv_matmul(
        weights, v_packed, v_norms, val_centroids, D, BITS,
        sparse_v_threshold=sparse_v_threshold,
    )
    mx.eval(out_rot)

    ops = {}
    ops["Q rotation"] = time_op(lambda: queries @ key_rotation_t)
    ops["QK kernel"] = time_op(
        lambda: polarquant_qk_matmul(q_rot, k_packed, k_norms, key_centroids, scale, BITS)
    )
    ops["Softmax"] = time_op(lambda: mx.softmax(scores, axis=-1, precise=True))
    ops["SV kernel"] = time_op(
        lambda: polarquant_sv_matmul(
            weights, v_packed, v_norms, val_centroids, D, BITS,
            sparse_v_threshold=sparse_v_threshold,
        )
    )
    ops["Out rotation"] = time_op(lambda: out_rot @ val_rotation)

    # Full pipeline without eval barriers
    def full():
        qr = queries @ key_rotation_t
        sc = polarquant_qk_matmul(qr, k_packed, k_norms, key_centroids, scale, BITS)
        wt = mx.softmax(sc, axis=-1, precise=True)
        or_ = polarquant_sv_matmul(
            wt, v_packed, v_norms, val_centroids, D, BITS,
            sparse_v_threshold=sparse_v_threshold,
        )
        return or_ @ val_rotation

    ops["Full pipeline"] = time_op(full)
    return ops


# ---------------------------------------------------------------------------
# Sparse V threshold sweep at a fixed context
# ---------------------------------------------------------------------------

def bench_sparse_sweep(L_kv, thresholds):
    """Benchmark different sparse_v_threshold values at a fixed context length."""
    np.random.seed(42)
    queries = mx.array(np.random.randn(B, N_HEADS, 1, D).astype(np.float32))
    keys_raw = mx.array(np.random.randn(B, N_KV_HEADS, L_kv, D).astype(np.float32))
    values_raw = mx.array(np.random.randn(B, N_KV_HEADS, L_kv, D).astype(np.float32))
    mx.eval(queries, keys_raw, values_raw)

    # Compute FP16 reference output for accuracy comparison
    scale = 1.0 / math.sqrt(D)
    rep = N_HEADS // N_KV_HEADS
    keys_exp = mx.repeat(keys_raw, rep, axis=1) if rep > 1 else keys_raw
    values_exp = mx.repeat(values_raw, rep, axis=1) if rep > 1 else values_raw
    s = (queries @ mx.swapaxes(keys_exp, -2, -1)) * scale
    w = mx.softmax(s, axis=-1, precise=True)
    fp16_out = w @ values_exp
    mx.eval(fp16_out)

    results = []
    for threshold in thresholds:
        # Use direct kernel calls with sparse_v_threshold
        pq_key = PolarQuant(bits=BITS, dim=D, seed=42)
        pq_val = PolarQuant(bits=BITS, dim=D, seed=43)
        k_idx, k_norms = pq_key.quantize(keys_raw)
        v_idx, v_norms = pq_val.quantize(values_raw)
        k_packed = pack_indices(k_idx, BITS)
        v_packed = pack_indices(v_idx, BITS)
        key_centroids = load_codebook_f32(BITS, D)
        val_centroids = load_codebook_f32(BITS, D)
        mx.eval(k_packed, k_norms, v_packed, v_norms)

        q_rot = queries @ pq_key.rotation_t
        mx.eval(q_rot)

        def run_with_thresh(t=threshold):
            qr = queries @ pq_key.rotation_t
            sc = polarquant_qk_matmul(qr, k_packed, k_norms, key_centroids, scale, BITS)
            wt = mx.softmax(sc, axis=-1, precise=True)
            or_ = polarquant_sv_matmul(
                wt, v_packed, v_norms, val_centroids, D, BITS,
                sparse_v_threshold=t,
            )
            return or_ @ pq_val.rotation

        t_med = time_op(run_with_thresh)
        out = run_with_thresh()
        mx.eval(out)
        cos = cosine_similarity(fp16_out, out)
        results.append((threshold, t_med * 1000, cos))

    return results


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("PolarQuant Metal -- Long-Context Stress Test")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA 4:1), D={D}, "
          f"bits={BITS}")
    print(f"Mode:       Decode (L_q=1)")
    print(f"Contexts:   {CONTEXT_LENGTHS}")
    print(f"Trials:     {N_TRIALS} (median), {WARMUP} warmup")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()

    # -----------------------------------------------------------------------
    # Part 1: Main benchmark across context lengths
    # -----------------------------------------------------------------------
    all_results = []

    for L_kv in CONTEXT_LENGTHS:
        print(f"\n{'='*78}")
        print(f"  Context Length: {L_kv:,} tokens")
        print(f"{'='*78}")

        try:
            # Generate synthetic data
            np.random.seed(42)
            queries = mx.array(np.random.randn(B, N_HEADS, 1, D).astype(np.float32))
            keys_raw = mx.array(
                np.random.randn(B, N_KV_HEADS, L_kv, D).astype(np.float32)
            )
            values_raw = mx.array(
                np.random.randn(B, N_KV_HEADS, L_kv, D).astype(np.float32)
            )
            mx.eval(queries, keys_raw, values_raw)

            # Memory estimates
            fp16_mb, pq_mb = estimate_memory_mb(B, N_KV_HEADS, L_kv, D, BITS)
            compression = fp16_mb / pq_mb

            print(f"  KV cache: FP16={fp16_mb:.1f} MB, PQ-{BITS}bit={pq_mb:.1f} MB "
                  f"({compression:.1f}x compression)")

            # --- FP16 baseline ---
            print("  Running FP16 baseline...", end="", flush=True)
            t_fp16, out_fp16 = bench_fp16(queries, keys_raw, values_raw)
            print(f" {t_fp16*1000:.2f} ms")

            # --- Naive dequantize ---
            print("  Running naive dequant...", end="", flush=True)
            t_naive, out_naive = bench_naive(queries, keys_raw, values_raw)
            print(f" {t_naive*1000:.2f} ms")

            # --- Fused Metal kernel ---
            print("  Running fused Metal...", end="", flush=True)
            t_fused, out_fused = bench_fused(queries, keys_raw, values_raw)
            print(f" {t_fused*1000:.2f} ms")

            # --- Accuracy ---
            cos_fused_fp16 = cosine_similarity(out_fp16, out_fused)
            cos_naive_fp16 = cosine_similarity(out_fp16, out_naive)

            # --- Per-operation breakdown ---
            print("  Running per-op profiling...", end="", flush=True)
            ops = profile_ops(queries, keys_raw, values_raw)
            print(" done")

            # --- Report ---
            print()
            print(f"  {'Metric':<30s} {'Value':>12s}")
            print(f"  {'-'*30} {'-'*12}")
            print(f"  {'FP16 baseline':<30s} {t_fp16*1000:>10.2f} ms")
            print(f"  {'Naive dequant+matmul':<30s} {t_naive*1000:>10.2f} ms")
            print(f"  {'Fused Metal kernel':<30s} {t_fused*1000:>10.2f} ms")
            print(f"  {'Speedup fused/FP16':<30s} {t_fp16/t_fused:>11.2f}x")
            print(f"  {'Speedup fused/naive':<30s} {t_naive/t_fused:>11.2f}x")
            print(f"  {'Naive overhead vs FP16':<30s} {t_naive/t_fp16:>11.2f}x")
            print(f"  {'Cosine sim (fused vs FP16)':<30s} {cos_fused_fp16:>12.6f}")
            print(f"  {'Cosine sim (naive vs FP16)':<30s} {cos_naive_fp16:>12.6f}")
            print(f"  {'KV cache FP16':<30s} {fp16_mb:>9.1f} MB")
            print(f"  {'KV cache PQ-{BITS}bit':<30s} {pq_mb:>9.1f} MB")
            print(f"  {'Compression ratio':<30s} {compression:>11.1f}x")

            # Op breakdown
            print()
            total_parts = sum(v for k, v in ops.items() if k != "Full pipeline")
            print(f"  {'Per-Op Breakdown':<30s} {'Time (ms)':>10s} {'% total':>8s}")
            print(f"  {'-'*30} {'-'*10} {'-'*8}")
            for name in ["Q rotation", "QK kernel", "Softmax", "SV kernel", "Out rotation"]:
                t = ops[name]
                pct = 100 * t / total_parts if total_parts > 0 else 0
                print(f"  {name:<30s} {t*1000:>10.3f} {pct:>7.1f}%")
            print(f"  {'-'*30} {'-'*10}")
            print(f"  {'Sum of parts':<30s} {total_parts*1000:>10.3f}")
            print(f"  {'Full pipeline (no barriers)':<30s} {ops['Full pipeline']*1000:>10.3f}")

            all_results.append({
                "L_kv": L_kv,
                "t_fp16": t_fp16,
                "t_naive": t_naive,
                "t_fused": t_fused,
                "cos_fused_fp16": cos_fused_fp16,
                "cos_naive_fp16": cos_naive_fp16,
                "fp16_mb": fp16_mb,
                "pq_mb": pq_mb,
                "compression": compression,
                "ops": ops,
                "error": None,
            })

        except Exception as e:
            print(f"\n  FAILED: {e}")
            traceback.print_exc()
            all_results.append({
                "L_kv": L_kv,
                "error": str(e),
            })

        # Force cleanup between context lengths
        gc.collect()

    # -----------------------------------------------------------------------
    # Part 2: Sparse V threshold sweep at 16K context
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*78}")
    print("  Sparse V Threshold Sweep (L_kv=16384)")
    print(f"{'='*78}")

    thresholds = [0.0, 0.01, 0.05]
    try:
        sparse_results = bench_sparse_sweep(16384, thresholds)
        print(f"\n  {'Threshold':<12s} {'Time (ms)':>10s} {'Cos sim vs FP16':>16s}")
        print(f"  {'-'*12} {'-'*10} {'-'*16}")
        for threshold, t_ms, cos in sparse_results:
            label = "disabled" if threshold == 0.0 else f"{threshold}"
            print(f"  {label:<12s} {t_ms:>10.2f} {cos:>16.6f}")
    except Exception as e:
        print(f"  Sparse V sweep FAILED: {e}")
        traceback.print_exc()
        sparse_results = None

    # -----------------------------------------------------------------------
    # Part 3: Summary tables
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*78}")
    print("  SUMMARY TABLE")
    print(f"{'='*78}\n")

    # Header
    hdr = (f"  {'L_kv':>6s} | {'FP16 (ms)':>10s} | {'Naive (ms)':>10s} | "
           f"{'Fused (ms)':>10s} | {'vs FP16':>8s} | {'vs Naive':>8s} | "
           f"{'Cos Sim':>8s} | {'FP16 MB':>8s} | {'PQ MB':>8s} | {'Compr':>6s}")
    print(hdr)
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    for r in all_results:
        if r.get("error"):
            print(f"  {r['L_kv']:>6,d} | {'OOM/FAIL':>10s} | {'':>10s} | "
                  f"{'':>10s} | {'':>8s} | {'':>8s} | {'':>8s} | {'':>8s} | "
                  f"{'':>8s} | {'':>6s}")
            continue
        print(f"  {r['L_kv']:>6,d} | {r['t_fp16']*1000:>10.2f} | "
              f"{r['t_naive']*1000:>10.2f} | {r['t_fused']*1000:>10.2f} | "
              f"{r['t_fp16']/r['t_fused']:>7.2f}x | "
              f"{r['t_naive']/r['t_fused']:>7.2f}x | "
              f"{r['cos_fused_fp16']:>8.4f} | {r['fp16_mb']:>8.1f} | "
              f"{r['pq_mb']:>8.1f} | {r['compression']:>5.1f}x")

    # -----------------------------------------------------------------------
    # Part 4: Save results as markdown
    # -----------------------------------------------------------------------
    md_path = os.path.join(os.path.dirname(__file__), "STRESS_RESULTS.md")
    with open(md_path, "w") as f:
        f.write("# PolarQuant Metal -- Long-Context Stress Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA 4:1), "
                f"D={D}, bits={BITS}  \n")
        f.write(f"**Mode:** Decode (L_q=1)  \n")
        f.write(f"**Trials:** {N_TRIALS} (median), {WARMUP} warmup  \n")
        f.write(f"**Device:** {mx.default_device()}, Metal={mx.metal.is_available()}  \n\n")

        # Main table
        f.write("## Performance vs Context Length\n\n")
        f.write("| Context | FP16 (ms) | Naive (ms) | Fused (ms) | "
                "Fused/FP16 | Fused/Naive | Cos Sim | FP16 (MB) | PQ-3bit (MB) | Compression |\n")
        f.write("|--------:|----------:|-----------:|-----------:|"
                "----------:|------------:|--------:|----------:|-------------:|------------:|\n")

        for r in all_results:
            if r.get("error"):
                f.write(f"| {r['L_kv']:,} | FAIL | | | | | | | | |\n")
                continue
            f.write(
                f"| {r['L_kv']:,} "
                f"| {r['t_fp16']*1000:.2f} "
                f"| {r['t_naive']*1000:.2f} "
                f"| {r['t_fused']*1000:.2f} "
                f"| {r['t_fp16']/r['t_fused']:.2f}x "
                f"| {r['t_naive']/r['t_fused']:.2f}x "
                f"| {r['cos_fused_fp16']:.6f} "
                f"| {r['fp16_mb']:.1f} "
                f"| {r['pq_mb']:.1f} "
                f"| {r['compression']:.1f}x |\n"
            )

        # Per-op breakdown table
        f.write("\n## Per-Operation Breakdown (Fused Pipeline)\n\n")
        f.write("| Context | Q Rot (ms) | QK Kernel (ms) | Softmax (ms) | "
                "SV Kernel (ms) | Out Rot (ms) | Pipeline (ms) |\n")
        f.write("|--------:|-----------:|---------------:|-------------:|"
                "---------------:|-------------:|--------------:|\n")

        for r in all_results:
            if r.get("error") or "ops" not in r:
                continue
            ops = r["ops"]
            f.write(
                f"| {r['L_kv']:,} "
                f"| {ops['Q rotation']*1000:.3f} "
                f"| {ops['QK kernel']*1000:.3f} "
                f"| {ops['Softmax']*1000:.3f} "
                f"| {ops['SV kernel']*1000:.3f} "
                f"| {ops['Out rotation']*1000:.3f} "
                f"| {ops['Full pipeline']*1000:.3f} |\n"
            )

        # Sparse V section
        f.write("\n## Sparse V Threshold Impact (16K Context)\n\n")
        if sparse_results:
            f.write("| Threshold | Time (ms) | Cos Sim vs FP16 | Notes |\n")
            f.write("|----------:|----------:|----------------:|:------|\n")
            for threshold, t_ms, cos in sparse_results:
                label = "0.0 (disabled)" if threshold == 0.0 else f"{threshold}"
                note = "baseline" if threshold == 0.0 else ""
                if threshold > 0 and sparse_results[0][1] > 0:
                    speedup = sparse_results[0][1] / t_ms
                    note = f"{speedup:.2f}x vs disabled"
                f.write(f"| {label} | {t_ms:.2f} | {cos:.6f} | {note} |\n")
        else:
            f.write("Sparse V sweep failed.\n")

        # Analysis
        f.write("\n## Analysis\n\n")
        ok = [r for r in all_results if not r.get("error")]
        if ok:
            # Scaling analysis
            if len(ok) >= 2:
                first, last = ok[0], ok[-1]
                ctx_ratio = last["L_kv"] / first["L_kv"]
                fused_ratio = (last["t_fused"] / first["t_fused"])
                fp16_ratio = (last["t_fp16"] / first["t_fp16"])
                f.write(f"- **Scaling ({first['L_kv']:,} to {last['L_kv']:,}):** "
                        f"context grows {ctx_ratio:.0f}x, fused time grows {fused_ratio:.1f}x, "
                        f"FP16 time grows {fp16_ratio:.1f}x\n")

            # Best speedup
            best = max(ok, key=lambda r: r["t_naive"] / r["t_fused"])
            f.write(f"- **Best fused/naive speedup:** {best['t_naive']/best['t_fused']:.2f}x "
                    f"at {best['L_kv']:,} context\n")

            # Memory savings at max context
            last = ok[-1]
            saved = last["fp16_mb"] - last["pq_mb"]
            f.write(f"- **Memory saved at {last['L_kv']:,}:** "
                    f"{saved:.1f} MB ({last['compression']:.1f}x compression)\n")

            # Accuracy
            min_cos = min(r["cos_fused_fp16"] for r in ok)
            f.write(f"- **Worst-case cosine similarity:** {min_cos:.6f}\n")

            # Check accuracy flag
            if min_cos < 0.99:
                f.write(f"- **WARNING:** Cosine similarity dropped below 0.99 "
                        f"at long contexts -- may need investigation\n")

    print(f"\n  Results saved to: {md_path}")
    print(f"\n{'='*78}")
    print("  STRESS TEST COMPLETE")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
