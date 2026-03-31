#!/usr/bin/env python3
"""Per-operation timing breakdown for PolarQuant fused attention.

Decomposes the fused attention pipeline into 5 operations and times each
with mx.eval() barriers to measure where time is spent:

  1. Q rotation  (D x D matmul, constant cost)
  2. QK kernel   (scales with L_kv)
  3. Softmax     (scales with L_kv)
  4. SV kernel   (scales with L_kv)
  5. Out rotation (D x D matmul, constant cost)
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from polarquant_metal.polar_quant import PolarQuant
from polarquant_metal.packing import pack_indices
from polarquant_metal.codebooks import load_codebook_f32
from polarquant_metal.kernels import polarquant_qk_matmul, polarquant_sv_matmul


def profile_attention(L_kv, bits=3, B=1, n_heads=32, n_kv_heads=8, D=128,
                      n_trials=20, warmup=5):
    """Profile each operation in fused PolarQuant attention."""
    scale = 1.0 / math.sqrt(D)

    np.random.seed(42)
    queries = mx.array(np.random.randn(B, n_heads, 1, D).astype(np.float32))
    keys_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    values_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))

    pq_key = PolarQuant(bits=bits, dim=D, seed=42)
    pq_val = PolarQuant(bits=bits, dim=D, seed=43)

    k_idx, k_norms = pq_key.quantize(keys_raw)
    v_idx, v_norms = pq_val.quantize(values_raw)
    k_packed = pack_indices(k_idx, bits)
    v_packed = pack_indices(v_idx, bits)

    key_centroids = load_codebook_f32(bits, D)
    val_centroids = load_codebook_f32(bits, D)
    key_rotation_t = pq_key.rotation_t
    val_rotation = pq_val.rotation

    mx.eval(queries, k_packed, k_norms, v_packed, v_norms,
            key_centroids, val_centroids, key_rotation_t, val_rotation)

    # Pre-compute intermediate values for isolated timing
    q_rot = queries @ key_rotation_t; mx.eval(q_rot)
    scores = polarquant_qk_matmul(q_rot, k_packed, k_norms, key_centroids, scale, bits); mx.eval(scores)
    weights = mx.softmax(scores, axis=-1, precise=True); mx.eval(weights)
    out_rot = polarquant_sv_matmul(weights, v_packed, v_norms, val_centroids, D, bits); mx.eval(out_rot)

    def time_op(fn, warmup_n=warmup, trials_n=n_trials):
        for _ in range(warmup_n):
            r = fn(); mx.eval(r)
        times = []
        for _ in range(trials_n):
            t0 = time.perf_counter()
            r = fn(); mx.eval(r)
            times.append(time.perf_counter() - t0)
        return np.median(times)

    ops = {}
    ops["Q rotation"] = time_op(lambda: queries @ key_rotation_t)
    ops["QK kernel"] = time_op(lambda: polarquant_qk_matmul(q_rot, k_packed, k_norms, key_centroids, scale, bits))
    ops["Softmax"] = time_op(lambda: mx.softmax(scores, axis=-1, precise=True))
    ops["SV kernel"] = time_op(lambda: polarquant_sv_matmul(weights, v_packed, v_norms, val_centroids, D, bits))
    ops["Out rotation"] = time_op(lambda: out_rot @ val_rotation)

    # Full pipeline (no eval barriers between ops)
    def full_pipeline():
        q_r = queries @ key_rotation_t
        s = polarquant_qk_matmul(q_r, k_packed, k_norms, key_centroids, scale, bits)
        w = mx.softmax(s, axis=-1, precise=True)
        o_r = polarquant_sv_matmul(w, v_packed, v_norms, val_centroids, D, bits)
        return o_r @ val_rotation

    ops["Full pipeline"] = time_op(full_pipeline)
    return ops


def main():
    print("PolarQuant Metal — Per-Operation Profiling")
    print("=" * 70)
    print("Config: 3-bit, B=1, 32 query heads / 8 KV heads, D=128, decode (L_q=1)")

    all_results = {}
    for L_kv in [64, 256, 512, 1024, 2048]:
        ops = profile_attention(L_kv)
        all_results[L_kv] = ops

        total_parts = sum(v for k, v in ops.items() if k != "Full pipeline")
        print(f"\nL_kv = {L_kv}")
        print(f"  {'Operation':<20s} {'Time (ms)':>10s} {'% of sum':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        for name, t in ops.items():
            if name == "Full pipeline":
                continue
            pct = 100 * t / total_parts if total_parts > 0 else 0
            print(f"  {name:<20s} {t*1000:10.3f} {pct:9.1f}%")
        print(f"  {'-'*20} {'-'*10}")
        print(f"  {'Sum of parts':<20s} {total_parts*1000:10.3f}")
        print(f"  {'Full pipeline':<20s} {ops['Full pipeline']*1000:10.3f}")
        savings = (total_parts - ops['Full pipeline']) / total_parts * 100 if total_parts > 0 else 0
        print(f"  Lazy eval savings:  {savings:+.1f}%")

    # Summary table (markdown)
    print("\n\n## Profiling Summary Table\n")
    print("| L_kv | Q Rot (ms) | QK Kern (ms) | Softmax (ms) | SV Kern (ms) | Out Rot (ms) | Sum (ms) | Pipeline (ms) |")
    print("|------|-----------|-------------|-------------|-------------|-------------|---------|--------------|")
    for L_kv, ops in all_results.items():
        total = sum(v for k, v in ops.items() if k != "Full pipeline")
        print(f"| {L_kv:4d} | {ops['Q rotation']*1000:9.3f} | {ops['QK kernel']*1000:11.3f} "
              f"| {ops['Softmax']*1000:11.3f} | {ops['SV kernel']*1000:11.3f} "
              f"| {ops['Out rotation']*1000:11.3f} | {total*1000:7.3f} | {ops['Full pipeline']*1000:12.3f} |")


if __name__ == "__main__":
    main()
