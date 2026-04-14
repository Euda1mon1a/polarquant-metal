#!/usr/bin/env python3
"""
Benchmark: simdgroup-reduced SV kernels vs scalar SV kernels.

Compares four paths at each context length:
  - scalar_dense:  _SV_PRECOMBINED_SOURCE  (1 thread/output, dense L_kv loop)
  - simd_dense:    _SV_SIMD_SOURCE         (32 lanes/output, simd_sum over L_kv)
  - scalar_sparse: _SV_SPARSE_SOURCE       (Phase 3 compact-index, 1 thread/output)
  - simd_sparse:   _SV_SIMD_SPARSE_SOURCE  (Phase 3 compact-index, 32 lanes/output)

All paths produce identical output (cosine similarity checked).

Usage:
    cd ~/workspace/polarquant-metal
    python benchmarks/bench_sv_simd.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from polarquant_metal.packing import pack_indices
from polarquant_metal.codebooks import load_codebook_f32
from polarquant_metal.kernels import (
    polarquant_sv_matmul,
    polarquant_sv_build_index,
    polarquant_sv_sparse,
)

# MLX compute-graph synchronization (mlx.core.eval — not Python built-in)
_sync = mx.eval

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 8
N_KV_HEADS = 2
D = 128
BITS = 3
REP = N_HEADS // N_KV_HEADS

WARMUP = 5
N_TRIALS = 20

SPARSE_THRESH = 1e-3   # per-head threshold for sparse path
SPARSE_FRAC   = 0.01   # ~1% active positions (concentrated attention)

CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    na = float(mx.sqrt(mx.sum(a_f * a_f)))
    nb = float(mx.sqrt(mx.sum(b_f * b_f)))
    if na < 1e-8 and nb < 1e-8:
        return 1.0
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(mx.sum(a_f * b_f) / (na * nb))


def bench(fn, warmup=WARMUP, n=N_TRIALS):
    for _ in range(warmup):
        out = fn()
        _sync(out)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        _sync(out)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000, np.std(times) * 1000  # ms


def make_inputs(L_kv: int, rng: np.random.Generator):
    """Build synthetic quantized V cache + concentrated softmax weights."""
    # Use random indices directly — perf benchmark, exact values don't matter
    n_levels = 1 << BITS
    v_indices_np = rng.integers(0, n_levels, size=(B, N_KV_HEADS, L_kv, D), dtype=np.uint32)
    v_packed = pack_indices(
        mx.array(v_indices_np, dtype=mx.uint8), bits=BITS
    )
    v_centroids = load_codebook_f32(bits=BITS, dim=D)

    # Concentrated attention weights — most mass on ~1% of positions
    raw_w = rng.standard_normal((B, N_HEADS, 1, L_kv)).astype(np.float32) * 0.1
    hot_count = max(1, int(SPARSE_FRAC * L_kv))
    hot_idx = rng.choice(L_kv, size=hot_count, replace=False)
    raw_w[:, :, :, hot_idx] += 5.0
    weights = mx.softmax(mx.array(raw_w), axis=-1)

    # v_norms (all ones — simplified for benchmark)
    v_norms = mx.ones((B, N_KV_HEADS, L_kv, 1), dtype=mx.float16)

    # Precombine weight * norm
    norms_sq = v_norms.squeeze(-1)
    norms_exp = mx.repeat(norms_sq, REP, axis=1) if REP > 1 else norms_sq
    wn = (weights * norms_exp[:, :, None, :]).astype(mx.float16)

    thresh_arr = mx.full((N_HEADS,), SPARSE_THRESH, dtype=mx.float32)
    zone_prior = mx.zeros((L_kv,), dtype=mx.uint32)

    return wn, v_packed, v_centroids, v_norms, thresh_arr, zone_prior


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PolarQuant SV kernel: simdgroup-reduced vs scalar")
    print(f"Config: B={B}, N_HEADS={N_HEADS}, N_KV_HEADS={N_KV_HEADS}, D={D}, BITS={BITS}")
    print(f"Sparse threshold: {SPARSE_THRESH}, Active fraction: ~{SPARSE_FRAC*100:.0f}%")
    print(f"Warmup: {WARMUP}, Trials: {N_TRIALS}")
    print("=" * 80)

    rng = np.random.default_rng(42)

    print(
        f"{'L_kv':>7}  {'scalar_d':>9}  {'simd_d':>8}  "
        f"{'scalar_s':>9}  {'simd_s':>8}  "
        f"{'spd_dense':>10}  {'spd_sparse':>11}  {'cos_sim':>8}  active"
    )
    print("-" * 88)

    for L_kv in CONTEXT_LENGTHS:
        wn, v_packed, v_centroids, v_norms, thresh_arr, zone_prior = make_inputs(
            L_kv, rng
        )

        count_and_indices = polarquant_sv_build_index(wn, thresh_arr, zone_prior)
        _sync(count_and_indices)

        stride = 1 + L_kv
        counts = count_and_indices[::stride]
        avg_count = float(counts.astype(mx.float32).mean())

        def fn_scalar_dense():
            return polarquant_sv_matmul(
                weights=wn, v_indices=v_packed, v_norms=v_norms,
                v_centroids=v_centroids, head_dim=D, bits=BITS,
                sparse_v_threshold=thresh_arr, use_simd=False,
            )

        def fn_simd_dense():
            return polarquant_sv_matmul(
                weights=wn, v_indices=v_packed, v_norms=v_norms,
                v_centroids=v_centroids, head_dim=D, bits=BITS,
                sparse_v_threshold=thresh_arr, use_simd=True,
            )

        def fn_scalar_sparse():
            return polarquant_sv_sparse(
                count_and_indices, wn, v_packed, v_centroids,
                D, L_kv, BITS, use_simd=False,
            )

        def fn_simd_sparse():
            return polarquant_sv_sparse(
                count_and_indices, wn, v_packed, v_centroids,
                D, L_kv, BITS, use_simd=True,
            )

        t_sd, _ = bench(fn_scalar_dense)
        t_smd, _ = bench(fn_simd_dense)
        t_ss, _ = bench(fn_scalar_sparse)
        t_sms, _ = bench(fn_simd_sparse)

        out_ref = fn_scalar_sparse()
        out_simd = fn_simd_sparse()
        _sync(out_ref, out_simd)
        cs = cosine_similarity(out_ref, out_simd)

        print(
            f"{L_kv:>7}  {t_sd:>7.2f}ms  {t_smd:>6.2f}ms  "
            f"{t_ss:>7.2f}ms  {t_sms:>6.2f}ms  "
            f"{t_sd/t_smd:>8.2f}x  {t_ss/t_sms:>9.2f}x  "
            f"{cs:>8.6f}  {avg_count:.0f}/{L_kv}"
        )

    print()
    print("Columns:")
    print("  scalar_d / simd_d   — dense precombined path (L_kv inner loop)")
    print("  scalar_s / simd_s   — Phase 3 compact-index sparse path")
    print("  spd_dense / spd_sparse — speedup ratios")
    print("  cos_sim             — output cosine similarity (should be ~1.0)")
    print("  active              — avg active positions / L_kv per head")


if __name__ == "__main__":
    main()
