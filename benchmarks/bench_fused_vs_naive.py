"""
Benchmark: Fused Metal kernel vs naive dequantize-on-fetch.

Measures time for the attention computation (Q@K^T + softmax + scores@V)
comparing:
  1. Naive: dequantize → standard matmul
  2. Fused: direct Metal kernel on packed indices

Run on Apple Silicon to see the speedup.
"""

import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from polarquant_metal.polar_quant import PolarQuant
from polarquant_metal.packing import pack_indices
from polarquant_metal.codebooks import load_codebook_f32
from polarquant_metal.kernels import polarquant_qk_matmul, polarquant_sv_matmul
from polarquant_metal.cache import FusedPolarQuantKVCache


def bench_naive(queries, keys_raw, values_raw, bits, n_trials=5, warmup=2):
    """Benchmark naive dequantize-then-matmul path."""
    D = queries.shape[-1]
    scale = 1.0 / math.sqrt(D)
    pq_key = PolarQuant(bits=bits, dim=D, seed=42)
    pq_val = PolarQuant(bits=bits, dim=D, seed=43)

    k_indices, k_norms = pq_key.quantize(keys_raw)
    v_indices, v_norms = pq_val.quantize(values_raw)

    # Warmup
    for _ in range(warmup):
        keys_deq = pq_key.dequantize(k_indices, k_norms)
        values_deq = pq_val.dequantize(v_indices, v_norms)
        scores = (queries @ mx.swapaxes(keys_deq, -2, -1)) * scale
        weights = mx.softmax(scores, axis=-1, precise=True)
        output = weights @ values_deq
        mx.eval(output)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        keys_deq = pq_key.dequantize(k_indices, k_norms)
        values_deq = pq_val.dequantize(v_indices, v_norms)
        scores = (queries @ mx.swapaxes(keys_deq, -2, -1)) * scale
        weights = mx.softmax(scores, axis=-1, precise=True)
        output = weights @ values_deq
        mx.eval(output)
        times.append(time.perf_counter() - t0)

    return np.median(times), output


def bench_fused(queries, keys_raw, values_raw, bits, n_trials=5, warmup=2):
    """Benchmark fused Metal kernel path."""
    D = queries.shape[-1]

    cache = FusedPolarQuantKVCache(bits=bits, head_dim=D, key_seed=42, value_seed=43)
    cache.update_and_fetch(keys_raw, values_raw)

    # Warmup
    for _ in range(warmup):
        output = cache.fused_attention(queries)
        mx.eval(output)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        output = cache.fused_attention(queries)
        mx.eval(output)
        times.append(time.perf_counter() - t0)

    return np.median(times), output


def bench_fp16_baseline(queries, keys_raw, values_raw, n_trials=5, warmup=2):
    """Benchmark uncompressed FP16 baseline (no quantization at all)."""
    D = queries.shape[-1]
    scale = 1.0 / math.sqrt(D)

    # Warmup
    for _ in range(warmup):
        scores = (queries @ mx.swapaxes(keys_raw, -2, -1)) * scale
        weights = mx.softmax(scores, axis=-1, precise=True)
        output = weights @ values_raw
        mx.eval(output)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        scores = (queries @ mx.swapaxes(keys_raw, -2, -1)) * scale
        weights = mx.softmax(scores, axis=-1, precise=True)
        output = weights @ values_raw
        mx.eval(output)
        times.append(time.perf_counter() - t0)

    return np.median(times), output


def run_benchmark(
    B=1,
    n_heads=32,
    n_kv_heads=8,
    L_q=1,       # decode: single token
    L_kv=512,    # cache length
    D=128,
    bits=3,
    n_trials=10,
):
    print(f"\nBenchmark: B={B}, heads={n_heads}/{n_kv_heads}, "
          f"L_q={L_q}, L_kv={L_kv}, D={D}, bits={bits}")
    print("-" * 70)

    np.random.seed(42)
    queries = mx.array(np.random.randn(B, n_heads, L_q, D).astype(np.float32))
    keys_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    values_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))

    # FP16 baseline (no quantization)
    t_fp16, _ = bench_fp16_baseline(queries, keys_raw, values_raw, n_trials)

    # Naive PolarQuant (dequantize-on-fetch)
    t_naive, out_naive = bench_naive(queries, keys_raw, values_raw, bits, n_trials)

    # Fused Metal kernel
    t_fused, out_fused = bench_fused(queries, keys_raw, values_raw, bits, n_trials)

    # Memory comparison
    vals_per_int = 32 // bits
    D_packed = (D + vals_per_int - 1) // vals_per_int
    compressed_kv_bytes = 2 * B * n_kv_heads * L_kv * (D_packed * 4 + 4)  # indices + norms
    fp16_kv_bytes = 2 * B * n_kv_heads * L_kv * D * 2  # keys + values in fp16
    compression_ratio = fp16_kv_bytes / compressed_kv_bytes

    print(f"  FP16 baseline:        {t_fp16*1000:8.2f} ms")
    print(f"  Naive dequant+matmul: {t_naive*1000:8.2f} ms "
          f"({t_naive/t_fp16:.2f}x vs FP16)")
    print(f"  Fused Metal kernel:   {t_fused*1000:8.2f} ms "
          f"({t_fused/t_fp16:.2f}x vs FP16)")
    print(f"  Speedup (fused/naive):{t_naive/t_fused:8.2f}x")
    print(f"  KV cache compression: {compression_ratio:.1f}x "
          f"({fp16_kv_bytes/1024:.0f} KB → {compressed_kv_bytes/1024:.0f} KB)")

    # Verify correctness
    cos_sim = float(
        mx.sum(out_naive * out_fused)
        / (mx.sqrt(mx.sum(out_naive**2)) * mx.sqrt(mx.sum(out_fused**2)) + 1e-10)
    )
    print(f"  Correctness (cos sim): {cos_sim:.6f}")


if __name__ == "__main__":
    print("=" * 70)
    print("PolarQuant Fused Metal Kernel Benchmark")
    print("=" * 70)

    # Decode scenario: L_q=1, varying L_kv
    for L_kv in [64, 256, 512, 1024, 2048]:
        run_benchmark(L_q=1, L_kv=L_kv, bits=3)

    # Prefill scenario: L_q=L_kv
    print("\n\nPrefill scenarios:")
    for L in [64, 256]:
        run_benchmark(L_q=L, L_kv=L, bits=3)

    # Bit width comparison
    print("\n\nBit width comparison (L_kv=512):")
    for bits in [2, 3, 4]:
        run_benchmark(L_q=1, L_kv=512, bits=bits)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
