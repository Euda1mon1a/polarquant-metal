"""
Tests for the fused PolarQuant Metal kernels.

Verifies correctness by comparing fused kernel output against
the naive dequantize-then-matmul reference implementation.
"""

import math
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from polarquant_metal.polar_quant import PolarQuant
from polarquant_metal.packing import pack_indices, unpack_indices
from polarquant_metal.codebooks import load_codebook_f32
from polarquant_metal.kernels import polarquant_qk_matmul, polarquant_sv_matmul
from polarquant_metal.cache import FusedPolarQuantKVCache


def reference_qk_matmul(queries, pq, indices, norms, scale):
    """Naive reference: dequantize then matmul."""
    # Dequantize keys
    keys_recon = pq.dequantize(indices, norms)
    # Rotate queries
    q_rotated_full = queries  # queries are already in original space
    # Standard attention: Q @ K^T * scale
    scores = (q_rotated_full @ mx.swapaxes(keys_recon, -2, -1)) * scale
    return scores


def reference_sv_matmul(weights, pq, indices, norms):
    """Naive reference: dequantize values then weighted sum."""
    values_recon = pq.dequantize(indices, norms)
    return weights @ values_recon


def test_pack_unpack_roundtrip():
    """Test that packing and unpacking indices is lossless."""
    for bits in [2, 3, 4]:
        n_levels = 2 ** bits
        shape = (2, 4, 8, 128)
        indices = mx.array(
            np.random.randint(0, n_levels, size=shape, dtype=np.uint8)
        )
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, 128)

        np.testing.assert_array_equal(
            np.array(indices), np.array(unpacked),
            f"Pack/unpack roundtrip failed for {bits}-bit"
        )
    print("PASS: test_pack_unpack_roundtrip")


def test_fused_qk_vs_reference():
    """Test fused Q@K^T kernel matches naive dequantize-then-matmul."""
    np.random.seed(42)

    B, n_heads, L_q, D = 1, 4, 2, 128
    n_kv_heads = 4
    L_kv = 8
    bits = 3
    scale = 1.0 / math.sqrt(D)

    pq = PolarQuant(bits=bits, dim=D, seed=42)

    # Generate random keys, quantize them
    keys_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    k_indices, k_norms = pq.quantize(keys_raw)

    # Generate random queries
    queries = mx.array(np.random.randn(B, n_heads, L_q, D).astype(np.float32))

    # Reference: dequantize then matmul
    ref_scores = reference_qk_matmul(queries, pq, k_indices, k_norms, scale)
    mx.eval(ref_scores)

    # Fused: pack indices, pre-rotate queries, call kernel
    k_packed = pack_indices(k_indices, bits)
    q_rotated = queries @ pq.rotation_t
    centroids_f32 = load_codebook_f32(bits, D)

    fused_scores = polarquant_qk_matmul(
        queries=q_rotated,
        indices=k_packed,
        norms=k_norms,
        centroids=centroids_f32,
        scale=scale,
        bits=bits,
        use_tiled=False,  # test simple kernel first
    )
    mx.eval(fused_scores)

    # Compare
    ref_np = np.array(ref_scores)
    fused_np = np.array(fused_scores)

    max_diff = np.max(np.abs(ref_np - fused_np))
    cos_sim = np.sum(ref_np * fused_np) / (
        np.linalg.norm(ref_np) * np.linalg.norm(fused_np) + 1e-10
    )

    print(f"Q@K^T simple kernel - max diff: {max_diff:.6e}, cos sim: {cos_sim:.8f}")
    assert max_diff < 0.01, f"Q@K^T simple kernel max diff too large: {max_diff}"
    assert cos_sim > 0.999, f"Q@K^T simple kernel cos sim too low: {cos_sim}"
    print("PASS: test_fused_qk_vs_reference (simple)")


def test_fused_qk_tiled_vs_reference():
    """Test tiled Q@K^T kernel matches reference."""
    np.random.seed(123)

    B, n_heads, L_q, D = 1, 4, 4, 128
    n_kv_heads = 4
    L_kv = 16
    bits = 3
    scale = 1.0 / math.sqrt(D)

    pq = PolarQuant(bits=bits, dim=D, seed=42)

    keys_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    k_indices, k_norms = pq.quantize(keys_raw)
    queries = mx.array(np.random.randn(B, n_heads, L_q, D).astype(np.float32))

    ref_scores = reference_qk_matmul(queries, pq, k_indices, k_norms, scale)
    mx.eval(ref_scores)

    k_packed = pack_indices(k_indices, bits)
    q_rotated = queries @ pq.rotation_t
    centroids_f32 = load_codebook_f32(bits, D)

    fused_scores = polarquant_qk_matmul(
        queries=q_rotated,
        indices=k_packed,
        norms=k_norms,
        centroids=centroids_f32,
        scale=scale,
        bits=bits,
        use_tiled=True,
    )
    mx.eval(fused_scores)

    ref_np = np.array(ref_scores)
    fused_np = np.array(fused_scores)

    max_diff = np.max(np.abs(ref_np - fused_np))
    cos_sim = np.sum(ref_np * fused_np) / (
        np.linalg.norm(ref_np) * np.linalg.norm(fused_np) + 1e-10
    )

    print(f"Q@K^T tiled kernel - max diff: {max_diff:.6e}, cos sim: {cos_sim:.8f}")
    assert max_diff < 0.01, f"Q@K^T tiled max diff too large: {max_diff}"
    assert cos_sim > 0.999, f"Q@K^T tiled cos sim too low: {cos_sim}"
    print("PASS: test_fused_qk_tiled_vs_reference")


def test_fused_sv_vs_reference():
    """Test fused weights@V kernel matches reference."""
    np.random.seed(456)

    B, n_heads, L_q, D = 1, 4, 2, 128
    n_kv_heads = 4
    L_kv = 8
    bits = 3

    pq = PolarQuant(bits=bits, dim=D, seed=43)

    values_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    v_indices, v_norms = pq.quantize(values_raw)

    # Random attention weights (post-softmax)
    weights_raw = np.random.rand(B, n_heads, L_q, L_kv).astype(np.float32)
    weights_raw = weights_raw / weights_raw.sum(axis=-1, keepdims=True)
    weights = mx.array(weights_raw)

    # Reference
    ref_output = reference_sv_matmul(weights, pq, v_indices, v_norms)
    mx.eval(ref_output)

    # Fused
    v_packed = pack_indices(v_indices, bits)
    centroids_f32 = load_codebook_f32(bits, D)

    fused_rotated = polarquant_sv_matmul(
        weights=weights,
        v_indices=v_packed,
        v_norms=v_norms,
        v_centroids=centroids_f32,
        head_dim=D,
        bits=bits,
    )
    # Inverse rotate
    fused_output = fused_rotated @ pq.rotation
    mx.eval(fused_output)

    ref_np = np.array(ref_output)
    fused_np = np.array(fused_output)

    max_diff = np.max(np.abs(ref_np - fused_np))
    cos_sim = np.sum(ref_np * fused_np) / (
        np.linalg.norm(ref_np) * np.linalg.norm(fused_np) + 1e-10
    )

    print(f"Weights@V kernel - max diff: {max_diff:.6e}, cos sim: {cos_sim:.8f}")
    assert max_diff < 0.01, f"SV kernel max diff too large: {max_diff}"
    assert cos_sim > 0.999, f"SV kernel cos sim too low: {cos_sim}"
    print("PASS: test_fused_sv_vs_reference")


def test_fused_cache_attention():
    """Test the full FusedPolarQuantKVCache.fused_attention path."""
    np.random.seed(789)

    B, n_heads, L_q, D = 1, 4, 2, 128
    n_kv_heads = 4
    L_kv = 8
    bits = 3
    scale = 1.0 / math.sqrt(D)

    # Reference path: quantize, dequantize, standard attention
    pq_key = PolarQuant(bits=bits, dim=D, seed=42)
    pq_val = PolarQuant(bits=bits, dim=D, seed=43)

    keys_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    values_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    queries = mx.array(np.random.randn(B, n_heads, L_q, D).astype(np.float32))

    k_indices, k_norms = pq_key.quantize(keys_raw)
    v_indices, v_norms = pq_val.quantize(values_raw)
    keys_deq = pq_key.dequantize(k_indices, k_norms)
    values_deq = pq_val.dequantize(v_indices, v_norms)

    ref_scores = (queries @ mx.swapaxes(keys_deq, -2, -1)) * scale
    ref_weights = mx.softmax(ref_scores, axis=-1, precise=True)
    ref_output = ref_weights @ values_deq
    mx.eval(ref_output)

    # Fused path
    cache = FusedPolarQuantKVCache(bits=bits, head_dim=D, key_seed=42, value_seed=43)
    cache.update_and_fetch(keys_raw, values_raw)
    fused_output = cache.fused_attention(queries)
    mx.eval(fused_output)

    ref_np = np.array(ref_output)
    fused_np = np.array(fused_output)

    max_diff = np.max(np.abs(ref_np - fused_np))
    cos_sim = np.sum(ref_np * fused_np) / (
        np.linalg.norm(ref_np) * np.linalg.norm(fused_np) + 1e-10
    )

    print(f"Full attention - max diff: {max_diff:.6e}, cos sim: {cos_sim:.8f}")
    assert max_diff < 0.05, f"Full attention max diff too large: {max_diff}"
    assert cos_sim > 0.99, f"Full attention cos sim too low: {cos_sim}"
    print("PASS: test_fused_cache_attention")


def test_gqa_support():
    """Test grouped query attention (n_heads > n_kv_heads)."""
    np.random.seed(101)

    B = 1
    n_heads = 8
    n_kv_heads = 2
    L_q = 2
    L_kv = 4
    D = 64
    bits = 4
    scale = 1.0 / math.sqrt(D)

    pq_key = PolarQuant(bits=bits, dim=D, seed=42)
    pq_val = PolarQuant(bits=bits, dim=D, seed=43)

    keys_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    values_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    queries = mx.array(np.random.randn(B, n_heads, L_q, D).astype(np.float32))

    # Reference: expand KV heads for GQA
    n_rep = n_heads // n_kv_heads
    k_indices, k_norms = pq_key.quantize(keys_raw)
    v_indices, v_norms = pq_val.quantize(values_raw)
    keys_deq = pq_key.dequantize(k_indices, k_norms)
    values_deq = pq_val.dequantize(v_indices, v_norms)

    # Expand for GQA
    keys_expanded = mx.repeat(keys_deq, n_rep, axis=1)
    values_expanded = mx.repeat(values_deq, n_rep, axis=1)

    ref_scores = (queries @ mx.swapaxes(keys_expanded, -2, -1)) * scale
    ref_weights = mx.softmax(ref_scores, axis=-1, precise=True)
    ref_output = ref_weights @ values_expanded
    mx.eval(ref_output)

    # Fused path
    cache = FusedPolarQuantKVCache(bits=bits, head_dim=D, key_seed=42, value_seed=43)
    cache.update_and_fetch(keys_raw, values_raw)
    fused_output = cache.fused_attention(queries)
    mx.eval(fused_output)

    ref_np = np.array(ref_output)
    fused_np = np.array(fused_output)

    max_diff = np.max(np.abs(ref_np - fused_np))
    cos_sim = np.sum(ref_np * fused_np) / (
        np.linalg.norm(ref_np) * np.linalg.norm(fused_np) + 1e-10
    )

    print(f"GQA (8h/2kv) - max diff: {max_diff:.6e}, cos sim: {cos_sim:.8f}")
    assert max_diff < 0.05, f"GQA max diff too large: {max_diff}"
    assert cos_sim > 0.99, f"GQA cos sim too low: {cos_sim}"
    print("PASS: test_gqa_support")


def test_all_bit_widths():
    """Test all supported bit widths (2, 3, 4)."""
    np.random.seed(202)

    B, n_heads, L_q, D = 1, 2, 2, 64
    n_kv_heads = 2
    L_kv = 4

    for bits in [2, 3, 4]:
        scale = 1.0 / math.sqrt(D)
        pq = PolarQuant(bits=bits, dim=D, seed=42)

        keys_raw = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
        queries = mx.array(np.random.randn(B, n_heads, L_q, D).astype(np.float32))

        k_indices, k_norms = pq.quantize(keys_raw)
        ref_scores = reference_qk_matmul(queries, pq, k_indices, k_norms, scale)
        mx.eval(ref_scores)

        k_packed = pack_indices(k_indices, bits)
        q_rotated = queries @ pq.rotation_t
        centroids_f32 = load_codebook_f32(bits, D)

        fused_scores = polarquant_qk_matmul(
            queries=q_rotated,
            indices=k_packed,
            norms=k_norms,
            centroids=centroids_f32,
            scale=scale,
            bits=bits,
            use_tiled=False,
        )
        mx.eval(fused_scores)

        max_diff = np.max(np.abs(np.array(ref_scores) - np.array(fused_scores)))
        print(f"  {bits}-bit: max diff = {max_diff:.6e}")
        assert max_diff < 0.01, f"{bits}-bit max diff too large"

    print("PASS: test_all_bit_widths")


if __name__ == "__main__":
    print("=" * 60)
    print("PolarQuant Fused Metal Kernel Tests")
    print("=" * 60)

    test_pack_unpack_roundtrip()
    print()

    test_fused_qk_vs_reference()
    print()

    test_fused_qk_tiled_vs_reference()
    print()

    test_fused_sv_vs_reference()
    print()

    test_fused_cache_attention()
    print()

    test_gqa_support()
    print()

    test_all_bit_widths()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
