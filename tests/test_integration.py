#!/usr/bin/env python3
"""
End-to-end integration test: TurboQuantKVCache with fused Metal kernels.

Tests:
1. Cache update/fetch roundtrip
2. Fused vs dequantized attention equivalence
3. Multi-step decode simulation
4. Quality: cosine similarity of fused output vs FP16 reference
5. Real model test (if mlx-lm model available)
"""

import math
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".")
from polarquant_metal.turboquant_cache import TurboQuantKVCache


def test_basic_update_fetch():
    """Test that update_and_fetch works and returns correct shapes."""
    for bits in [2, 3, 4]:
        # Fused mode: returns packed arrays
        cache = TurboQuantKVCache(bits=bits, fused=True)
        assert cache.empty()

        k = mx.random.normal(shape=(1, 8, 16, 128))
        v = mx.random.normal(shape=(1, 8, 16, 128))
        ret_k, ret_v = cache.update_and_fetch(k, v)
        mx.eval(ret_k, ret_v)

        assert cache.size() == 16
        assert not cache.empty()
        assert ret_k.dtype == mx.uint32  # packed indices, not float

        # Decode step
        k2 = mx.random.normal(shape=(1, 8, 1, 128))
        v2 = mx.random.normal(shape=(1, 8, 1, 128))
        cache.update_and_fetch(k2, v2)

        assert cache.size() == 17

        # fused_sdpa should work
        q = mx.random.normal(shape=(1, 8, 1, 128))
        out = cache.fused_sdpa(q)
        mx.eval(out)
        assert out.shape == (1, 8, 1, 128)
        print(f"  {bits}-bit: fused OK, offset={cache.size()}")

        # Non-fused mode: returns dequantized floats
        cache_deq = TurboQuantKVCache(bits=bits, fused=False)
        dk, dv = cache_deq.update_and_fetch(k, v)
        mx.eval(dk, dv)
        assert dk.shape == (1, 8, 16, 128)
        assert dk.dtype != mx.uint32
        print(f"  {bits}-bit: dequant OK")

    print("PASS: test_basic_update_fetch")


def test_fused_vs_dequantized():
    """Compare fused_sdpa output vs manual dequantized attention."""
    B, n_heads, n_kv_heads, L_kv, D = 1, 32, 8, 64, 128
    bits = 3
    scale = 1.0 / math.sqrt(D)
    rep = n_heads // n_kv_heads

    np.random.seed(42)
    queries = mx.array(np.random.randn(B, n_heads, 1, D).astype(np.float32))
    keys = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    values = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))

    # Dequantized reference path (fused=False)
    cache_deq = TurboQuantKVCache(bits=bits, fused=False)
    dk, dv = cache_deq.update_and_fetch(keys, values)
    mx.eval(dk, dv)

    dk_exp = mx.repeat(dk, rep, axis=1)
    dv_exp = mx.repeat(dv, rep, axis=1)
    scores_ref = (queries @ mx.swapaxes(dk_exp, -2, -1)) * scale
    weights_ref = mx.softmax(scores_ref, axis=-1, precise=True)
    out_ref = weights_ref @ dv_exp
    mx.eval(out_ref)

    # Fused path (fused=True)
    cache_fused = TurboQuantKVCache(bits=bits, fused=True)
    cache_fused.update_and_fetch(keys, values)  # returns packed, ignored
    out_fused = cache_fused.fused_sdpa(queries, scale=scale)
    mx.eval(out_fused)

    cos_sim = float(mx.mean(
        mx.sum(out_ref * out_fused, axis=-1) /
        (mx.linalg.norm(out_ref, axis=-1) * mx.linalg.norm(out_fused, axis=-1) + 1e-8)
    ))
    max_diff = float(mx.max(mx.abs(out_ref - out_fused)))

    print(f"  Fused vs dequantized: max_diff={max_diff:.6e}, cos_sim={cos_sim:.8f}")
    assert cos_sim > 0.999, f"Cosine similarity too low: {cos_sim}"
    assert max_diff < 0.01, f"Max diff too high: {max_diff}"
    print("PASS: test_fused_vs_dequantized")


def test_multi_step_decode():
    """Simulate a multi-step decode loop."""
    B, n_heads, n_kv_heads, D = 1, 32, 8, 128
    bits = 3

    cache = TurboQuantKVCache(bits=bits, fused=True)

    # Prefill 32 tokens
    k = mx.random.normal(shape=(B, n_kv_heads, 32, D))
    v = mx.random.normal(shape=(B, n_kv_heads, 32, D))
    cache.update_and_fetch(k, v)
    assert cache.size() == 32

    # Decode 10 tokens one at a time
    for step in range(10):
        k_new = mx.random.normal(shape=(B, n_kv_heads, 1, D))
        v_new = mx.random.normal(shape=(B, n_kv_heads, 1, D))
        cache.update_and_fetch(k_new, v_new)

        q = mx.random.normal(shape=(B, n_heads, 1, D))
        out = cache.fused_sdpa(q)
        mx.eval(out)

        assert out.shape == (B, n_heads, 1, D)
        assert cache.size() == 33 + step

    print(f"  10 decode steps: final cache size={cache.size()}")
    print("PASS: test_multi_step_decode")


def test_quality_vs_fp16():
    """Compare fused PolarQuant attention vs full FP16 reference."""
    B, n_heads, n_kv_heads, L_kv, D = 1, 32, 8, 128, 128
    rep = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(D)

    np.random.seed(123)
    queries = mx.array(np.random.randn(B, n_heads, 1, D).astype(np.float32))
    keys = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))
    values = mx.array(np.random.randn(B, n_kv_heads, L_kv, D).astype(np.float32))

    # FP16 reference (no quantization at all)
    k_exp = mx.repeat(keys, rep, axis=1)
    v_exp = mx.repeat(values, rep, axis=1)
    scores_fp16 = (queries @ mx.swapaxes(k_exp, -2, -1)) * scale
    weights_fp16 = mx.softmax(scores_fp16, axis=-1, precise=True)
    out_fp16 = weights_fp16 @ v_exp
    mx.eval(out_fp16)

    for bits in [3, 4]:
        cache = TurboQuantKVCache(bits=bits, fused=True)
        cache.update_and_fetch(keys, values)
        out_fused = cache.fused_sdpa(queries, scale=scale)
        mx.eval(out_fused)

        cos_sim = float(mx.mean(
            mx.sum(out_fp16 * out_fused, axis=-1) /
            (mx.linalg.norm(out_fp16, axis=-1) * mx.linalg.norm(out_fused, axis=-1) + 1e-8)
        ))
        print(f"  {bits}-bit vs FP16: cos_sim={cos_sim:.6f}")
        assert cos_sim > 0.90, f"{bits}-bit quality too low vs FP16: {cos_sim}"

    print("PASS: test_quality_vs_fp16")


def test_decode_speed():
    """Benchmark decode speed: fused (no dequant) vs full dequantized path."""
    B, n_heads, n_kv_heads, L_kv, D = 1, 32, 8, 1024, 128
    bits = 3
    rep = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(D)

    keys = mx.random.normal(shape=(B, n_kv_heads, L_kv, D))
    values = mx.random.normal(shape=(B, n_kv_heads, L_kv, D))
    queries = mx.random.normal(shape=(B, n_heads, 1, D))

    # Fused cache (skips dequant in update_and_fetch)
    cache_fused = TurboQuantKVCache(bits=bits, fused=True)
    cache_fused.update_and_fetch(keys, values)
    mx.eval(cache_fused._k_packed)

    # Dequantized cache
    cache_deq = TurboQuantKVCache(bits=bits, fused=False)
    dk, dv = cache_deq.update_and_fetch(keys, values)
    mx.eval(dk, dv)

    # Warmup fused
    for _ in range(3):
        out = cache_fused.fused_sdpa(queries, scale=scale)
        mx.eval(out)

    # Bench fused
    fused_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = cache_fused.fused_sdpa(queries, scale=scale)
        mx.eval(out)
        fused_times.append(time.perf_counter() - t0)

    # Warmup dequantized
    for _ in range(3):
        dk_exp = mx.repeat(dk, rep, axis=1)
        dv_exp = mx.repeat(dv, rep, axis=1)
        s = (queries @ mx.swapaxes(dk_exp, -2, -1)) * scale
        w = mx.softmax(s, axis=-1, precise=True)
        o = w @ dv_exp
        mx.eval(o)

    # Bench dequantized
    deq_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        dk_exp = mx.repeat(dk, rep, axis=1)
        dv_exp = mx.repeat(dv, rep, axis=1)
        s = (queries @ mx.swapaxes(dk_exp, -2, -1)) * scale
        w = mx.softmax(s, axis=-1, precise=True)
        o = w @ dv_exp
        mx.eval(o)
        deq_times.append(time.perf_counter() - t0)

    t_fused = np.median(fused_times) * 1000
    t_deq = np.median(deq_times) * 1000
    speedup = t_deq / t_fused

    print(f"  L_kv={L_kv}: fused={t_fused:.2f}ms, dequantized={t_deq:.2f}ms, speedup={speedup:.2f}x")
    print("PASS: test_decode_speed")


def test_trim():
    """Test cache trimming."""
    cache = TurboQuantKVCache(bits=3, fused=True)
    k = mx.random.normal(shape=(1, 4, 20, 64))
    v = mx.random.normal(shape=(1, 4, 20, 64))
    cache.update_and_fetch(k, v)
    assert cache.size() == 20

    trimmed = cache.trim(5)
    assert trimmed == 5
    assert cache.size() == 15

    # Should still work after trim
    q = mx.random.normal(shape=(1, 4, 1, 64))
    out = cache.fused_sdpa(q)
    mx.eval(out)
    assert out.shape == (1, 4, 1, 64)
    print("PASS: test_trim")


def test_state_save_restore():
    """Test state serialization."""
    cache = TurboQuantKVCache(bits=3, fused=True)
    k = mx.random.normal(shape=(1, 8, 16, 128))
    v = mx.random.normal(shape=(1, 8, 16, 128))
    cache.update_and_fetch(k, v)

    # Save state
    state = cache.state
    meta = cache.meta_state

    # Restore into new cache
    cache2 = TurboQuantKVCache(bits=3, fused=True)
    cache2.meta_state = meta
    cache2.state = state

    assert cache2.size() == cache.size()
    assert cache2.turbo_bits == cache.turbo_bits
    print("PASS: test_state_save_restore")


def test_sdpa_patch():
    """Test that patched SDPA correctly dispatches to fused path."""
    from polarquant_metal.integration import patch_sdpa, unpatch_sdpa
    import mlx_lm.models.base as base_module

    B, n_heads, n_kv_heads, L_kv, D = 1, 8, 4, 32, 64
    scale = 1.0 / math.sqrt(D)

    keys = mx.random.normal(shape=(B, n_kv_heads, L_kv, D))
    values = mx.random.normal(shape=(B, n_kv_heads, L_kv, D))
    queries = mx.random.normal(shape=(B, n_heads, 1, D))

    # Fused cache
    cache = TurboQuantKVCache(bits=3, fused=True)
    ret_k, ret_v = cache.update_and_fetch(keys, values)

    # Reference: direct fused_sdpa call
    out_direct = cache.fused_sdpa(queries, scale=scale)
    mx.eval(out_direct)

    # Patch SDPA and call through it
    patch_sdpa()
    out_patched = base_module.scaled_dot_product_attention(
        queries, ret_k, ret_v, cache=cache, scale=scale, mask=None,
    )
    mx.eval(out_patched)

    # Should be identical (same code path)
    max_diff = float(mx.max(mx.abs(out_direct - out_patched)))
    print(f"  Patched SDPA vs direct fused_sdpa: max_diff={max_diff:.6e}")
    assert max_diff < 1e-6, f"Patched SDPA diverges: {max_diff}"

    unpatch_sdpa()
    print("PASS: test_sdpa_patch")


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuantKVCache + Fused Metal Kernel Integration Tests")
    print("=" * 60)

    test_basic_update_fetch()
    print()
    test_fused_vs_dequantized()
    print()
    test_multi_step_decode()
    print()
    test_quality_vs_fp16()
    print()
    test_decode_speed()
    print()
    test_trim()
    print()
    test_state_save_restore()
    print()
    test_sdpa_patch()

    print()
    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
