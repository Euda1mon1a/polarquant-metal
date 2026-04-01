#!/usr/bin/env python3
"""
Experiment 4: Spectral Concentration for Per-Head Adaptive Bit-Width Selection

Hypothesis: Different attention heads exhibit different spectral profiles.
Heads with concentrated spectral energy (periodic attention patterns) are
predictable and tolerate cheaper 2-bit quantization. Heads with diffuse
spectral energy (uniform/complex patterns) need 3-bit or 4-bit to maintain
quality.

Inspired by AAPM's subharmonic detector: compute periodograms of attention
weight patterns per head. High spectral concentration = periodic = predictable
= cheaper quantization safe.

Key question: Can periodic/sparse heads survive 2-bit quantization with
acceptable quality (cos_sim > 0.95)?

Usage:
    cd ~/workspace/polarquant-metal
    python benchmarks/exp4_spectral_bitwidth.py
"""

import gc
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

from polarquant_metal.polar_quant import PolarQuant
from polarquant_metal.packing import pack_indices
from polarquant_metal.codebooks import load_codebook_f32
from polarquant_metal.kernels import polarquant_sv_matmul

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 8          # query heads
N_KV_HEADS = 2       # kv heads (GQA 4:1)
D = 128              # head_dim
L_KV = 8192          # 8K context
L_Q = 1              # decode mode
REP = N_HEADS // N_KV_HEADS  # 4

CONTEXT_SIZES = [8192, 16384]  # for memory savings calculation

QUALITY_THRESHOLD = 0.95  # minimum acceptable cosine similarity

# Spectral concentration thresholds for bit-width recommendation
SPECTRAL_HIGH = 0.30   # above -> 2-bit safe
SPECTRAL_MED = 0.10    # above -> 3-bit; below -> 4-bit


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def spectral_concentration(weights):
    """Compute spectral concentration of attention weights.

    Uses periodogram to get power spectral density.
    Concentration = peak_power / total_power (0 to 1).
    High = energy in few frequencies = periodic = predictable.
    Low = energy spread = diffuse = unpredictable.

    Args:
        weights: mx.array, will be squeezed to 1D (L_kv,)

    Returns:
        float: spectral concentration ratio in [0, 1]
    """
    w = np.array(weights.reshape(-1), dtype=np.float64)
    # Compute periodogram via FFT
    w_centered = w - w.mean()
    psd = np.abs(np.fft.rfft(w_centered)) ** 2
    peak = psd.max()
    total = psd.sum()
    return float(peak / total) if total > 0 else 0.0


def spectral_concentration_top_k(weights, k=5):
    """Compute fraction of spectral energy in top-k frequency bins.

    More robust than single-peak ratio for patterns with harmonics.
    """
    w = np.array(weights.reshape(-1), dtype=np.float64)
    w_centered = w - w.mean()
    psd = np.abs(np.fft.rfft(w_centered)) ** 2
    total = psd.sum()
    if total == 0:
        return 0.0
    top_k_power = np.sort(psd)[-k:].sum()
    return float(top_k_power / total)


def recommend_bits(spectral_conc):
    """Map spectral concentration to minimum safe bit-width.

    High concentration (>0.3) -> 2-bit safe (periodic, predictable)
    Medium (0.1-0.3)          -> 3-bit
    Low (<0.1)                -> 4-bit (diffuse, needs precision)
    """
    if spectral_conc > SPECTRAL_HIGH:
        return 2
    elif spectral_conc > SPECTRAL_MED:
        return 3
    else:
        return 4


# ---------------------------------------------------------------------------
# Attention weight pattern generators
# ---------------------------------------------------------------------------

def make_periodic_weights(period=64):
    """Create periodic attention: peaks at positions 0, period, 2*period, ...

    Simulates a head that attends to every Nth token (positional pattern).
    """
    weights = np.zeros((1, 1, 1, L_KV), dtype=np.float32)
    for i in range(0, L_KV, period):
        weights[0, 0, 0, i] = 1.0
    # Normalize to sum to 1
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_sparse_weights(rng, n_positions=20):
    """Create sparse attention: mass concentrated on ~n_positions random tokens.

    Simulates a retrieval head that attends to specific stored facts.
    """
    positions = rng.choice(L_KV, size=n_positions, replace=False)
    weights = np.zeros((1, 1, 1, L_KV), dtype=np.float32)
    # Assign random (positive) masses to chosen positions
    masses = rng.exponential(1.0, size=n_positions).astype(np.float32)
    weights[0, 0, 0, positions] = masses
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_diffuse_weights(rng):
    """Create roughly uniform attention (global context head).

    Uses softmax(randn * 0.05) for near-uniform distribution.
    """
    logits = rng.standard_normal((1, 1, 1, L_KV)).astype(np.float32) * 0.05
    logits -= logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_realistic_8head_weights(rng):
    """Create 8 heads with varied attention patterns for realistic simulation.

    Head 0: strong periodic (every 32 tokens)
    Head 1: strong periodic (every 128 tokens)
    Head 2: sparse retrieval (10 positions)
    Head 3: sparse retrieval (50 positions)
    Head 4: decaying (recent bias, exponential falloff)
    Head 5: bursty (clusters of attention)
    Head 6: nearly uniform (global context)
    Head 7: nearly uniform with slight recency
    """
    all_weights = np.zeros((1, N_HEADS, 1, L_KV), dtype=np.float32)

    # Head 0: periodic every 32
    for i in range(0, L_KV, 32):
        all_weights[0, 0, 0, i] = 1.0

    # Head 1: periodic every 128
    for i in range(0, L_KV, 128):
        all_weights[0, 1, 0, i] = 1.0

    # Head 2: sparse 10 positions
    pos2 = rng.choice(L_KV, size=10, replace=False)
    all_weights[0, 2, 0, pos2] = rng.exponential(1.0, size=10).astype(np.float32)

    # Head 3: sparse 50 positions
    pos3 = rng.choice(L_KV, size=50, replace=False)
    all_weights[0, 3, 0, pos3] = rng.exponential(1.0, size=50).astype(np.float32)

    # Head 4: exponential decay (recency bias)
    decay = np.exp(-np.arange(L_KV, dtype=np.float64)[::-1] / 200.0).astype(np.float32)
    all_weights[0, 4, 0, :] = decay

    # Head 5: bursty (5 clusters of 20 consecutive tokens)
    for _ in range(5):
        start = rng.randint(0, L_KV - 20)
        all_weights[0, 5, 0, start:start+20] += rng.exponential(
            1.0, size=20
        ).astype(np.float32)

    # Head 6: nearly uniform
    all_weights[0, 6, 0, :] = 1.0 + rng.standard_normal(L_KV).astype(
        np.float32
    ) * 0.01

    # Head 7: uniform with slight recency
    base = np.ones(L_KV, dtype=np.float32)
    recency = np.linspace(0.8, 1.2, L_KV, dtype=np.float32)
    all_weights[0, 7, 0, :] = base * recency

    # Normalize each head independently
    for h in range(N_HEADS):
        # Clamp negatives from noise
        all_weights[0, h, 0, :] = np.maximum(all_weights[0, h, 0, :], 0.0)
        s = all_weights[0, h, 0, :].sum()
        if s > 0:
            all_weights[0, h, 0, :] /= s

    return mx.array(all_weights)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    na = mx.sqrt(mx.sum(a_flat * a_flat))
    nb = mx.sqrt(mx.sum(b_flat * b_flat))
    return float(dot / (na * nb + 1e-10))


# ---------------------------------------------------------------------------
# FP16 baseline SV matmul
# ---------------------------------------------------------------------------

def fp16_sv_matmul(weights, values):
    """Standard FP16: weights @ V (with GQA expansion if needed).

    weights: (B, n_heads, L_Q, L_KV)
    values:  (B, N_KV_HEADS, L_KV, D)
    """
    n_heads = weights.shape[1]
    n_kv = values.shape[1]
    rep = n_heads // n_kv
    if rep > 1:
        values_exp = mx.repeat(values, rep, axis=1)
    else:
        values_exp = values
    return weights @ values_exp


# ---------------------------------------------------------------------------
# Quantized SV pipeline
# ---------------------------------------------------------------------------

def quantize_v(values_raw, bits):
    """Quantize V vectors and return everything the SV kernel needs."""
    pq = PolarQuant(bits=bits, dim=D, seed=43)
    v_idx, v_norms = pq.quantize(values_raw)
    v_packed = pack_indices(v_idx, bits)
    v_centroids = load_codebook_f32(bits, D)
    mx.eval(v_packed, v_norms, v_centroids)
    return pq, v_packed, v_norms, v_centroids


def run_sv_kernel(weights, v_packed, v_norms, v_centroids, pq, bits):
    """Run fused SV kernel and inverse-rotate output."""
    out_rot = polarquant_sv_matmul(
        weights=weights,
        v_indices=v_packed,
        v_norms=v_norms,
        v_centroids=v_centroids,
        head_dim=D,
        bits=bits,
        sparse_v_threshold=0.0,
    )
    return out_rot @ pq.rotation


def compute_memory_bytes(n_kv_heads, seq_len, head_dim, bits):
    """Compute V cache memory for a given bit-width.

    Storage: packed indices (uint32) + norms (float16)
    """
    vals_per_int = 32 // bits
    d_packed = (head_dim + vals_per_int - 1) // vals_per_int
    # packed indices: n_kv_heads * seq_len * d_packed * 4 bytes (uint32)
    packed_bytes = n_kv_heads * seq_len * d_packed * 4
    # norms: n_kv_heads * seq_len * 1 * 2 bytes (float16)
    norm_bytes = n_kv_heads * seq_len * 2
    return packed_bytes + norm_bytes


# ---------------------------------------------------------------------------
# Per-pattern quality test
# ---------------------------------------------------------------------------

def test_pattern_at_bits(pattern_name, weights_1head, values_raw, bits_list):
    """Test a single attention pattern at multiple bit-widths.

    weights_1head: (1, 1, 1, L_KV) single-head attention weights
    values_raw:    (1, N_KV_HEADS, L_KV, D) raw V vectors

    For each bit-width, quantize V, compute SV matmul, compare to FP16.

    Returns:
        dict with spectral info + per-bitwidth quality metrics
    """
    # Spectral analysis
    sc = spectral_concentration(weights_1head)
    sc_top5 = spectral_concentration_top_k(weights_1head, k=5)
    rec_bits = recommend_bits(sc)

    # FP16 baseline: use only 1 kv head (head 0) for single-head test
    v_1kv = values_raw[:, 0:1, :, :]  # (1, 1, L_KV, D)
    fp16_out = weights_1head @ v_1kv   # (1, 1, 1, D)
    mx.eval(fp16_out)

    results = {
        "pattern": pattern_name,
        "spectral_conc": sc,
        "spectral_top5": sc_top5,
        "recommended_bits": rec_bits,
        "bit_results": {},
    }

    for bits in bits_list:
        pq, v_packed, v_norms, v_centroids = quantize_v(v_1kv, bits)

        # Run fused SV kernel
        out_rot = polarquant_sv_matmul(
            weights=weights_1head,
            v_indices=v_packed,
            v_norms=v_norms,
            v_centroids=v_centroids,
            head_dim=D,
            bits=bits,
            sparse_v_threshold=0.0,
        )
        quant_out = out_rot @ pq.rotation
        mx.eval(quant_out)

        cs = cosine_similarity(fp16_out, quant_out)
        mem = compute_memory_bytes(1, L_KV, D, bits)

        results["bit_results"][bits] = {
            "cos_sim": cs,
            "memory_bytes": mem,
        }

    return results


# ---------------------------------------------------------------------------
# Multi-head adaptive bit-width test
# ---------------------------------------------------------------------------

def test_adaptive_bitwidth(head_names, all_weights, values_raw):
    """Test adaptive bit-width selection across 8 heads.

    For each head:
    1. Compute spectral concentration
    2. Recommend bit-width
    3. Quantize V at recommended bits
    4. Compute quality vs FP16

    Compare total memory: uniform 3-bit vs adaptive.

    all_weights: (1, N_HEADS, 1, L_KV) with per-head patterns
    values_raw:  (1, N_KV_HEADS, L_KV, D)
    """
    # FP16 baseline for all heads
    fp16_out = fp16_sv_matmul(all_weights, values_raw)
    mx.eval(fp16_out)

    head_info = []

    for h in range(N_HEADS):
        w_h = all_weights[:, h:h+1, :, :]  # (1, 1, 1, L_KV)
        sc = spectral_concentration(w_h)
        sc5 = spectral_concentration_top_k(w_h, k=5)
        rec = recommend_bits(sc)
        kv_h = h // REP  # GQA mapping

        # Quality at recommended bits
        v_kv = values_raw[:, kv_h:kv_h+1, :, :]
        pq_rec, vp_rec, vn_rec, vc_rec = quantize_v(v_kv, rec)
        out_rot_rec = polarquant_sv_matmul(
            weights=w_h, v_indices=vp_rec, v_norms=vn_rec,
            v_centroids=vc_rec, head_dim=D, bits=rec,
        )
        out_rec = out_rot_rec @ pq_rec.rotation
        mx.eval(out_rec)

        fp16_h = fp16_out[:, h:h+1, :, :]
        cs_rec = cosine_similarity(fp16_h, out_rec)

        # Quality at uniform 3-bit for comparison
        pq_3, vp_3, vn_3, vc_3 = quantize_v(v_kv, 3)
        out_rot_3 = polarquant_sv_matmul(
            weights=w_h, v_indices=vp_3, v_norms=vn_3,
            v_centroids=vc_3, head_dim=D, bits=3,
        )
        out_3 = out_rot_3 @ pq_3.rotation
        mx.eval(out_3)
        cs_3 = cosine_similarity(fp16_h, out_3)

        head_info.append({
            "head": h,
            "name": head_names[h],
            "kv_head": kv_h,
            "spectral_conc": sc,
            "spectral_top5": sc5,
            "recommended_bits": rec,
            "cos_sim_recommended": cs_rec,
            "cos_sim_3bit": cs_3,
        })

    return head_info


# ---------------------------------------------------------------------------
# Memory savings calculation
# ---------------------------------------------------------------------------

def compute_memory_savings(head_info, context_lengths):
    """Compute memory savings from adaptive vs uniform 3-bit.

    Since KV heads are shared (GQA), the V cache bit-width per KV head
    must be the MAX of all query heads sharing it. This is the conservative
    approach -- a KV head is quantized at the highest bit-width any of its
    query heads requires.
    """
    savings = {}
    for L in context_lengths:
        # Uniform 3-bit: all KV heads at 3-bit
        uniform_bytes = compute_memory_bytes(N_KV_HEADS, L, D, 3)

        # Adaptive: per-KV-head bit-width = max of sharing query heads
        adaptive_bytes = 0
        kv_head_bits = {}
        for info in head_info:
            kv_h = info["kv_head"]
            rec = info["recommended_bits"]
            if kv_h not in kv_head_bits:
                kv_head_bits[kv_h] = rec
            else:
                kv_head_bits[kv_h] = max(kv_head_bits[kv_h], rec)

        for kv_h in range(N_KV_HEADS):
            bits = kv_head_bits.get(kv_h, 3)
            adaptive_bytes += compute_memory_bytes(1, L, D, bits)

        # Also compute "optimistic" adaptive: if we could set bits per query
        # head (requires per-head V cache, not GQA-shared)
        optimistic_bytes = 0
        for info in head_info:
            rec = info["recommended_bits"]
            # Each query head would need its own V cache slice
            optimistic_bytes += compute_memory_bytes(1, L, D, rec) // REP

        savings[L] = {
            "uniform_3bit": uniform_bytes,
            "adaptive_gqa": adaptive_bytes,
            "adaptive_optimistic": optimistic_bytes,
            "kv_head_bits": dict(kv_head_bits),
            "saving_gqa_pct": (1.0 - adaptive_bytes / uniform_bytes) * 100
            if uniform_bytes > 0 else 0.0,
            "saving_optimistic_pct": (1.0 - optimistic_bytes / uniform_bytes) * 100
            if uniform_bytes > 0 else 0.0,
        }

    return savings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Experiment 4: Spectral Concentration for Per-Head Adaptive Bit-Width")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), D={D}")
    print(f"Context:    L_kv={L_KV:,}, L_q={L_Q} (decode)")
    print(f"Bit-widths: 2, 3, 4 (codebook levels: 4, 8, 16)")
    print(f"Quality:    cos_sim threshold = {QUALITY_THRESHOLD}")
    print(f"Spectral:   high>{SPECTRAL_HIGH} (2-bit), "
          f"med>{SPECTRAL_MED} (3-bit), low (4-bit)")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print(f"Note:       TurboQuantKVCache supports bits_v param for "
          f"separate K/V bit-widths")
    print()

    rng = np.random.RandomState(42)
    bits_list = [2, 3, 4]

    # Shared V values
    values_raw = mx.array(
        rng.standard_normal((B, N_KV_HEADS, L_KV, D)).astype(np.float32)
    )
    mx.eval(values_raw)

    # ==================================================================
    # PART 1: Per-Pattern Spectral Analysis + Quality at Different Bits
    # ==================================================================
    print("\n" + "=" * 78)
    print("  PART 1: Individual Pattern Analysis")
    print("=" * 78)

    patterns = [
        ("Periodic-64", make_periodic_weights(period=64)),
        ("Periodic-128", make_periodic_weights(period=128)),
        ("Sparse-20", make_sparse_weights(rng, n_positions=20)),
        ("Sparse-100", make_sparse_weights(rng, n_positions=100)),
        ("Diffuse", make_diffuse_weights(rng)),
    ]

    pattern_results = []
    for name, w in patterns:
        mx.eval(w)
        r = test_pattern_at_bits(name, w, values_raw, bits_list)
        pattern_results.append(r)

        print(f"\n  {name}:")
        print(f"    Spectral concentration: {r['spectral_conc']:.4f} "
              f"(top-5: {r['spectral_top5']:.4f})")
        print(f"    Recommended bit-width:  {r['recommended_bits']}-bit")
        for bits in bits_list:
            br = r["bit_results"][bits]
            mem_kb = br["memory_bytes"] / 1024
            safe = "PASS" if br["cos_sim"] >= QUALITY_THRESHOLD else "FAIL"
            print(f"    {bits}-bit: cos_sim={br['cos_sim']:.6f}  "
                  f"mem={mem_kb:.1f} KB  [{safe}]")

    # Summary table
    print(f"\n\n  {'Pattern':<14s} | {'Spec.Conc':>9s} | {'Rec':>3s} | "
          f"{'2-bit cos':>10s} | {'3-bit cos':>10s} | {'4-bit cos':>10s}")
    print(f"  {'-'*14}-+-{'-'*9}-+-{'-'*3}-+-"
          f"{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for r in pattern_results:
        c2 = r["bit_results"][2]["cos_sim"]
        c3 = r["bit_results"][3]["cos_sim"]
        c4 = r["bit_results"][4]["cos_sim"]
        mark2 = "*" if c2 >= QUALITY_THRESHOLD else " "
        mark3 = "*" if c3 >= QUALITY_THRESHOLD else " "
        mark4 = "*" if c4 >= QUALITY_THRESHOLD else " "
        print(f"  {r['pattern']:<14s} | {r['spectral_conc']:>9.4f} | "
              f"{r['recommended_bits']:>3d} | "
              f"{c2:>9.6f}{mark2} | {c3:>9.6f}{mark3} | {c4:>9.6f}{mark4}")
    print("  (* = passes quality threshold)")

    gc.collect()

    # ==================================================================
    # PART 2: Realistic 8-Head Adaptive Bit-Width Selection
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 2: Realistic 8-Head Adaptive Bit-Width")
    print("=" * 78)

    head_names = [
        "periodic-32", "periodic-128", "sparse-10", "sparse-50",
        "decay", "bursty", "uniform", "uniform+recency",
    ]
    all_weights = make_realistic_8head_weights(rng)
    mx.eval(all_weights)

    head_info = test_adaptive_bitwidth(head_names, all_weights, values_raw)

    print(f"\n  {'Head':>4s} {'Name':<16s} | {'Spec.Conc':>9s} | {'Rec':>3s} | "
          f"{'CosSim Rec':>10s} | {'CosSim 3b':>10s} | {'Delta':>8s}")
    print(f"  {'-'*4} {'-'*16}-+-{'-'*9}-+-{'-'*3}-+-"
          f"{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for info in head_info:
        delta = info["cos_sim_recommended"] - info["cos_sim_3bit"]
        safe = "OK" if info["cos_sim_recommended"] >= QUALITY_THRESHOLD else "LOW"
        print(f"  H{info['head']:>2d} {info['name']:<16s} | "
              f"{info['spectral_conc']:>9.4f} | {info['recommended_bits']:>3d} | "
              f"{info['cos_sim_recommended']:>9.6f} | {info['cos_sim_3bit']:>9.6f} | "
              f"{delta:>+8.4f} [{safe}]")

    gc.collect()

    # ==================================================================
    # PART 3: Memory Savings
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 3: Memory Savings (V Cache Only)")
    print("=" * 78)

    savings = compute_memory_savings(head_info, CONTEXT_SIZES)

    for L in CONTEXT_SIZES:
        s = savings[L]
        print(f"\n  Context length: {L:,}")
        print(f"    KV head bit assignments: {s['kv_head_bits']}")
        u_mb = s["uniform_3bit"] / (1024 * 1024)
        a_mb = s["adaptive_gqa"] / (1024 * 1024)
        o_mb = s["adaptive_optimistic"] / (1024 * 1024)
        print(f"    Uniform 3-bit:           {u_mb:>8.2f} MB")
        print(f"    Adaptive (GQA-safe):     {a_mb:>8.2f} MB  "
              f"({s['saving_gqa_pct']:+.1f}%)")
        print(f"    Adaptive (per-head opt): {o_mb:>8.2f} MB  "
              f"({s['saving_optimistic_pct']:+.1f}%)")
        print(f"    GQA saving:              "
              f"{(s['uniform_3bit'] - s['adaptive_gqa']) / 1024:.1f} KB")

    # ==================================================================
    # PART 4: Threshold Validation
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 4: Threshold Validation")
    print("=" * 78)
    print(f"\n  Checking if spectral thresholds correctly predict 2-bit safety:")
    print(f"  (Quality threshold: cos_sim >= {QUALITY_THRESHOLD})\n")

    correct = 0
    total = 0
    for r in pattern_results:
        sc = r["spectral_conc"]
        rec = r["recommended_bits"]
        # Check: if rec=2, does 2-bit actually pass quality?
        # If rec=3 or 4, does 2-bit actually fail?
        c2 = r["bit_results"][2]["cos_sim"]
        passes_2bit = c2 >= QUALITY_THRESHOLD
        predicted_2bit_safe = (rec == 2)
        is_correct = (predicted_2bit_safe == passes_2bit)
        correct += int(is_correct)
        total += 1
        mark = "CORRECT" if is_correct else "WRONG"
        print(f"    {r['pattern']:<14s}: SC={sc:.4f}, rec={rec}-bit, "
              f"2-bit cos={c2:.4f} {'PASS' if passes_2bit else 'FAIL'} "
              f"-> [{mark}]")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n  Prediction accuracy: {correct}/{total} ({accuracy:.0f}%)")

    # Key finding: can periodic/sparse heads survive 2-bit?
    print(f"\n  KEY QUESTION: Can periodic/sparse heads survive 2-bit?")
    for r in pattern_results:
        if "Periodic" in r["pattern"] or "Sparse" in r["pattern"]:
            c2 = r["bit_results"][2]["cos_sim"]
            verdict = "YES" if c2 >= QUALITY_THRESHOLD else "NO"
            print(f"    {r['pattern']}: {verdict} (cos_sim={c2:.6f})")

    # ==================================================================
    # PART 5: TurboQuantKVCache bits_v Integration Note
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 5: Integration with TurboQuantKVCache")
    print("=" * 78)
    print(f"""
  TurboQuantKVCache already supports separate K/V bit-widths:
    cache = TurboQuantKVCache(bits=3, bits_v=2)

  This means adaptive V bit-width is deployable TODAY:
  1. At model load, profile attention patterns on calibration data
  2. Compute spectral concentration per head
  3. Set bits_v per layer based on head pattern analysis
  4. Heads with high spectral concentration -> bits_v=2
  5. Heads with low spectral concentration -> bits_v=3

  Limitation: bits_v is per-cache (per-layer), not per-head within a layer.
  For true per-head adaptive bits, the kernel would need per-head bit params.
  With GQA, per-KV-head bits is the practical granularity.
""")

    # ==================================================================
    # Save results to markdown
    # ==================================================================
    md_path = os.path.join(os.path.dirname(__file__), "EXP4_RESULTS.md")
    _write_results_md(md_path, pattern_results, head_info, savings, accuracy)
    print(f"  Results saved to: {md_path}")

    print(f"\n{'='*78}")
    print("  EXPERIMENT 4 COMPLETE")
    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _write_results_md(md_path, pattern_results, head_info, savings, accuracy):
    with open(md_path, "w") as f:
        f.write("# Experiment 4: Spectral Concentration for "
                "Per-Head Adaptive Bit-Width\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} "
                f"(GQA {REP}:1), D={D}  \n")
        f.write(f"**Context:** L_kv={L_KV:,}, L_q={L_Q} (decode)  \n")
        f.write(f"**Bit-widths tested:** 2, 3, 4  \n")
        f.write(f"**Quality threshold:** cos_sim >= {QUALITY_THRESHOLD}  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        # Hypothesis
        f.write("## Hypothesis\n\n")
        f.write("Different attention heads have different spectral profiles. "
                "Heads with concentrated spectral energy (periodic patterns) "
                "are predictable and tolerate cheaper 2-bit quantization. "
                "Heads with diffuse spectral energy need 3-bit or more. "
                "By analyzing the power spectral density of attention weights, "
                "we can adaptively select bit-widths per head, saving memory "
                "without sacrificing quality.\n\n")

        # Part 1: Pattern Analysis
        f.write("## Part 1: Individual Pattern Analysis\n\n")
        f.write("| Pattern | Spectral Conc. | Top-5 Conc. | Rec. Bits | "
                "2-bit CosSim | 3-bit CosSim | 4-bit CosSim |\n")
        f.write("|:--------|---------------:|------------:|----------:|"
                "-------------:|-------------:|-------------:|\n")
        for r in pattern_results:
            c2 = r["bit_results"][2]["cos_sim"]
            c3 = r["bit_results"][3]["cos_sim"]
            c4 = r["bit_results"][4]["cos_sim"]
            f.write(f"| {r['pattern']} | {r['spectral_conc']:.4f} | "
                    f"{r['spectral_top5']:.4f} | {r['recommended_bits']} | "
                    f"{c2:.6f} | {c3:.6f} | {c4:.6f} |\n")

        # Memory per pattern
        f.write("\n### Memory per Pattern (1 KV head, L_kv=8192)\n\n")
        f.write("| Pattern | 2-bit (KB) | 3-bit (KB) | 4-bit (KB) | "
                "FP16 (KB) |\n")
        f.write("|:--------|----------:|-----------:|-----------:|"
                "----------:|\n")
        fp16_bytes = N_KV_HEADS * L_KV * D * 2  # float16 = 2 bytes
        for r in pattern_results:
            m2 = r["bit_results"][2]["memory_bytes"] / 1024
            m3 = r["bit_results"][3]["memory_bytes"] / 1024
            m4 = r["bit_results"][4]["memory_bytes"] / 1024
            fp16_kb = (1 * L_KV * D * 2) / 1024  # 1 kv head
            f.write(f"| {r['pattern']} | {m2:.1f} | {m3:.1f} | "
                    f"{m4:.1f} | {fp16_kb:.1f} |\n")

        # Part 2: 8-Head Adaptive
        f.write("\n## Part 2: Realistic 8-Head Adaptive Bit-Width\n\n")
        f.write("| Head | Name | KV Head | Spectral Conc. | "
                "Rec. Bits | CosSim Rec. | CosSim 3-bit | Delta |\n")
        f.write("|-----:|:-----|--------:|---------------:|"
                "----------:|------------:|-------------:|------:|\n")
        for info in head_info:
            delta = info["cos_sim_recommended"] - info["cos_sim_3bit"]
            f.write(f"| H{info['head']} | {info['name']} | "
                    f"KV{info['kv_head']} | {info['spectral_conc']:.4f} | "
                    f"{info['recommended_bits']} | "
                    f"{info['cos_sim_recommended']:.6f} | "
                    f"{info['cos_sim_3bit']:.6f} | {delta:+.4f} |\n")

        # Part 3: Memory savings
        f.write("\n## Part 3: Memory Savings (V Cache Only)\n\n")
        for L in CONTEXT_SIZES:
            s = savings[L]
            u_mb = s["uniform_3bit"] / (1024 * 1024)
            a_mb = s["adaptive_gqa"] / (1024 * 1024)
            o_mb = s["adaptive_optimistic"] / (1024 * 1024)
            f.write(f"### Context length: {L:,}\n\n")
            f.write(f"KV head bit assignments: {s['kv_head_bits']}\n\n")
            f.write("| Strategy | Memory | Saving |\n")
            f.write("|:---------|-------:|-------:|\n")
            f.write(f"| Uniform 3-bit | {u_mb:.2f} MB | baseline |\n")
            f.write(f"| Adaptive (GQA-safe) | {a_mb:.2f} MB | "
                    f"{s['saving_gqa_pct']:+.1f}% |\n")
            f.write(f"| Adaptive (per-head optimistic) | {o_mb:.2f} MB | "
                    f"{s['saving_optimistic_pct']:+.1f}% |\n\n")

        # Part 4: Threshold validation
        f.write("## Part 4: Threshold Validation\n\n")
        f.write(f"Spectral concentration thresholds: "
                f"high > {SPECTRAL_HIGH} (2-bit), "
                f"medium > {SPECTRAL_MED} (3-bit), "
                f"low (4-bit)\n\n")
        f.write("| Pattern | Spectral Conc. | Recommended | "
                "2-bit CosSim | 2-bit Safe? | Prediction |\n")
        f.write("|:--------|---------------:|:------------|"
                "-------------:|:------------|:-----------|\n")
        for r in pattern_results:
            sc = r["spectral_conc"]
            rec = r["recommended_bits"]
            c2 = r["bit_results"][2]["cos_sim"]
            passes = c2 >= QUALITY_THRESHOLD
            predicted_safe = (rec == 2)
            correct = (predicted_safe == passes)
            f.write(f"| {r['pattern']} | {sc:.4f} | {rec}-bit | "
                    f"{c2:.6f} | {'Yes' if passes else 'No'} | "
                    f"{'Correct' if correct else 'Wrong'} |\n")
        f.write(f"\n**Prediction accuracy: {accuracy:.0f}%**\n\n")

        # Part 5: Integration note
        f.write("## Part 5: TurboQuantKVCache Integration\n\n")
        f.write("`TurboQuantKVCache` already supports separate K/V bit-widths "
                "via the `bits_v` parameter:\n\n")
        f.write("```python\n")
        f.write("cache = TurboQuantKVCache(bits=3, bits_v=2)  # K=3-bit, V=2-bit\n")
        f.write("```\n\n")
        f.write("This means adaptive V bit-width is deployable now. "
                "The `bits_v` parameter is per-cache (per-layer), not per-head. "
                "For true per-head adaptive bits within a layer, the kernel "
                "would need per-head bit-width parameters. With GQA, "
                "per-KV-head is the practical granularity.\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")

        # Check key findings
        periodic_pass_2bit = all(
            r["bit_results"][2]["cos_sim"] >= QUALITY_THRESHOLD
            for r in pattern_results if "Periodic" in r["pattern"]
        )
        diffuse_fail_2bit = all(
            r["bit_results"][2]["cos_sim"] < QUALITY_THRESHOLD
            for r in pattern_results if r["pattern"] == "Diffuse"
        )
        any_gqa_saving = any(
            savings[L]["saving_gqa_pct"] > 0 for L in CONTEXT_SIZES
        )

        f.write("### Key Findings\n\n")
        if periodic_pass_2bit:
            f.write("- **Periodic heads survive 2-bit quantization** with "
                    "acceptable quality (cos_sim >= 0.95). Spectral "
                    "concentration correctly identifies these heads.\n")
        else:
            f.write("- Periodic heads show varied tolerance for 2-bit "
                    "quantization. Spectral concentration alone may not "
                    "be sufficient to predict 2-bit safety.\n")

        if diffuse_fail_2bit:
            f.write("- **Diffuse heads need higher bit-widths** as expected. "
                    "2-bit quantization degrades quality for uniform "
                    "attention patterns.\n")
        else:
            f.write("- Diffuse heads show unexpectedly good 2-bit quality, "
                    "suggesting PolarQuant's rotation decorrelation helps "
                    "even for uniform patterns.\n")

        f.write(f"- Prediction accuracy: {accuracy:.0f}% across "
                f"{len(pattern_results)} test patterns.\n")

        if any_gqa_saving:
            max_s = max(savings[L]["saving_gqa_pct"] for L in CONTEXT_SIZES)
            f.write(f"- GQA-safe adaptive bit-width saves up to "
                    f"{max_s:.1f}% V cache memory.\n")
        else:
            f.write("- GQA constraints limit memory savings when query heads "
                    "sharing a KV head have mixed patterns (the KV head must "
                    "use the highest bit-width any sharing head needs).\n")

        f.write("\n### Verdict\n\n")
        if periodic_pass_2bit and accuracy >= 80:
            f.write("**POSITIVE**: Spectral concentration is a viable signal "
                    "for adaptive bit-width selection. Periodic and sparse "
                    "heads tolerate 2-bit quantization, enabling memory "
                    "savings for models with diverse attention patterns. "
                    "The approach integrates naturally with "
                    "`TurboQuantKVCache(bits_v=...)` at the per-layer level "
                    "and could extend to per-KV-head granularity.\n")
        elif accuracy >= 60:
            f.write("**MIXED**: Spectral concentration partially predicts "
                    "2-bit safety but thresholds need refinement. "
                    "Consider combining spectral analysis with entropy or "
                    "kurtosis for a more robust predictor.\n")
        else:
            f.write("**NEGATIVE**: Spectral concentration alone is "
                    "insufficient for reliable bit-width prediction. "
                    "PolarQuant's rotation decorrelation may make all "
                    "patterns similarly robust (or fragile) to quantization, "
                    "reducing the value of per-head adaptation.\n")


if __name__ == "__main__":
    main()
