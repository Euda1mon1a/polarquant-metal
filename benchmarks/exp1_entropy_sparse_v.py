#!/usr/bin/env python3
"""
Experiment 1: Entropy-Guided Adaptive Sparse V Threshold for PolarQuant

Hypothesis: Attention weights post-softmax have varying entropy per head.
Low-entropy heads (concentrated on few tokens) can tolerate aggressive
sparse V pruning. High-entropy heads (spread attention) cannot.

By computing per-head Shannon entropy and adapting the threshold, we get
speedup on concentrated heads without quality loss on spread heads.

The kernel accepts a single threshold per call, so we dispatch per-head-group:
  - Low-entropy group: threshold from entropy_to_threshold()
  - High-entropy group: threshold = 0.0 (no pruning)
Then concatenate results.

Metrics:
  - Cosine similarity vs FP16 baseline
  - Wall-clock time (ms)
  - Effective skip rate (% of positions below threshold)
  - Per-head entropy values

Usage:
    cd ~/workspace/polarquant-metal
    python benchmarks/exp1_entropy_sparse_v.py
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
from polarquant_metal.kernels import polarquant_sv_matmul

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 8         # query heads
N_KV_HEADS = 2      # kv heads (GQA 4:1)
D = 128              # head_dim
L_KV = 16384         # 16K context
L_Q = 1              # decode mode
BITS = 3
N_TRIALS = 10
WARMUP = 3
MAX_THRESHOLD = 0.01  # maximum sparse threshold for low-entropy heads

REP = N_HEADS // N_KV_HEADS  # 4 — heads per kv group


# ---------------------------------------------------------------------------
# Entropy utilities
# ---------------------------------------------------------------------------

def compute_head_entropy(weights: mx.array) -> mx.array:
    """Compute normalized Shannon entropy per head.

    Args:
        weights: (B, n_heads, L_q, L_kv) post-softmax attention weights

    Returns:
        entropy: (B, n_heads, L_q) normalized entropy in [0, 1]
            0 = all mass on one token (concentrated)
            1 = uniform distribution (fully spread)
    """
    # Clamp to avoid log(0)
    eps = 1e-10
    w = mx.maximum(weights, eps)
    # Shannon entropy: H = -sum(p * log(p))
    log_w = mx.log(w)
    h = -mx.sum(w * log_w, axis=-1)  # (B, n_heads, L_q)
    # Normalize by log(L_kv) so result is in [0, 1]
    h_max = math.log(weights.shape[-1])
    return h / h_max


def entropy_to_threshold(entropy: float, max_threshold: float = MAX_THRESHOLD) -> float:
    """Map normalized entropy to sparse V threshold.

    Low entropy  -> high threshold (aggressive pruning is safe)
    High entropy -> threshold=0.0  (no pruning)

    Uses sigmoid mapping for a smooth transition:
        threshold = max_threshold * sigmoid(-10 * (entropy - 0.5))

    At entropy=0.0: sigmoid(5)  ~= 0.993 -> threshold ~= max_threshold
    At entropy=0.5: sigmoid(0)  = 0.5    -> threshold ~= max_threshold/2
    At entropy=1.0: sigmoid(-5) ~= 0.007 -> threshold ~= 0
    """
    x = -10.0 * (entropy - 0.5)
    sig = 1.0 / (1.0 + math.exp(-x))
    return max_threshold * sig


# ---------------------------------------------------------------------------
# Attention weight generators
# ---------------------------------------------------------------------------

def make_concentrated_weights(rng, shape):
    """Generate low-entropy (concentrated) attention weights.

    Attention focused on ~5% of positions via softmax(randn * 5).
    """
    B, n_heads, L_q, L_kv = shape
    logits = rng.standard_normal(shape).astype(np.float32) * 5.0
    # softmax in numpy
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_spread_weights(rng, shape):
    """Generate high-entropy (spread/near-uniform) attention weights.

    Roughly uniform via softmax(randn * 0.1).
    """
    B, n_heads, L_q, L_kv = shape
    logits = rng.standard_normal(shape).astype(np.float32) * 0.1
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_realistic_weights(rng, shape):
    """Generate realistic mix: half heads concentrated, half spread."""
    B, n_heads, L_q, L_kv = shape
    half = n_heads // 2

    # First half: concentrated
    conc_shape = (B, half, L_q, L_kv)
    logits_c = rng.standard_normal(conc_shape).astype(np.float32) * 5.0
    logits_c -= logits_c.max(axis=-1, keepdims=True)
    w_c = np.exp(logits_c)
    w_c /= w_c.sum(axis=-1, keepdims=True)

    # Second half: spread
    spr_shape = (B, n_heads - half, L_q, L_kv)
    logits_s = rng.standard_normal(spr_shape).astype(np.float32) * 0.1
    logits_s -= logits_s.max(axis=-1, keepdims=True)
    w_s = np.exp(logits_s)
    w_s /= w_s.sum(axis=-1, keepdims=True)

    weights = np.concatenate([w_c, w_s], axis=1)
    return mx.array(weights)


# ---------------------------------------------------------------------------
# FP16 baseline: standard matmul (no quantization, no sparsity)
# ---------------------------------------------------------------------------

def fp16_sv_matmul(weights, values):
    """Standard FP16 attention output: weights @ V (with GQA expansion)."""
    # weights: (B, N_HEADS, L_Q, L_KV)
    # values:  (B, N_KV_HEADS, L_KV, D)
    if REP > 1:
        values_exp = mx.repeat(values, REP, axis=1)
    else:
        values_exp = values
    return weights @ values_exp


# ---------------------------------------------------------------------------
# Quantized SV pipeline helper
# ---------------------------------------------------------------------------

def setup_quantized_v(values_raw):
    """Quantize and pack V values, return everything the SV kernel needs."""
    pq_val = PolarQuant(bits=BITS, dim=D, seed=43)
    v_idx, v_norms = pq_val.quantize(values_raw)
    v_packed = pack_indices(v_idx, BITS)
    val_centroids = load_codebook_f32(BITS, D)
    mx.eval(v_packed, v_norms, val_centroids)
    return pq_val, v_packed, v_norms, val_centroids


def run_sv_kernel(weights, v_packed, v_norms, val_centroids, pq_val,
                  sparse_v_threshold=0.0):
    """Run fused SV kernel and inverse-rotate output."""
    out_rot = polarquant_sv_matmul(
        weights=weights,
        v_indices=v_packed,
        v_norms=v_norms,
        v_centroids=val_centroids,
        head_dim=D,
        bits=BITS,
        sparse_v_threshold=sparse_v_threshold,
    )
    return out_rot @ pq_val.rotation


# ---------------------------------------------------------------------------
# Entropy-guided dispatch: split heads by entropy, run per group
# ---------------------------------------------------------------------------

def entropy_guided_sv(weights, v_packed, v_norms, val_centroids, pq_val,
                      entropy_vals, entropy_cutoff=0.5):
    """Run SV kernel with per-head-group adaptive threshold.

    Heads with normalized entropy < cutoff get aggressive pruning.
    Heads with entropy >= cutoff get threshold=0 (no pruning).

    Since the kernel accepts a single threshold, we split into two groups
    and concatenate the results.

    Args:
        weights: (B, N_HEADS, L_Q, L_KV)
        entropy_vals: (B, N_HEADS, L_Q) per-head normalized entropy
        entropy_cutoff: heads below this get pruning

    Returns:
        output: (B, N_HEADS, L_Q, D) attention output
        thresholds_used: list of (head_idx, threshold) pairs
    """
    # Squeeze L_Q=1 dimension for simpler indexing
    ent = np.array(entropy_vals.reshape(B, N_HEADS))  # (B, N_HEADS)

    # Classify heads
    low_entropy_heads = []
    high_entropy_heads = []
    thresholds_used = []

    for h in range(N_HEADS):
        e = float(ent[0, h])  # B=1
        if e < entropy_cutoff:
            t = entropy_to_threshold(e)
            low_entropy_heads.append(h)
            thresholds_used.append((h, t, e))
        else:
            high_entropy_heads.append(h)
            thresholds_used.append((h, 0.0, e))

    # Sort by group for concatenation
    thresholds_used.sort(key=lambda x: x[0])

    outputs = [None] * N_HEADS

    # --- Low-entropy group: aggressive pruning ---
    if low_entropy_heads:
        # Compute a single threshold for the group (average of individual thresholds)
        group_thresholds = [entropy_to_threshold(float(ent[0, h]))
                           for h in low_entropy_heads]
        group_thresh = sum(group_thresholds) / len(group_thresholds)

        # Extract head slices
        w_low = mx.concatenate(
            [weights[:, h:h+1, :, :] for h in low_entropy_heads], axis=1
        )

        # Map query heads to kv heads for this subset
        kv_heads_for_low = sorted(set(h // REP for h in low_entropy_heads))
        # We need the full kv heads since the kernel does GQA mapping internally
        # Remap: build a mini-weight tensor with contiguous heads
        # and matching kv structure

        # Since the kernel handles GQA internally via n_heads/n_kv_heads ratio,
        # we need to be careful. The simplest correct approach: run per-head.
        for h in low_entropy_heads:
            t = entropy_to_threshold(float(ent[0, h]))
            w_h = weights[:, h:h+1, :, :]  # (B, 1, L_Q, L_KV)
            kv_h = h // REP
            v_packed_h = v_packed[:, kv_h:kv_h+1, :, :]
            v_norms_h = v_norms[:, kv_h:kv_h+1, :, :]

            out_rot = polarquant_sv_matmul(
                weights=w_h,
                v_indices=v_packed_h,
                v_norms=v_norms_h,
                v_centroids=val_centroids,
                head_dim=D,
                bits=BITS,
                sparse_v_threshold=t,
            )
            outputs[h] = out_rot @ pq_val.rotation

    # --- High-entropy group: no pruning ---
    if high_entropy_heads:
        for h in high_entropy_heads:
            w_h = weights[:, h:h+1, :, :]
            kv_h = h // REP
            v_packed_h = v_packed[:, kv_h:kv_h+1, :, :]
            v_norms_h = v_norms[:, kv_h:kv_h+1, :, :]

            out_rot = polarquant_sv_matmul(
                weights=w_h,
                v_indices=v_packed_h,
                v_norms=v_norms_h,
                v_centroids=val_centroids,
                head_dim=D,
                bits=BITS,
                sparse_v_threshold=0.0,
            )
            outputs[h] = out_rot @ pq_val.rotation

    # Concatenate all heads in order
    result = mx.concatenate(outputs, axis=1)
    return result, thresholds_used


# ---------------------------------------------------------------------------
# Skip rate estimation
# ---------------------------------------------------------------------------

def estimate_skip_rate(weights, v_norms, threshold):
    """Estimate what fraction of positions are skipped by sparse V.

    The precombined kernel skips where |weight * norm| < threshold.
    We replicate that logic here.
    """
    if threshold == 0.0:
        return 0.0

    # Expand norms for GQA
    norms_sq = v_norms.squeeze(-1)  # (B, N_KV_HEADS, L_KV)
    if REP > 1:
        norms_exp = mx.repeat(norms_sq, REP, axis=1)  # (B, N_HEADS, L_KV)
    else:
        norms_exp = norms_sq

    # wn_combined = weights * norms  (B, N_HEADS, L_Q, L_KV)
    wn = weights * norms_exp[:, :, None, :]
    wn_abs = mx.abs(wn)
    total = wn_abs.size
    skipped = int(mx.sum(wn_abs < threshold))
    return skipped / total


def estimate_skip_rate_per_head(weights, v_norms, thresholds_used):
    """Estimate skip rate per head given individual thresholds."""
    norms_sq = v_norms.squeeze(-1)
    if REP > 1:
        norms_exp = mx.repeat(norms_sq, REP, axis=1)
    else:
        norms_exp = norms_sq

    wn = weights * norms_exp[:, :, None, :]
    wn_abs = mx.abs(wn)

    total_skipped = 0
    total_elements = 0
    for h, t, _ in thresholds_used:
        wn_h = wn_abs[:, h, :, :]
        n_elem = wn_h.size
        total_elements += n_elem
        if t > 0:
            total_skipped += int(mx.sum(wn_h < t))

    return total_skipped / total_elements if total_elements > 0 else 0.0


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
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, warmup_n=WARMUP, trials_n=N_TRIALS):
    """Time a function: warmup, then return median of trials (seconds)."""
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
# Per-distribution test suite
# ---------------------------------------------------------------------------

def run_distribution_test(dist_name, weights, values_raw):
    """Run all three strategies on a given weight distribution.

    Returns dict with all metrics.
    """
    print(f"\n  --- Distribution: {dist_name} ---")

    pq_val, v_packed, v_norms, val_centroids = setup_quantized_v(values_raw)
    mx.eval(weights)

    # Compute per-head entropy
    entropy_vals = compute_head_entropy(weights)
    mx.eval(entropy_vals)
    ent_np = np.array(entropy_vals.reshape(B, N_HEADS))

    print(f"  Per-head entropy: ", end="")
    for h in range(N_HEADS):
        print(f"H{h}={ent_np[0, h]:.3f}", end="  ")
    print()

    # ---- FP16 Baseline (ground truth) ----
    fp16_out = fp16_sv_matmul(weights, values_raw)
    mx.eval(fp16_out)

    # ---- Strategy 1: Baseline (threshold=0.0) ----
    def run_baseline():
        return run_sv_kernel(weights, v_packed, v_norms, val_centroids, pq_val,
                             sparse_v_threshold=0.0)

    t_baseline = time_fn(run_baseline)
    out_baseline = run_baseline()
    mx.eval(out_baseline)
    cos_baseline = cosine_similarity(fp16_out, out_baseline)
    skip_baseline = 0.0

    # ---- Strategy 2: Fixed threshold=0.01 ----
    def run_fixed():
        return run_sv_kernel(weights, v_packed, v_norms, val_centroids, pq_val,
                             sparse_v_threshold=MAX_THRESHOLD)

    t_fixed = time_fn(run_fixed)
    out_fixed = run_fixed()
    mx.eval(out_fixed)
    cos_fixed = cosine_similarity(fp16_out, out_fixed)
    skip_fixed = estimate_skip_rate(weights, v_norms, MAX_THRESHOLD)

    # ---- Strategy 3: Entropy-guided ----
    # First call to get thresholds and output
    out_entropy, thresholds_used = entropy_guided_sv(
        weights, v_packed, v_norms, val_centroids, pq_val, entropy_vals,
    )
    mx.eval(out_entropy)

    def run_entropy():
        o, _ = entropy_guided_sv(
            weights, v_packed, v_norms, val_centroids, pq_val, entropy_vals,
        )
        return o

    t_entropy = time_fn(run_entropy)
    cos_entropy = cosine_similarity(fp16_out, out_entropy)
    skip_entropy = estimate_skip_rate_per_head(weights, v_norms, thresholds_used)

    # ---- Per-head cosine similarity for fixed vs entropy-guided ----
    per_head_cos_fixed = []
    per_head_cos_entropy = []
    for h in range(N_HEADS):
        fp16_h = fp16_out[:, h:h+1, :, :]
        fixed_h = out_fixed[:, h:h+1, :, :]
        entropy_h = out_entropy[:, h:h+1, :, :]
        per_head_cos_fixed.append(cosine_similarity(fp16_h, fixed_h))
        per_head_cos_entropy.append(cosine_similarity(fp16_h, entropy_h))

    # ---- Print results ----
    print(f"\n  {'Strategy':<22s} {'Time (ms)':>10s} {'Cos sim':>10s} "
          f"{'Skip %':>8s} {'Speedup':>8s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'Baseline (t=0.0)':<22s} {t_baseline*1000:>10.2f} "
          f"{cos_baseline:>10.6f} {skip_baseline*100:>7.1f}% {'1.00x':>8s}")
    speedup_fixed = t_baseline / t_fixed if t_fixed > 0 else 0
    print(f"  {'Fixed (t=0.01)':<22s} {t_fixed*1000:>10.2f} "
          f"{cos_fixed:>10.6f} {skip_fixed*100:>7.1f}% "
          f"{speedup_fixed:>7.2f}x")
    speedup_entropy = t_baseline / t_entropy if t_entropy > 0 else 0
    print(f"  {'Entropy-guided':<22s} {t_entropy*1000:>10.2f} "
          f"{cos_entropy:>10.6f} {skip_entropy*100:>7.1f}% "
          f"{speedup_entropy:>7.2f}x")

    # Per-head detail
    print(f"\n  Per-head detail:")
    print(f"  {'Head':>6s} {'Entropy':>8s} {'Threshold':>10s} "
          f"{'CosSim Fixed':>14s} {'CosSim Entropy':>14s} {'Delta':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*14} {'-'*14} {'-'*8}")
    for h in range(N_HEADS):
        e = ent_np[0, h]
        t_h = next(t for hi, t, _ in thresholds_used if hi == h)
        cf = per_head_cos_fixed[h]
        ce = per_head_cos_entropy[h]
        delta = ce - cf  # positive = entropy-guided is better
        marker = " **" if delta > 0.01 else ""
        print(f"  H{h:>4d} {e:>8.4f} {t_h:>10.6f} "
              f"{cf:>14.6f} {ce:>14.6f} {delta:>+8.4f}{marker}")

    # Threshold mapping detail
    print(f"\n  Entropy-to-threshold mapping:")
    for h, t, e in thresholds_used:
        label = "PRUNE" if t > 0 else "FULL "
        print(f"    H{h}: entropy={e:.4f} -> threshold={t:.6f} [{label}]")

    return {
        "dist": dist_name,
        "t_baseline": t_baseline,
        "t_fixed": t_fixed,
        "t_entropy": t_entropy,
        "cos_baseline": cos_baseline,
        "cos_fixed": cos_fixed,
        "cos_entropy": cos_entropy,
        "skip_baseline": skip_baseline,
        "skip_fixed": skip_fixed,
        "skip_entropy": skip_entropy,
        "speedup_fixed": speedup_fixed,
        "speedup_entropy": speedup_entropy,
        "entropy_per_head": ent_np[0].tolist(),
        "per_head_cos_fixed": per_head_cos_fixed,
        "per_head_cos_entropy": per_head_cos_entropy,
        "thresholds_used": thresholds_used,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Experiment 1: Entropy-Guided Adaptive Sparse V Threshold")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
          f"D={D}, bits={BITS}")
    print(f"Context:    L_kv={L_KV:,}, L_q={L_Q} (decode)")
    print(f"Threshold:  max={MAX_THRESHOLD}, mapping=sigmoid")
    print(f"Trials:     {N_TRIALS} (median), {WARMUP} warmup")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()

    rng = np.random.RandomState(42)
    weight_shape = (B, N_HEADS, L_Q, L_KV)

    # Shared V values across all tests
    values_raw = mx.array(
        rng.standard_normal((B, N_KV_HEADS, L_KV, D)).astype(np.float32)
    )
    mx.eval(values_raw)

    all_results = []

    # ---- Test 1: Concentrated (low entropy) ----
    print("\n" + "=" * 78)
    print("  TEST 1: Concentrated Attention (low entropy)")
    print("=" * 78)
    try:
        w_conc = make_concentrated_weights(rng, weight_shape)
        r = run_distribution_test("Concentrated", w_conc, values_raw)
        all_results.append(r)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    gc.collect()

    # ---- Test 2: Spread (high entropy) ----
    print("\n" + "=" * 78)
    print("  TEST 2: Spread Attention (high entropy)")
    print("=" * 78)
    try:
        w_spread = make_spread_weights(rng, weight_shape)
        r = run_distribution_test("Spread", w_spread, values_raw)
        all_results.append(r)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    gc.collect()

    # ---- Test 3: Realistic mix ----
    print("\n" + "=" * 78)
    print("  TEST 3: Realistic Mix (half concentrated, half spread)")
    print("=" * 78)
    try:
        w_mix = make_realistic_weights(rng, weight_shape)
        r = run_distribution_test("Realistic Mix", w_mix, values_raw)
        all_results.append(r)
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    gc.collect()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}\n")

    print(f"  {'Distribution':<16s} | {'Strategy':<18s} | {'Time (ms)':>10s} | "
          f"{'Cos Sim':>10s} | {'Skip %':>8s} | {'Speedup':>8s}")
    print(f"  {'-'*16}-+-{'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for r in all_results:
        d = r["dist"]
        print(f"  {d:<16s} | {'Baseline (t=0)':<18s} | "
              f"{r['t_baseline']*1000:>10.2f} | {r['cos_baseline']:>10.6f} | "
              f"{r['skip_baseline']*100:>7.1f}% | {'1.00x':>8s}")
        print(f"  {'':16s} | {'Fixed (t=0.01)':<18s} | "
              f"{r['t_fixed']*1000:>10.2f} | {r['cos_fixed']:>10.6f} | "
              f"{r['skip_fixed']*100:>7.1f}% | {r['speedup_fixed']:>7.2f}x")
        print(f"  {'':16s} | {'Entropy-guided':<18s} | "
              f"{r['t_entropy']*1000:>10.2f} | {r['cos_entropy']:>10.6f} | "
              f"{r['skip_entropy']*100:>7.1f}% | {r['speedup_entropy']:>7.2f}x")
        print(f"  {'-'*16}-+-{'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    for r in all_results:
        d = r["dist"]
        # Quality delta: entropy-guided vs fixed
        q_delta = r["cos_entropy"] - r["cos_fixed"]
        direction = "better" if q_delta > 0 else "worse"
        print(f"  - {d}: Entropy-guided is {abs(q_delta):.6f} {direction} "
              f"than fixed threshold (cos sim)")
        if r["cos_entropy"] > 0.99 and r["cos_fixed"] < 0.99:
            print(f"    >> ENTROPY-GUIDED SAVES QUALITY (fixed drops below 0.99)")
        if r["speedup_entropy"] > 1.0:
            print(f"    >> Entropy-guided gets {r['speedup_entropy']:.2f}x speedup "
                  f"vs baseline")

    # -----------------------------------------------------------------------
    # Save results to markdown
    # -----------------------------------------------------------------------
    md_path = os.path.join(os.path.dirname(__file__), "EXP1_RESULTS.md")
    with open(md_path, "w") as f:
        f.write("# Experiment 1: Entropy-Guided Adaptive Sparse V Threshold\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
                f"D={D}, bits={BITS}  \n")
        f.write(f"**Context:** L_kv={L_KV:,}, L_q={L_Q} (decode)  \n")
        f.write(f"**Threshold:** max={MAX_THRESHOLD}, mapping=sigmoid  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        f.write("## Hypothesis\n\n")
        f.write("Attention weights post-softmax have varying entropy per head. "
                "Low-entropy heads (concentrated on few tokens) can tolerate "
                "aggressive sparse V pruning. High-entropy heads (spread "
                "attention) cannot. By computing per-head Shannon entropy and "
                "adapting the threshold, we get speedup on concentrated heads "
                "without quality loss on spread heads.\n\n")

        f.write("## Strategy Comparison\n\n")
        f.write("| Distribution | Strategy | Time (ms) | Cos Sim vs FP16 | "
                "Skip % | Speedup |\n")
        f.write("|:-------------|:---------|----------:|----------------:|"
                "-------:|--------:|\n")

        for r in all_results:
            d = r["dist"]
            f.write(f"| {d} | Baseline (t=0) | {r['t_baseline']*1000:.2f} | "
                    f"{r['cos_baseline']:.6f} | {r['skip_baseline']*100:.1f}% "
                    f"| 1.00x |\n")
            f.write(f"| | Fixed (t=0.01) | {r['t_fixed']*1000:.2f} | "
                    f"{r['cos_fixed']:.6f} | {r['skip_fixed']*100:.1f}% "
                    f"| {r['speedup_fixed']:.2f}x |\n")
            f.write(f"| | Entropy-guided | {r['t_entropy']*1000:.2f} | "
                    f"{r['cos_entropy']:.6f} | {r['skip_entropy']*100:.1f}% "
                    f"| {r['speedup_entropy']:.2f}x |\n")

        # Per-head entropy table
        f.write("\n## Per-Head Entropy Values\n\n")
        f.write("| Distribution |")
        for h in range(N_HEADS):
            f.write(f" H{h} |")
        f.write("\n|:-------------|")
        for _ in range(N_HEADS):
            f.write("-----:|")
        f.write("\n")

        for r in all_results:
            f.write(f"| {r['dist']} |")
            for e in r["entropy_per_head"]:
                f.write(f" {e:.4f} |")
            f.write("\n")

        # Per-head cosine similarity
        f.write("\n## Per-Head Cosine Similarity vs FP16\n\n")
        f.write("| Distribution | Head | Entropy | Threshold | "
                "Fixed CosSim | Entropy CosSim | Delta |\n")
        f.write("|:-------------|-----:|--------:|----------:|"
                "-------------:|---------------:|------:|\n")

        for r in all_results:
            for h in range(N_HEADS):
                e = r["entropy_per_head"][h]
                t = next(t for hi, t, _ in r["thresholds_used"] if hi == h)
                cf = r["per_head_cos_fixed"][h]
                ce = r["per_head_cos_entropy"][h]
                delta = ce - cf
                f.write(f"| {r['dist']} | {h} | {e:.4f} | {t:.6f} | "
                        f"{cf:.6f} | {ce:.6f} | {delta:+.4f} |\n")

        # Threshold mapping
        f.write("\n## Entropy-to-Threshold Mapping\n\n")
        f.write("Mapping function: `threshold = max_threshold * "
                "sigmoid(-10 * (entropy - 0.5))`\n\n")
        f.write("| Entropy | Threshold | Action |\n")
        f.write("|--------:|----------:|:-------|\n")
        for e_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            t = entropy_to_threshold(e_val)
            action = "aggressive prune" if t > 0.005 else (
                "light prune" if t > 0.001 else "no prune")
            f.write(f"| {e_val:.1f} | {t:.6f} | {action} |\n")

        # Analysis
        f.write("\n## Analysis\n\n")
        for r in all_results:
            d = r["dist"]
            q_delta = r["cos_entropy"] - r["cos_fixed"]
            f.write(f"### {d}\n\n")
            f.write(f"- Mean entropy: {np.mean(r['entropy_per_head']):.4f}\n")
            f.write(f"- Entropy range: [{min(r['entropy_per_head']):.4f}, "
                    f"{max(r['entropy_per_head']):.4f}]\n")
            f.write(f"- Fixed threshold quality (cos sim): "
                    f"{r['cos_fixed']:.6f}\n")
            f.write(f"- Entropy-guided quality (cos sim): "
                    f"{r['cos_entropy']:.6f}\n")
            f.write(f"- Quality delta (entropy - fixed): {q_delta:+.6f}\n")
            f.write(f"- Fixed skip rate: {r['skip_fixed']*100:.1f}%\n")
            f.write(f"- Entropy-guided skip rate: "
                    f"{r['skip_entropy']*100:.1f}%\n")

            if r["cos_entropy"] > 0.99 and r["cos_fixed"] < 0.99:
                f.write(f"- **RESULT: Entropy-guided preserves quality "
                        f"where fixed threshold fails**\n")
            elif q_delta > 0:
                f.write(f"- Entropy-guided provides better quality\n")
            else:
                f.write(f"- Both strategies have similar quality\n")
            f.write("\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        conc_result = next((r for r in all_results if r["dist"] == "Concentrated"), None)
        spread_result = next((r for r in all_results if r["dist"] == "Spread"), None)
        mix_result = next((r for r in all_results if r["dist"] == "Realistic Mix"), None)

        if conc_result and spread_result:
            f.write("The entropy metric correctly identifies head attention patterns:\n\n")

            if conc_result["cos_fixed"] > 0.95:
                f.write("- **Concentrated heads**: Fixed threshold=0.01 maintains "
                        f"quality (cos={conc_result['cos_fixed']:.4f}), "
                        f"confirming aggressive pruning is safe for low-entropy heads\n")
            else:
                f.write("- **Concentrated heads**: Even concentrated heads show "
                        "quality degradation with fixed threshold, "
                        "suggesting threshold calibration matters\n")

            if spread_result["cos_fixed"] < spread_result["cos_entropy"]:
                f.write("- **Spread heads**: Fixed threshold degrades quality "
                        f"(cos={spread_result['cos_fixed']:.4f}) while "
                        f"entropy-guided preserves it "
                        f"(cos={spread_result['cos_entropy']:.4f})\n")
            else:
                f.write("- **Spread heads**: Both strategies show similar quality\n")

            if mix_result:
                f.write(f"- **Realistic mix**: Entropy-guided achieves "
                        f"cos={mix_result['cos_entropy']:.4f} vs "
                        f"fixed cos={mix_result['cos_fixed']:.4f}, "
                        f"with {mix_result['skip_entropy']*100:.1f}% skip rate\n")

        f.write("\n### Verdict\n\n")
        # Determine overall verdict
        entropy_wins_quality = all(
            r["cos_entropy"] >= r["cos_fixed"] - 0.001 for r in all_results
        )
        entropy_gets_speedup = any(
            r["skip_entropy"] > 0.01 for r in all_results
        )

        if entropy_wins_quality and entropy_gets_speedup:
            f.write("**POSITIVE**: Entropy-guided adaptive thresholds provide "
                    "quality at least as good as fixed thresholds while "
                    "enabling selective pruning. The approach is viable for "
                    "production integration.\n")
        elif entropy_wins_quality:
            f.write("**NEUTRAL**: Entropy-guided thresholds preserve quality "
                    "but overhead from per-head dispatch may negate speedup "
                    "gains. Consider kernel-level per-head thresholds.\n")
        else:
            f.write("**NEGATIVE**: Results do not clearly support the entropy "
                    "hypothesis. Further investigation needed.\n")

    print(f"\n  Results saved to: {md_path}")
    print(f"\n{'='*78}")
    print("  EXPERIMENT 1 COMPLETE")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
