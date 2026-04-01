#!/usr/bin/env python3
"""
Experiment 5: Hub Token Protection in Sparse V for PolarQuant

Hypothesis: Some token positions are "hub tokens" -- critical across ALL heads
simultaneously (system prompt tokens, key instructions, conversation anchors).
Per-head entropy-guided thresholds (Phase 2a) might prune these hub positions on
individual heads even though they carry disproportionate global importance.

Inspired by scale-free network analysis: hub nodes have influence that exceeds
what any single local metric captures.  In attention, a hub token is a position
that receives moderate-to-high weight across MANY heads simultaneously, even if
no single head gives it peak attention.

Strategy comparison:
  a) Baseline (threshold=0.0): no pruning, full quality
  b) Fixed threshold (0.01): uniform pruning, no protection
  c) Entropy-guided (per-head adaptive): current Phase 2a, no hub protection
  d) Hub-protected: entropy-guided thresholds + hub positions always processed

Hub protection mechanism:
  After computing wn_combined (weights * norms), boost hub positions whose
  |wn| falls below the per-head threshold to just above it, preserving their
  sign but ensuring they pass the kernel's threshold check.

Additional tests:
  - Multiple hub fractions: 1%, 2%, 5%, 10%
  - Hub stability: do hub positions identified from one query remain hubs
    for a different query at the same context?

Usage:
    cd ~/workspace/polarquant-metal
    python3 benchmarks/exp5_hub_tokens.py
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
MAX_THRESHOLD = 0.01

REP = N_HEADS // N_KV_HEADS  # 4

HUB_FRACTIONS = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%
N_HUB_POSITIONS = 50   # system prompt tokens for the main test


# ---------------------------------------------------------------------------
# Hub token identification
# ---------------------------------------------------------------------------

def identify_hub_tokens(weights, top_k_fraction=0.05):
    """Find positions with high attention across multiple heads.

    Hub score = mean attention weight across all heads for each position.
    Hub tokens = top top_k_fraction positions by hub score.

    Args:
        weights: (B, n_heads, L_q, L_kv) attention weights
        top_k_fraction: fraction of positions to protect (default 5%)

    Returns:
        hub_mask: (L_kv,) boolean mask, True for hub positions
        hub_scores: (L_kv,) mean attention across heads per position
    """
    # Average attention across all heads: (B, L_q, L_kv) -> (L_kv,)
    mean_attn = weights.mean(axis=1).squeeze()  # (L_kv,)
    mx.eval(mean_attn)

    k = max(1, int(weights.shape[-1] * top_k_fraction))
    # Top-k: argpartition gives k smallest of -mean_attn = k largest of mean_attn
    mean_attn_np = np.array(mean_attn)
    top_indices = np.argpartition(-mean_attn_np, k)[:k]

    hub_mask = mx.zeros((weights.shape[-1],), dtype=mx.bool_)
    # Build mask via numpy then convert
    mask_np = np.zeros(weights.shape[-1], dtype=bool)
    mask_np[top_indices] = True
    hub_mask = mx.array(mask_np)

    return hub_mask, mean_attn


def hub_overlap(mask_a, mask_b):
    """Jaccard similarity between two boolean masks."""
    a_np = np.array(mask_a).astype(bool)
    b_np = np.array(mask_b).astype(bool)
    intersection = np.sum(a_np & b_np)
    union = np.sum(a_np | b_np)
    if union == 0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Entropy utilities (from exp1)
# ---------------------------------------------------------------------------

def compute_head_entropy(weights):
    """Compute normalized Shannon entropy per head.

    Returns:
        entropy: (n_heads,) normalized entropy in [0, 1]
    """
    eps = 1e-10
    w = mx.maximum(weights, eps)
    log_w = mx.log(w)
    h = -mx.sum(w * log_w, axis=-1).mean(axis=(0, 2))  # (n_heads,)
    h_max = math.log(weights.shape[-1])
    mx.eval(h)
    return h / h_max


def entropy_to_threshold(entropy_val, max_threshold=MAX_THRESHOLD):
    """Map normalized entropy to sparse V threshold (sigmoid)."""
    x = -10.0 * (entropy_val - 0.5)
    sig = 1.0 / (1.0 + math.exp(-x))
    return max_threshold * sig


def compute_adaptive_thresholds(weights):
    """Compute per-head adaptive thresholds. Returns (n_heads,) array + list."""
    entropy = compute_head_entropy(weights)
    mx.eval(entropy)
    ent_np = np.array(entropy)
    thresholds = []
    for h in range(N_HEADS):
        t = entropy_to_threshold(float(ent_np[h]))
        thresholds.append(t)
    return mx.array(thresholds, dtype=mx.float32), thresholds, ent_np


# ---------------------------------------------------------------------------
# Attention weight generation with hub tokens
# ---------------------------------------------------------------------------

def make_hub_attention_weights(rng, n_hub=N_HUB_POSITIONS):
    """Create attention weights where positions 0..n_hub-1 are hub tokens.

    Each head has its own concentrated pattern (low entropy), but hub token
    positions receive moderate attention from ALL heads -- simulating system
    prompt tokens that every head attends to.

    Args:
        rng: numpy RandomState
        n_hub: number of hub token positions (default 50)

    Returns:
        weights: (B, N_HEADS, L_Q, L_KV) normalized attention weights
    """
    shape = (B, N_HEADS, L_Q, L_KV)
    logits = np.full(shape, -10.0, dtype=np.float32)  # start very low

    for h in range(N_HEADS):
        # Each head concentrates on ~200 random non-hub positions
        hot_count = 200
        hot_positions = rng.choice(
            np.arange(n_hub, L_KV), size=hot_count, replace=False
        )
        # High logits for head-specific positions
        logits[0, h, 0, hot_positions] = rng.uniform(3.0, 8.0, size=hot_count)

    # Hub tokens: moderate logits across ALL heads (system prompt)
    # Not the highest per head, but consistently present everywhere
    hub_logits = rng.uniform(1.5, 3.0, size=(1, N_HEADS, 1, n_hub))
    logits[:, :, :, :n_hub] = hub_logits

    # Softmax
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights.astype(np.float32))


def make_varied_hub_weights(rng, n_hub=N_HUB_POSITIONS):
    """Like make_hub_attention_weights but with more variety in head patterns.

    Heads 0-3: concentrated (low entropy), strong head-specific + hub tokens
    Heads 4-7: more spread (higher entropy), weaker head-specific + hub tokens
    """
    shape = (B, N_HEADS, L_Q, L_KV)
    logits = np.full(shape, -10.0, dtype=np.float32)

    for h in range(N_HEADS):
        if h < 4:
            # Concentrated heads: few hot positions with high logits
            hot_count = 100
            hot_positions = rng.choice(
                np.arange(n_hub, L_KV), size=hot_count, replace=False
            )
            logits[0, h, 0, hot_positions] = rng.uniform(5.0, 10.0, size=hot_count)
        else:
            # Spread heads: more positions with moderate logits
            hot_count = 2000
            hot_positions = rng.choice(
                np.arange(n_hub, L_KV), size=hot_count, replace=False
            )
            logits[0, h, 0, hot_positions] = rng.uniform(0.5, 3.0, size=hot_count)

    # Hub tokens across all heads
    hub_logits = rng.uniform(1.0, 2.5, size=(1, N_HEADS, 1, n_hub))
    logits[:, :, :, :n_hub] = hub_logits

    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights.astype(np.float32))


# ---------------------------------------------------------------------------
# FP16 baseline
# ---------------------------------------------------------------------------

def fp16_sv_matmul(weights, values):
    """Standard FP16 attention output: weights @ V (with GQA expansion)."""
    if REP > 1:
        values_exp = mx.repeat(values, REP, axis=1)
    else:
        values_exp = values
    return weights @ values_exp


# ---------------------------------------------------------------------------
# Quantized V setup
# ---------------------------------------------------------------------------

def setup_quantized_v(values_raw):
    """Quantize and pack V values, return kernel inputs."""
    pq_val = PolarQuant(bits=BITS, dim=D, seed=43)
    v_idx, v_norms = pq_val.quantize(values_raw)
    v_packed = pack_indices(v_idx, BITS)
    val_centroids = load_codebook_f32(BITS, D)
    mx.eval(v_packed, v_norms, val_centroids)
    return pq_val, v_packed, v_norms, val_centroids


# ---------------------------------------------------------------------------
# SV kernel runners
# ---------------------------------------------------------------------------

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


def run_sv_kernel_hub_protected(weights, v_packed, v_norms, val_centroids,
                                pq_val, per_head_thresholds, hub_mask):
    """Run SV kernel with hub protection.

    Hub positions have their wn_combined values boosted above the per-head
    threshold so they always pass the kernel's threshold check.

    This manipulates the wn_combined array before dispatch:
    for hub positions where |wn| < threshold, set |wn| = threshold + epsilon,
    preserving the original sign.
    """
    # Pre-combine weights * norms (same as the kernel's precombine path)
    norms_sq = v_norms.squeeze(-1)  # (B, N_KV_HEADS, L_KV)
    if REP > 1:
        norms_exp = mx.repeat(norms_sq, REP, axis=1)  # (B, N_HEADS, L_KV)
    else:
        norms_exp = norms_sq
    wn = weights * norms_exp[:, :, None, :]  # (B, N_HEADS, L_Q, L_KV)

    # Hub protection: boost hub positions above threshold per head
    hub_mask_float = hub_mask.astype(mx.float32)  # (L_KV,)
    thresh_list = list(np.array(per_head_thresholds))

    wn_np = np.array(wn)
    for h in range(N_HEADS):
        thresh_h = thresh_list[h]
        if thresh_h <= 0:
            continue
        hub_np = np.array(hub_mask).astype(bool)
        # For hub positions where |wn| < thresh, boost to just above thresh
        wn_h = wn_np[0, h, 0, :]  # (L_KV,)
        needs_boost = hub_np & (np.abs(wn_h) < thresh_h)
        if np.any(needs_boost):
            signs = np.sign(wn_h[needs_boost])
            # If wn is exactly zero, give it a small positive value
            signs[signs == 0] = 1.0
            wn_h[needs_boost] = signs * (thresh_h + 1e-6)
            wn_np[0, h, 0, :] = wn_h

    wn_protected = mx.array(wn_np.astype(np.float32))

    # Now run the precombined kernel directly with the protected wn
    # We pass wn_protected as weights and provide identity norms (all 1.0)
    # since we already incorporated the norms into wn_protected.
    identity_norms = mx.ones_like(v_norms)

    out_rot = polarquant_sv_matmul(
        weights=wn_protected,
        v_indices=v_packed,
        v_norms=identity_norms,
        v_centroids=val_centroids,
        head_dim=D,
        bits=BITS,
        sparse_v_threshold=per_head_thresholds,
    )
    return out_rot @ pq_val.rotation


# ---------------------------------------------------------------------------
# Skip rate estimation
# ---------------------------------------------------------------------------

def estimate_skip_rate(weights, v_norms, threshold):
    """Fraction of positions skipped by sparse V (scalar threshold)."""
    if isinstance(threshold, (int, float)) and threshold == 0.0:
        return 0.0
    norms_sq = v_norms.squeeze(-1)
    if REP > 1:
        norms_exp = mx.repeat(norms_sq, REP, axis=1)
    else:
        norms_exp = norms_sq
    wn = weights * norms_exp[:, :, None, :]
    wn_abs = mx.abs(wn)

    if isinstance(threshold, mx.array):
        # Per-head: expand thresholds to (1, N_HEADS, 1, 1)
        thresh_exp = threshold.reshape(1, N_HEADS, 1, 1)
        skipped = int(mx.sum(wn_abs < thresh_exp))
    else:
        skipped = int(mx.sum(wn_abs < threshold))
    total = wn_abs.size
    return skipped / total


def estimate_skip_rate_hub_split(weights, v_norms, threshold, hub_mask):
    """Skip rate split by hub vs non-hub positions."""
    norms_sq = v_norms.squeeze(-1)
    if REP > 1:
        norms_exp = mx.repeat(norms_sq, REP, axis=1)
    else:
        norms_exp = norms_sq
    wn = weights * norms_exp[:, :, None, :]
    wn_abs = mx.abs(wn)  # (B, N_HEADS, L_Q, L_KV)

    hub_np = np.array(hub_mask).astype(bool)
    wn_abs_np = np.array(wn_abs)

    if isinstance(threshold, mx.array):
        thresh_np = np.array(threshold)  # (N_HEADS,)
        # Expand for broadcasting: (1, N_HEADS, 1, 1)
        thresh_exp = thresh_np.reshape(1, N_HEADS, 1, 1)
        below = wn_abs_np < thresh_exp
    else:
        below = wn_abs_np < threshold

    # Hub positions
    hub_below = below[:, :, :, hub_np]
    hub_total = hub_below.size
    hub_skipped = int(np.sum(hub_below)) if hub_total > 0 else 0

    # Non-hub positions
    nonhub_below = below[:, :, :, ~hub_np]
    nonhub_total = nonhub_below.size
    nonhub_skipped = int(np.sum(nonhub_below)) if nonhub_total > 0 else 0

    overall_skip = (hub_skipped + nonhub_skipped) / (hub_total + nonhub_total)
    hub_skip = hub_skipped / hub_total if hub_total > 0 else 0.0
    nonhub_skip = nonhub_skipped / nonhub_total if nonhub_total > 0 else 0.0

    return overall_skip, hub_skip, nonhub_skip


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """Global cosine similarity between two arrays."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    na = mx.sqrt(mx.sum(a_flat * a_flat))
    nb = mx.sqrt(mx.sum(b_flat * b_flat))
    mx.eval(dot, na, nb)
    return float(dot / (na * nb + 1e-10))


def cosine_similarity_at_positions(a, b, mask):
    """Cosine similarity only at masked positions (last dim)."""
    mask_np = np.array(mask).astype(bool)
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    # Extract positions along last dim
    a_sel = a_np[..., mask_np].ravel()
    b_sel = b_np[..., mask_np].ravel()
    dot = np.sum(a_sel * b_sel)
    na = np.sqrt(np.sum(a_sel * a_sel))
    nb = np.sqrt(np.sum(b_sel * b_sel))
    if na * nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


def per_head_cosine(fp16_out, test_out):
    """Cosine similarity per head."""
    result = []
    for h in range(N_HEADS):
        result.append(cosine_similarity(
            fp16_out[:, h:h+1, :, :], test_out[:, h:h+1, :, :]
        ))
    return result


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
# Main test: four strategies
# ---------------------------------------------------------------------------

def run_strategy_comparison(weights, values_raw, hub_mask, label=""):
    """Run all four strategies and collect metrics."""
    print(f"\n  --- {label} ---")

    pq_val, v_packed, v_norms, val_centroids = setup_quantized_v(values_raw)
    mx.eval(weights)

    # FP16 ground truth
    fp16_out = fp16_sv_matmul(weights, values_raw)
    mx.eval(fp16_out)

    # Entropy and adaptive thresholds
    adaptive_thresh, thresh_list, ent_np = compute_adaptive_thresholds(weights)
    print(f"  Per-head entropy:   ", end="")
    for h in range(N_HEADS):
        print(f"H{h}={ent_np[h]:.3f}", end="  ")
    print()
    print(f"  Per-head threshold: ", end="")
    for h in range(N_HEADS):
        print(f"H{h}={thresh_list[h]:.5f}", end="  ")
    print()

    hub_np = np.array(hub_mask).astype(bool)
    n_hub = int(np.sum(hub_np))
    print(f"  Hub positions: {n_hub} ({n_hub/L_KV*100:.1f}%)")

    results = {}

    # (a) Baseline: threshold=0
    def fn_baseline():
        return run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                             pq_val, sparse_v_threshold=0.0)
    t_a = time_fn(fn_baseline)
    out_a = fn_baseline()
    mx.eval(out_a)
    results["baseline"] = {
        "time_ms": t_a * 1000,
        "cos_global": cosine_similarity(fp16_out, out_a),
        "cos_per_head": per_head_cosine(fp16_out, out_a),
        "skip_overall": 0.0,
        "skip_hub": 0.0,
        "skip_nonhub": 0.0,
    }

    # (b) Fixed threshold=0.01
    def fn_fixed():
        return run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                             pq_val, sparse_v_threshold=MAX_THRESHOLD)
    t_b = time_fn(fn_fixed)
    out_b = fn_fixed()
    mx.eval(out_b)
    skip_b, skip_hub_b, skip_nonhub_b = estimate_skip_rate_hub_split(
        weights, v_norms, MAX_THRESHOLD, hub_mask
    )
    results["fixed"] = {
        "time_ms": t_b * 1000,
        "cos_global": cosine_similarity(fp16_out, out_b),
        "cos_per_head": per_head_cosine(fp16_out, out_b),
        "skip_overall": skip_b,
        "skip_hub": skip_hub_b,
        "skip_nonhub": skip_nonhub_b,
    }

    # (c) Entropy-guided (per-head adaptive, no hub protection)
    def fn_entropy():
        return run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                             pq_val, sparse_v_threshold=adaptive_thresh)
    t_c = time_fn(fn_entropy)
    out_c = fn_entropy()
    mx.eval(out_c)
    skip_c, skip_hub_c, skip_nonhub_c = estimate_skip_rate_hub_split(
        weights, v_norms, adaptive_thresh, hub_mask
    )
    results["entropy"] = {
        "time_ms": t_c * 1000,
        "cos_global": cosine_similarity(fp16_out, out_c),
        "cos_per_head": per_head_cosine(fp16_out, out_c),
        "skip_overall": skip_c,
        "skip_hub": skip_hub_c,
        "skip_nonhub": skip_nonhub_c,
    }

    # (d) Hub-protected: entropy-guided + hub positions always processed
    def fn_hub():
        return run_sv_kernel_hub_protected(
            weights, v_packed, v_norms, val_centroids,
            pq_val, adaptive_thresh, hub_mask
        )
    t_d = time_fn(fn_hub)
    out_d = fn_hub()
    mx.eval(out_d)
    # Hub-protected: hub positions have 0% skip by design
    skip_d_overall, _, skip_nonhub_d = estimate_skip_rate_hub_split(
        weights, v_norms, adaptive_thresh, hub_mask
    )
    results["hub_protected"] = {
        "time_ms": t_d * 1000,
        "cos_global": cosine_similarity(fp16_out, out_d),
        "cos_per_head": per_head_cosine(fp16_out, out_d),
        "skip_overall": skip_nonhub_d * (1 - n_hub / L_KV),  # only non-hub can skip
        "skip_hub": 0.0,  # by design
        "skip_nonhub": skip_nonhub_d,
    }

    # Quality at hub vs non-hub positions (against FP16)
    # Note: output is (B, N_HEADS, L_Q, D), hub positions affect via attention
    # We compare the full output quality since hub tokens influence all dims

    # Print results table
    print(f"\n  {'Strategy':<20s} {'Time(ms)':>9s} {'CosSim':>9s} "
          f"{'Skip%':>7s} {'HubSkip%':>9s} {'NonHubSkip%':>12s} {'Speedup':>8s}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*12} {'-'*8}")

    t_base = results["baseline"]["time_ms"]
    for name, key in [("Baseline (t=0)", "baseline"),
                      ("Fixed (t=0.01)", "fixed"),
                      ("Entropy-guided", "entropy"),
                      ("Hub-protected", "hub_protected")]:
        r = results[key]
        speedup = t_base / r["time_ms"] if r["time_ms"] > 0 else 0
        print(f"  {name:<20s} {r['time_ms']:>9.2f} {r['cos_global']:>9.6f} "
              f"{r['skip_overall']*100:>6.1f}% {r['skip_hub']*100:>8.1f}% "
              f"{r['skip_nonhub']*100:>11.1f}% {speedup:>7.2f}x")

    # Per-head detail
    print(f"\n  Per-head cosine similarity vs FP16:")
    print(f"  {'Head':>6s} {'Entropy':>8s} {'Thresh':>8s} "
          f"{'Baseline':>10s} {'Fixed':>10s} {'Entropy':>10s} {'Hub-Prot':>10s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for h in range(N_HEADS):
        print(f"  H{h:>4d} {ent_np[h]:>8.4f} {thresh_list[h]:>8.5f} "
              f"{results['baseline']['cos_per_head'][h]:>10.6f} "
              f"{results['fixed']['cos_per_head'][h]:>10.6f} "
              f"{results['entropy']['cos_per_head'][h]:>10.6f} "
              f"{results['hub_protected']['cos_per_head'][h]:>10.6f}")

    results["thresholds"] = thresh_list
    results["entropy_per_head"] = ent_np.tolist()
    results["n_hub"] = n_hub
    return results


# ---------------------------------------------------------------------------
# Hub fraction sweep
# ---------------------------------------------------------------------------

def run_hub_fraction_sweep(weights, values_raw, rng):
    """Test multiple hub fractions to find the quality/speed tradeoff."""
    print(f"\n\n{'='*78}")
    print("  HUB FRACTION SWEEP")
    print(f"{'='*78}")

    pq_val, v_packed, v_norms, val_centroids = setup_quantized_v(values_raw)
    adaptive_thresh, thresh_list, ent_np = compute_adaptive_thresholds(weights)

    fp16_out = fp16_sv_matmul(weights, values_raw)
    mx.eval(fp16_out)

    # Baseline quality (no pruning)
    out_baseline = run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                                 pq_val, sparse_v_threshold=0.0)
    mx.eval(out_baseline)
    cos_baseline = cosine_similarity(fp16_out, out_baseline)

    # Entropy-guided without hub protection
    out_entropy = run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                                pq_val, sparse_v_threshold=adaptive_thresh)
    mx.eval(out_entropy)
    cos_entropy = cosine_similarity(fp16_out, out_entropy)

    results = []
    print(f"\n  {'Fraction':>10s} {'N_hub':>7s} {'CosSim':>9s} "
          f"{'vs Entropy':>11s} {'SkipRate':>9s} {'Time(ms)':>9s}")
    print(f"  {'-'*10} {'-'*7} {'-'*9} {'-'*11} {'-'*9} {'-'*9}")

    print(f"  {'none':>10s} {'0':>7s} {cos_entropy:>9.6f} "
          f"{'---':>11s} "
          f"{estimate_skip_rate(weights, v_norms, adaptive_thresh)*100:>8.1f}% "
          f"{'---':>9s}")

    for frac in HUB_FRACTIONS:
        hub_mask, _ = identify_hub_tokens(weights, top_k_fraction=frac)
        n_hub = int(np.sum(np.array(hub_mask).astype(bool)))

        def fn():
            return run_sv_kernel_hub_protected(
                weights, v_packed, v_norms, val_centroids,
                pq_val, adaptive_thresh, hub_mask
            )

        t = time_fn(fn)
        out = fn()
        mx.eval(out)
        cos = cosine_similarity(fp16_out, out)
        delta_vs_entropy = cos - cos_entropy
        skip = estimate_skip_rate_hub_split(
            weights, v_norms, adaptive_thresh, hub_mask
        )[0]

        results.append({
            "fraction": frac,
            "n_hub": n_hub,
            "cos": cos,
            "delta_vs_entropy": delta_vs_entropy,
            "skip_rate": skip,
            "time_ms": t * 1000,
        })

        print(f"  {frac*100:>9.0f}% {n_hub:>7d} {cos:>9.6f} "
              f"{delta_vs_entropy:>+10.6f} {skip*100:>8.1f}% {t*1000:>9.2f}")

    return results, cos_baseline, cos_entropy


# ---------------------------------------------------------------------------
# Hub stability test
# ---------------------------------------------------------------------------

def run_hub_stability_test(values_raw, rng, n_queries=5):
    """Test if hub tokens are stable across different queries.

    If hub-ness is a property of the KV cache (stable), the same positions
    should be hubs for different random queries.  If hub-ness depends on the
    query (unstable), hub identification from one query won't help another.
    """
    print(f"\n\n{'='*78}")
    print("  HUB STABILITY TEST")
    print(f"{'='*78}")
    print(f"  Testing {n_queries} random queries at same 16K context")
    print(f"  Question: Are hub positions a stable property of the KV cache?")

    # Generate multiple attention weight patterns (simulating different queries)
    # All share the same KV cache but have different query vectors
    hub_masks = []
    hub_scores_list = []

    for q in range(n_queries):
        # Each query produces different attention patterns
        # but hub tokens (0:50) should still attract attention
        w = make_hub_attention_weights(rng, n_hub=N_HUB_POSITIONS)
        mx.eval(w)
        mask, scores = identify_hub_tokens(w, top_k_fraction=0.05)
        hub_masks.append(mask)
        hub_scores_list.append(scores)

    # Pairwise Jaccard similarity of hub masks
    print(f"\n  Pairwise hub overlap (Jaccard similarity, 5% fraction):")
    print(f"  {'':>8s}", end="")
    for j in range(n_queries):
        print(f"  Q{j:>3d}", end="")
    print()

    overlaps = np.zeros((n_queries, n_queries))
    for i in range(n_queries):
        print(f"  Q{i:>3d}    ", end="")
        for j in range(n_queries):
            ov = hub_overlap(hub_masks[i], hub_masks[j])
            overlaps[i, j] = ov
            print(f"{ov:>6.3f}", end="")
        print()

    # Are the KNOWN hub positions (0:50) consistently identified?
    true_hubs = np.zeros(L_KV, dtype=bool)
    true_hubs[:N_HUB_POSITIONS] = True

    print(f"\n  Recovery of true hub positions (0:{N_HUB_POSITIONS}):")
    print(f"  {'Query':>7s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
    print(f"  {'-'*7} {'-'*10} {'-'*8} {'-'*8}")

    for q in range(n_queries):
        pred = np.array(hub_masks[q]).astype(bool)
        tp = np.sum(true_hubs & pred)
        fp = np.sum(~true_hubs & pred)
        fn = np.sum(true_hubs & ~pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  Q{q:>5d} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")

    # Mean off-diagonal overlap
    mask_offdiag = ~np.eye(n_queries, dtype=bool)
    mean_overlap = overlaps[mask_offdiag].mean()
    print(f"\n  Mean off-diagonal Jaccard overlap: {mean_overlap:.4f}")
    if mean_overlap > 0.5:
        print(f"  --> Hub positions are STABLE across queries (overlap > 0.5)")
    elif mean_overlap > 0.2:
        print(f"  --> Hub positions are MODERATELY stable (0.2 < overlap < 0.5)")
    else:
        print(f"  --> Hub positions are UNSTABLE across queries (overlap < 0.2)")

    return {
        "overlaps": overlaps.tolist(),
        "mean_overlap": mean_overlap,
        "n_queries": n_queries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Experiment 5: Hub Token Protection in Sparse V")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
          f"D={D}, bits={BITS}")
    print(f"Context:    L_kv={L_KV:,}, L_q={L_Q} (decode)")
    print(f"Hub:        {N_HUB_POSITIONS} system prompt positions, "
          f"fractions={HUB_FRACTIONS}")
    print(f"Threshold:  max={MAX_THRESHOLD}")
    print(f"Trials:     {N_TRIALS} (median), {WARMUP} warmup")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()

    rng = np.random.RandomState(42)

    # Shared V values
    values_raw = mx.array(
        rng.standard_normal((B, N_KV_HEADS, L_KV, D)).astype(np.float32)
    )
    mx.eval(values_raw)

    all_results = {}

    # -----------------------------------------------------------------------
    # Test 1: Hub attention pattern (all heads have hub tokens)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  TEST 1: Hub Attention Pattern (50 system prompt hub tokens)")
    print("=" * 78)
    try:
        w1 = make_hub_attention_weights(rng, n_hub=N_HUB_POSITIONS)
        hub_mask_1, hub_scores_1 = identify_hub_tokens(w1, top_k_fraction=0.05)
        r1 = run_strategy_comparison(w1, values_raw, hub_mask_1,
                                     "Hub pattern (50 sys prompt tokens)")
        all_results["hub_pattern"] = r1
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    gc.collect()

    # -----------------------------------------------------------------------
    # Test 2: Varied entropy pattern (mixed concentrated + spread heads)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  TEST 2: Varied Entropy + Hub Tokens")
    print("=" * 78)
    try:
        w2 = make_varied_hub_weights(rng, n_hub=N_HUB_POSITIONS)
        hub_mask_2, hub_scores_2 = identify_hub_tokens(w2, top_k_fraction=0.05)
        r2 = run_strategy_comparison(w2, values_raw, hub_mask_2,
                                     "Varied entropy + hub tokens")
        all_results["varied_entropy"] = r2
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    gc.collect()

    # -----------------------------------------------------------------------
    # Test 3: Hub fraction sweep
    # -----------------------------------------------------------------------
    try:
        w_sweep = make_hub_attention_weights(rng, n_hub=N_HUB_POSITIONS)
        sweep_results, cos_base, cos_ent = run_hub_fraction_sweep(
            w_sweep, values_raw, rng
        )
        all_results["fraction_sweep"] = {
            "results": sweep_results,
            "cos_baseline": cos_base,
            "cos_entropy": cos_ent,
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    gc.collect()

    # -----------------------------------------------------------------------
    # Test 4: Hub stability across queries
    # -----------------------------------------------------------------------
    try:
        stability = run_hub_stability_test(values_raw, rng, n_queries=5)
        all_results["stability"] = stability
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    gc.collect()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}")

    for test_name in ["hub_pattern", "varied_entropy"]:
        if test_name not in all_results:
            continue
        r = all_results[test_name]
        print(f"\n  {test_name}:")
        for strat in ["baseline", "fixed", "entropy", "hub_protected"]:
            s = r[strat]
            print(f"    {strat:<20s}  cos={s['cos_global']:.6f}  "
                  f"skip={s['skip_overall']*100:.1f}%  "
                  f"hub_skip={s['skip_hub']*100:.1f}%  "
                  f"time={s['time_ms']:.1f}ms")

        # Key comparison
        e = r["entropy"]
        hp = r["hub_protected"]
        delta = hp["cos_global"] - e["cos_global"]
        print(f"    --> Hub protection vs entropy-guided: "
              f"cos delta={delta:+.6f}, "
              f"hub skip {e['skip_hub']*100:.1f}% -> {hp['skip_hub']*100:.1f}%")

    if "stability" in all_results:
        s = all_results["stability"]
        print(f"\n  Hub stability: mean Jaccard overlap = "
              f"{s['mean_overlap']:.4f} across {s['n_queries']} queries")

    # -----------------------------------------------------------------------
    # Save results to markdown
    # -----------------------------------------------------------------------
    md_path = os.path.join(os.path.dirname(__file__), "EXP5_RESULTS.md")
    _save_results_md(md_path, all_results)
    print(f"\n  Results saved to: {md_path}")
    print(f"\n{'='*78}")
    print("  EXPERIMENT 5 COMPLETE")
    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _save_results_md(path, all_results):
    with open(path, "w") as f:
        f.write("# Experiment 5: Hub Token Protection in Sparse V\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
                f"D={D}, bits={BITS}  \n")
        f.write(f"**Context:** L_kv={L_KV:,}, L_q={L_Q} (decode)  \n")
        f.write(f"**Hub positions:** {N_HUB_POSITIONS} (system prompt tokens)  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        f.write("## Hypothesis\n\n")
        f.write("Some token positions are \"hub tokens\" -- critical across ALL "
                "heads simultaneously (system prompt tokens, key instructions, "
                "conversation anchors). Per-head entropy-guided thresholds "
                "(Phase 2a) might prune these hub positions on individual heads "
                "even though they carry disproportionate global importance. "
                "Hub protection ensures these positions always pass the "
                "sparse V threshold check.\n\n")

        # Strategy comparison tables
        f.write("## Strategy Comparison\n\n")
        for test_name, label in [("hub_pattern", "Hub Attention Pattern"),
                                  ("varied_entropy", "Varied Entropy + Hubs")]:
            if test_name not in all_results:
                continue
            r = all_results[test_name]

            f.write(f"### {label}\n\n")

            # Entropy per head
            f.write("Per-head entropy: ")
            for h in range(N_HEADS):
                f.write(f"H{h}={r['entropy_per_head'][h]:.3f} ")
            f.write("\n\n")

            f.write("| Strategy | Time (ms) | Cos Sim | Skip % | "
                    "Hub Skip % | Non-Hub Skip % | Speedup |\n")
            f.write("|:---------|----------:|--------:|-------:|"
                    "----------:|---------------:|--------:|\n")

            t_base = r["baseline"]["time_ms"]
            for name, key in [("Baseline (t=0)", "baseline"),
                              ("Fixed (t=0.01)", "fixed"),
                              ("Entropy-guided", "entropy"),
                              ("Hub-protected", "hub_protected")]:
                s = r[key]
                speedup = t_base / s["time_ms"] if s["time_ms"] > 0 else 0
                f.write(f"| {name} | {s['time_ms']:.2f} | "
                        f"{s['cos_global']:.6f} | {s['skip_overall']*100:.1f}% | "
                        f"{s['skip_hub']*100:.1f}% | "
                        f"{s['skip_nonhub']*100:.1f}% | "
                        f"{speedup:.2f}x |\n")

            # Per-head cosine
            f.write(f"\n**Per-head cosine similarity vs FP16:**\n\n")
            f.write("| Head | Entropy | Threshold | Baseline | Fixed | "
                    "Entropy | Hub-Protected |\n")
            f.write("|-----:|--------:|----------:|---------:|------:|"
                    "--------:|--------------:|\n")
            for h in range(N_HEADS):
                f.write(f"| H{h} | {r['entropy_per_head'][h]:.4f} | "
                        f"{r['thresholds'][h]:.5f} | "
                        f"{r['baseline']['cos_per_head'][h]:.6f} | "
                        f"{r['fixed']['cos_per_head'][h]:.6f} | "
                        f"{r['entropy']['cos_per_head'][h]:.6f} | "
                        f"{r['hub_protected']['cos_per_head'][h]:.6f} |\n")
            f.write("\n")

        # Fraction sweep
        if "fraction_sweep" in all_results:
            f.write("## Hub Fraction Sweep\n\n")
            f.write("How much of the KV cache should be protected as hub tokens?\n\n")
            sweep = all_results["fraction_sweep"]
            f.write(f"Baseline (no pruning) cos sim: {sweep['cos_baseline']:.6f}  \n")
            f.write(f"Entropy-guided (no hub prot) cos sim: "
                    f"{sweep['cos_entropy']:.6f}  \n\n")
            f.write("| Protected % | N Positions | Cos Sim | "
                    "Delta vs Entropy | Skip Rate | Time (ms) |\n")
            f.write("|-----------:|:-----------:|--------:|"
                    "----------------:|----------:|----------:|\n")
            for sr in sweep["results"]:
                f.write(f"| {sr['fraction']*100:.0f}% | {sr['n_hub']} | "
                        f"{sr['cos']:.6f} | {sr['delta_vs_entropy']:+.6f} | "
                        f"{sr['skip_rate']*100:.1f}% | {sr['time_ms']:.2f} |\n")
            f.write("\n")

        # Stability
        if "stability" in all_results:
            f.write("## Hub Stability Across Queries\n\n")
            stab = all_results["stability"]
            f.write(f"Mean pairwise Jaccard overlap of hub masks across "
                    f"{stab['n_queries']} queries: "
                    f"**{stab['mean_overlap']:.4f}**\n\n")
            if stab["mean_overlap"] > 0.5:
                f.write("Hub positions are **stable** across different queries "
                        "at the same context. This means hub identification can "
                        "be computed once (e.g., after prefill) and reused during "
                        "decode without re-identification per token.\n\n")
            elif stab["mean_overlap"] > 0.2:
                f.write("Hub positions are **moderately stable**. Hub identification "
                        "should be refreshed periodically but not every token.\n\n")
            else:
                f.write("Hub positions are **unstable** across queries. "
                        "Hub identification would need per-query computation, "
                        "making it less practical for production.\n\n")

        # Analysis & Conclusion
        f.write("## Analysis\n\n")

        if "hub_pattern" in all_results and "varied_entropy" in all_results:
            for test_name, label in [("hub_pattern", "Hub Attention Pattern"),
                                      ("varied_entropy", "Varied Entropy")]:
                r = all_results[test_name]
                e = r["entropy"]
                hp = r["hub_protected"]
                delta = hp["cos_global"] - e["cos_global"]

                f.write(f"### {label}\n\n")
                f.write(f"- Entropy-guided cos sim: {e['cos_global']:.6f}, "
                        f"hub skip: {e['skip_hub']*100:.1f}%\n")
                f.write(f"- Hub-protected cos sim: {hp['cos_global']:.6f}, "
                        f"hub skip: {hp['skip_hub']*100:.1f}% (by design)\n")
                f.write(f"- Quality delta: {delta:+.6f}\n")

                if delta > 0.001:
                    f.write(f"- **Hub protection measurably improves quality** "
                            f"by preventing pruning of globally-important positions\n")
                elif delta > 0:
                    f.write(f"- Hub protection provides marginal improvement\n")
                else:
                    f.write(f"- Hub protection shows no quality benefit in this pattern\n")
                f.write("\n")

        f.write("## Conclusion\n\n")

        # Determine verdict
        improvements = []
        for test_name in ["hub_pattern", "varied_entropy"]:
            if test_name not in all_results:
                continue
            r = all_results[test_name]
            delta = (r["hub_protected"]["cos_global"]
                     - r["entropy"]["cos_global"])
            hub_skip_reduced = (r["entropy"]["skip_hub"]
                                - r["hub_protected"]["skip_hub"])
            improvements.append((delta, hub_skip_reduced))

        has_quality_gain = any(d > 0.0005 for d, _ in improvements)
        has_hub_skip_reduction = any(h > 0.01 for _, h in improvements)
        stable_hubs = ("stability" in all_results
                       and all_results["stability"]["mean_overlap"] > 0.3)

        if has_quality_gain and stable_hubs:
            f.write("### Verdict: POSITIVE\n\n")
            f.write("Hub token protection improves quality over plain "
                    "entropy-guided thresholds, and hub positions are stable "
                    "across queries. The approach is viable for production:\n\n")
            f.write("1. Identify hub tokens once after prefill "
                    "(mean attention across heads, top-k selection)\n")
            f.write("2. During decode, boost hub positions' wn_combined above "
                    "threshold before kernel dispatch\n")
            f.write("3. Minimal overhead: only a mask broadcast + element-wise "
                    "max, no extra kernel call\n\n")

            if "fraction_sweep" in all_results:
                best = max(all_results["fraction_sweep"]["results"],
                           key=lambda x: x["delta_vs_entropy"])
                f.write(f"**Recommended hub fraction:** {best['fraction']*100:.0f}% "
                        f"({best['n_hub']} positions) -- best quality/skip tradeoff\n")
        elif has_hub_skip_reduction and not has_quality_gain:
            f.write("### Verdict: NEUTRAL\n\n")
            f.write("Hub tokens ARE being pruned by entropy-guided thresholds, "
                    "but protecting them doesn't measurably improve output quality "
                    "at this scale. The hub positions' contributions may be small "
                    "enough that pruning them is tolerable. Consider re-testing at "
                    "longer contexts (32K+) where hub token influence accumulates.\n")
        else:
            f.write("### Verdict: NEEDS INVESTIGATION\n\n")
            f.write("Results don't clearly support or refute the hypothesis. "
                    "The synthetic attention patterns may not capture real-world "
                    "hub token behavior. Next steps:\n\n")
            f.write("1. Test with actual model attention weights from Qwen3.5\n")
            f.write("2. Focus on system prompt tokens in real conversations\n")
            f.write("3. Measure perplexity impact, not just cosine similarity\n")

        f.write("\n### Next Steps\n\n")
        f.write("- Integrate hub identification into `TurboQuantKVCache.fused_sdpa()`\n")
        f.write("- Profile overhead of hub mask computation during prefill\n")
        f.write("- Test with real model weights from Qwen3.5-35B at 32K+ context\n")
        f.write("- Explore dynamic hub re-identification during long conversations\n")


if __name__ == "__main__":
    main()
