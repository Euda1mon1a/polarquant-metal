#!/usr/bin/env python3
"""
Experiment 9: Erlang Queuing Model for Sparse V Workload Prediction

Hypothesis: Attention weight distributions have predictable structure.
From the top-k attention weights, we can estimate the total number of
above-threshold positions using Erlang-style utilization analysis,
without scanning all L_kv positions.

The SV kernel (kernels.py:296-310) iterates over ALL L_kv positions and
checks `if (|wn_val| > threshold)` before doing codebook lookup + multiply.
With entropy-guided adaptive thresholds, concentrated heads skip ~99% of
positions. But the kernel still iterates ALL positions to CHECK the threshold.

Inspired by AAPM's `queuing/erlang_c.py`: queuing theory predicts wait times
and queue lengths from arrival rates and service rates. Applied to sparse V:
if we can predict HOW MANY positions will exceed the threshold (the "arrival
rate" of non-zero work), we can:
  1. Pre-compute the expected workload per head
  2. Use this to decide kernel dispatch strategy (sparse vs dense)
  3. Skip the kernel entirely if predicted workload is ~0

Key questions:
  - Can we predict SV kernel workload within 20% error from a cheap sample?
  - Is the prediction cheaper than just running the threshold check?
  - Does this enable better dispatch decisions (skip kernel when work ~ 0)?

Usage:
    cd ~/workspace/polarquant-metal
    python3 benchmarks/exp9_erlang_workload.py
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 8          # query heads
N_KV_HEADS = 2       # kv heads (GQA 4:1)
D = 128              # head_dim
L_KV = 16384         # 16K context
L_Q = 1              # decode mode
BITS = 3

THRESHOLDS = [0.0001, 0.0005, 0.001, 0.005, 0.01]
TOP_K_VALUES = [50, 100, 200, 500]
N_TIMING_TRIALS = 50
WARMUP = 5

# Dispatch policy thresholds (fraction of L_kv)
SKIP_THRESHOLD = 0.01     # <1% active -> skip kernel entirely
SPARSE_THRESHOLD = 0.10   # <10% active -> sparse dispatch (threshold enabled)
# >50% active -> dense dispatch (threshold=0, no checking overhead)
DENSE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Attention weight generators
# ---------------------------------------------------------------------------

def make_concentrated_weights(rng, shape):
    """Low entropy: ~50 positions with high weight, rest near-zero.

    Uses softmax(randn * 8) to create sharp peaks.
    """
    B, n_heads, L_q, L_kv = shape
    logits = rng.standard_normal(shape).astype(np.float32) * 8.0
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_moderate_weights(rng, shape):
    """Moderate entropy: ~500 positions with meaningful weight.

    Uses softmax(randn * 3) for wider but still peaked distribution.
    """
    B, n_heads, L_q, L_kv = shape
    logits = rng.standard_normal(shape).astype(np.float32) * 3.0
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_spread_weights(rng, shape):
    """High entropy: nearly uniform attention.

    Uses softmax(randn * 0.1) for very flat distribution.
    """
    B, n_heads, L_q, L_kv = shape
    logits = rng.standard_normal(shape).astype(np.float32) * 0.1
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return mx.array(weights)


def make_powerlaw_weights(rng, shape):
    """Power-law (Zipf) distribution: realistic for transformers.

    Attention ~ 1/rank^alpha, then softmaxed.
    """
    B, n_heads, L_q, L_kv = shape
    weights_np = np.zeros(shape, dtype=np.float32)
    for b in range(B):
        for h in range(n_heads):
            # Zipf-like: sample from Zipf, scatter to random positions
            alpha = 1.2
            ranks = np.arange(1, L_kv + 1, dtype=np.float32)
            raw = 1.0 / (ranks ** alpha)
            # Shuffle to randomize which positions get high weight
            perm = rng.permutation(L_kv)
            raw_shuffled = raw[perm]
            # Normalize to probability
            raw_shuffled /= raw_shuffled.sum()
            weights_np[b, h, 0, :] = raw_shuffled
    return mx.array(weights_np)


def make_bimodal_weights(rng, shape):
    """Bimodal: two clusters of attended positions.

    Two Gaussian peaks at different locations in the sequence.
    """
    B, n_heads, L_q, L_kv = shape
    weights_np = np.zeros(shape, dtype=np.float32)
    positions = np.arange(L_kv, dtype=np.float32)
    for b in range(B):
        for h in range(n_heads):
            # Two peaks at 1/3 and 2/3 of context
            center1 = L_kv // 3 + rng.randint(-100, 100)
            center2 = 2 * L_kv // 3 + rng.randint(-100, 100)
            sigma = 30.0
            peak1 = np.exp(-0.5 * ((positions - center1) / sigma) ** 2)
            peak2 = np.exp(-0.5 * ((positions - center2) / sigma) ** 2)
            combined = peak1 + peak2
            combined /= combined.sum()
            weights_np[b, h, 0, :] = combined
    return mx.array(weights_np)


# ---------------------------------------------------------------------------
# Actual workload measurement
# ---------------------------------------------------------------------------

def count_above_threshold(weights, threshold):
    """Count positions above threshold per head.

    Args:
        weights: (B, n_heads, L_q, L_kv) attention weights
        threshold: scalar threshold

    Returns:
        counts: (B, n_heads, L_q) integer counts
    """
    above = mx.abs(weights) > threshold
    counts = mx.sum(above.astype(mx.int32), axis=-1)
    mx.eval(counts)
    return counts


# ---------------------------------------------------------------------------
# Workload prediction: Top-k exponential tail model
# ---------------------------------------------------------------------------

def predict_workload_topk(weights, threshold, k=100):
    """Predict number of above-threshold positions from top-k sample.

    Sample the top-k weights. Fit an exponential tail model.
    Extrapolate to predict total count above threshold.

    Model: The top-k weights define a tail distribution. Between rank k
    and rank L_kv, we model the weight as decaying exponentially:
        w(rank) ~ w_k * exp(-lambda * (rank - k))
    The count above threshold is the rank at which w(rank) = threshold.

    Args:
        weights: (B, n_heads, L_q, L_kv) attention weights
        threshold: scalar threshold
        k: number of top weights to sample

    Returns:
        predicted_counts: (B, n_heads, L_q) float predictions
    """
    B_dim, n_heads, L_q, L_kv = weights.shape
    predictions = np.zeros((B_dim, n_heads, L_q), dtype=np.float32)

    for b in range(B_dim):
        for h in range(n_heads):
            for q in range(L_q):
                w_slice = weights[b, h, q, :]  # (L_kv,)
                w_abs = mx.abs(w_slice)

                # Get top-k weights (partial sort is O(L_kv))
                topk = mx.topk(w_abs, k)
                mx.eval(topk)
                topk_np = np.array(topk)
                topk_sorted = np.sort(topk_np)[::-1]  # descending

                w_1 = float(topk_sorted[0])    # largest weight
                w_k = float(topk_sorted[-1])    # k-th largest

                if w_1 <= 0 or w_k <= 0:
                    predictions[b, h, q] = 0.0
                    continue

                if threshold >= w_1:
                    # Threshold above all weights: nothing active
                    predictions[b, h, q] = 0.0
                elif threshold <= w_k:
                    # Threshold below w_k: all top-k are active plus extrapolated tail

                    # Fit decay rate from the top-k spread
                    # The top-k weights span ranks 1..k with values w_1..w_k
                    # If exponential: w_k = w_1 * exp(-lambda * (k-1))
                    # lambda = ln(w_1/w_k) / (k-1)
                    lambda_decay = math.log(w_1 / w_k) / max(k - 1, 1)

                    if lambda_decay <= 0:
                        # No decay (flat distribution) — all positions may be active
                        predictions[b, h, q] = L_kv
                        continue

                    # Predict rank where weight drops below threshold:
                    # w(rank) = w_1 * exp(-lambda * rank) = threshold
                    # rank = ln(w_1/threshold) / lambda
                    predicted_rank = math.log(w_1 / threshold) / lambda_decay
                    predictions[b, h, q] = max(0, min(L_kv, predicted_rank))
                else:
                    # Threshold between w_k and w_1: interpolate within top-k

                    # Count how many in top-k are above threshold
                    above_in_topk = int(np.sum(topk_sorted > threshold))
                    predictions[b, h, q] = float(above_in_topk)

    return predictions


# ---------------------------------------------------------------------------
# Workload prediction: Erlang utilization model
# ---------------------------------------------------------------------------

def predict_workload_utilization(weights, threshold):
    """Predict using Erlang utilization concept.

    Utilization rho = (mean weight) / threshold.
    For exponential distribution: P(w > t) = exp(-t/mean_w)
    Expected active = L_kv * P(w > threshold)

    This maps directly to the Erlang C queuing insight:
    - arrival_rate lambda = mean attention weight (avg "work" per position)
    - service_rate mu = threshold (minimum weight to get "served" by the kernel)
    - utilization rho = lambda / mu
    - When rho << 1: most positions are idle (below threshold)
    - When rho -> 1: queue builds (many above threshold)

    Args:
        weights: (B, n_heads, L_q, L_kv) attention weights
        threshold: scalar threshold

    Returns:
        predicted_counts: (B, n_heads, L_q) float predictions
    """
    B_dim, n_heads, L_q, L_kv = weights.shape
    predictions = np.zeros((B_dim, n_heads, L_q), dtype=np.float32)

    for b in range(B_dim):
        for h in range(n_heads):
            for q in range(L_q):
                w_slice = weights[b, h, q, :]
                w_abs = mx.abs(w_slice)
                mean_w = float(mx.mean(w_abs).item())

                if mean_w <= 0 or threshold <= 0:
                    predictions[b, h, q] = 0
                    continue

                # Erlang utilization: rho = mean / threshold
                # For exponential: P(w > t) = exp(-t/mean)
                p_active = math.exp(-threshold / mean_w)
                predictions[b, h, q] = L_kv * p_active

    return predictions


# ---------------------------------------------------------------------------
# Workload prediction: Hybrid (top-k informed Erlang)
# ---------------------------------------------------------------------------

def predict_workload_hybrid(weights, threshold, k=100):
    """Hybrid: use top-k to calibrate the Erlang tail model.

    1. Get top-k to estimate the tail decay rate lambda
    2. Use the exponential CDF: P(w > t) = exp(-lambda * (t - w_k))
       for t > w_k, else count directly from top-k
    3. Scale by L_kv to get predicted count

    This combines the accuracy of sampling with the theoretical backing
    of the queuing model.

    Args:
        weights: (B, n_heads, L_q, L_kv) attention weights
        threshold: scalar threshold
        k: number of top weights to sample

    Returns:
        predicted_counts: (B, n_heads, L_q) float predictions
    """
    B_dim, n_heads, L_q, L_kv = weights.shape
    predictions = np.zeros((B_dim, n_heads, L_q), dtype=np.float32)

    for b in range(B_dim):
        for h in range(n_heads):
            for q in range(L_q):
                w_slice = weights[b, h, q, :]
                w_abs = mx.abs(w_slice)

                topk = mx.topk(w_abs, k)
                mx.eval(topk)
                topk_np = np.array(topk)
                topk_sorted = np.sort(topk_np)[::-1]

                w_1 = float(topk_sorted[0])
                w_k = float(topk_sorted[-1])

                if w_1 <= 0:
                    predictions[b, h, q] = 0.0
                    continue

                if threshold >= w_1:
                    predictions[b, h, q] = 0.0
                    continue

                if threshold <= w_k:
                    # Count above in top-k is k; estimate tail contribution
                    # The tail below w_k follows an Erlang-like queue drain rate
                    # Mean weight in tail ~ w_k / 2 (conservative)
                    # Number in tail above threshold:
                    #   (L_kv - k) * P(w_tail > threshold)
                    # where P(w_tail > threshold) from exponential tail
                    if w_k > 0 and threshold < w_k:
                        # Fit tail decay from the k-th to 2k-th weight
                        # (approximate: use w_k/w_1 ratio)
                        lambda_tail = math.log(w_1 / w_k) / max(k - 1, 1)
                        if lambda_tail > 0:
                            # From rank k, weight continues to decay
                            # w(rank) ~ w_k * exp(-lambda_tail * (rank - k))
                            # Solve for rank where w(rank) = threshold
                            extra_rank = math.log(w_k / threshold) / lambda_tail
                            predictions[b, h, q] = min(L_kv, k + extra_rank)
                        else:
                            predictions[b, h, q] = L_kv
                    else:
                        predictions[b, h, q] = k
                else:
                    # Threshold between w_k and w_1: count within top-k
                    above_in_topk = int(np.sum(topk_sorted > threshold))
                    predictions[b, h, q] = float(above_in_topk)

    return predictions


# ---------------------------------------------------------------------------
# Dispatch decision logic
# ---------------------------------------------------------------------------

def decide_dispatch(predicted_active, L_kv):
    """Decide kernel dispatch strategy from predicted active count.

    Returns:
        "skip"   if predicted_active < 1% of L_kv
        "sparse" if predicted_active < 10% of L_kv
        "dense"  if predicted_active > 50% of L_kv
        "sparse" otherwise (10-50% = default sparse)
    """
    frac = predicted_active / L_kv
    if frac < SKIP_THRESHOLD:
        return "skip"
    elif frac < SPARSE_THRESHOLD:
        return "sparse"
    elif frac > DENSE_THRESHOLD:
        return "dense"
    else:
        return "sparse"


def oracle_dispatch(actual_active, L_kv):
    """Oracle dispatch: what we WOULD decide with perfect knowledge."""
    return decide_dispatch(actual_active, L_kv)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_prediction(fn, weights, threshold, warmup=WARMUP, trials=N_TIMING_TRIALS):
    """Time a prediction function in microseconds."""
    # Warmup
    for _ in range(warmup):
        _ = fn(weights, threshold)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = fn(weights, threshold)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    return np.median(times), np.mean(times)


def time_full_scan(weights, threshold, warmup=WARMUP, trials=N_TIMING_TRIALS):
    """Time the full scan approach (what the kernel currently does)."""
    for _ in range(warmup):
        _ = count_above_threshold(weights, threshold)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = count_above_threshold(weights, threshold)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return np.median(times), np.mean(times)


# ---------------------------------------------------------------------------
# Shannon entropy for characterization
# ---------------------------------------------------------------------------

def compute_entropy(weights):
    """Compute normalized Shannon entropy per head."""
    eps = 1e-10
    w = mx.maximum(mx.abs(weights), eps)
    # Normalize to probability distribution per head
    w_sum = mx.sum(w, axis=-1, keepdims=True)
    w_norm = w / w_sum
    log_w = mx.log(w_norm)
    h = -mx.sum(w_norm * log_w, axis=-1)
    h_max = math.log(weights.shape[-1])
    normalized = h / h_max
    mx.eval(normalized)
    return normalized


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Experiment 9: Erlang Queuing Model for Sparse V Workload Prediction")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA 4:1), "
          f"D={D}, bits={BITS}")
    print(f"Context:    L_kv={L_KV:,}, L_q={L_Q} (decode)")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Top-k:      {TOP_K_VALUES}")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()

    rng = np.random.RandomState(42)
    weight_shape = (B, N_HEADS, L_Q, L_KV)

    # ===================================================================
    # Generate attention distributions
    # ===================================================================
    distributions = {
        "Concentrated": make_concentrated_weights(rng, weight_shape),
        "Moderate":     make_moderate_weights(rng, weight_shape),
        "Spread":       make_spread_weights(rng, weight_shape),
        "Power-law":    make_powerlaw_weights(rng, weight_shape),
        "Bimodal":      make_bimodal_weights(rng, weight_shape),
    }

    for name, w in distributions.items():
        mx.eval(w)

    # Characterize distributions
    print("=" * 78)
    print("  DISTRIBUTION CHARACTERIZATION")
    print("=" * 78)
    print(f"\n  {'Distribution':<15s} {'Mean entropy':>13s} {'Mean weight':>12s} "
          f"{'Max weight':>12s} {'Min weight':>12s}")
    print(f"  {'-'*15} {'-'*13} {'-'*12} {'-'*12} {'-'*12}")

    dist_entropy = {}
    for name, w in distributions.items():
        ent = compute_entropy(w)
        ent_np = np.array(ent.reshape(-1))
        mean_ent = float(np.mean(ent_np))
        dist_entropy[name] = mean_ent
        mean_w = float(mx.mean(mx.abs(w)).item())
        max_w = float(mx.max(mx.abs(w)).item())
        min_w = float(mx.min(mx.abs(w)).item())
        print(f"  {name:<15s} {mean_ent:>13.4f} {mean_w:>12.6f} "
              f"{max_w:>12.6f} {min_w:>12.8f}")

    # ===================================================================
    # PHASE 1: Prediction accuracy across distributions and thresholds
    # ===================================================================
    print("\n\n" + "=" * 78)
    print("  PHASE 1: PREDICTION ACCURACY")
    print("=" * 78)

    # Storage for all results
    all_results = []

    for dist_name, weights in distributions.items():
        print(f"\n  --- {dist_name} (entropy={dist_entropy[dist_name]:.4f}) ---")
        print(f"  {'Threshold':>10s} {'Actual':>8s} {'TopK-100':>10s} "
              f"{'Err%':>7s} {'Erlang':>10s} {'Err%':>7s} "
              f"{'Hybrid':>10s} {'Err%':>7s}")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*7} {'-'*10} "
              f"{'-'*7} {'-'*10} {'-'*7}")

        for threshold in THRESHOLDS:
            actual_counts = count_above_threshold(weights, threshold)
            actual_mean = float(mx.mean(actual_counts.astype(mx.float32)).item())

            pred_topk = predict_workload_topk(weights, threshold, k=100)
            pred_topk_mean = float(np.mean(pred_topk))

            pred_erlang = predict_workload_utilization(weights, threshold)
            pred_erlang_mean = float(np.mean(pred_erlang))

            pred_hybrid = predict_workload_hybrid(weights, threshold, k=100)
            pred_hybrid_mean = float(np.mean(pred_hybrid))

            err_topk = abs(pred_topk_mean - actual_mean) / max(actual_mean, 1) * 100
            err_erlang = abs(pred_erlang_mean - actual_mean) / max(actual_mean, 1) * 100
            err_hybrid = abs(pred_hybrid_mean - actual_mean) / max(actual_mean, 1) * 100

            print(f"  {threshold:>10.4f} {actual_mean:>8.0f} "
                  f"{pred_topk_mean:>10.0f} {err_topk:>6.1f}% "
                  f"{pred_erlang_mean:>10.0f} {err_erlang:>6.1f}% "
                  f"{pred_hybrid_mean:>10.0f} {err_hybrid:>6.1f}%")

            all_results.append({
                "dist": dist_name,
                "threshold": threshold,
                "actual": actual_mean,
                "pred_topk": pred_topk_mean,
                "pred_erlang": pred_erlang_mean,
                "pred_hybrid": pred_hybrid_mean,
                "err_topk": err_topk,
                "err_erlang": err_erlang,
                "err_hybrid": err_hybrid,
            })

    # ===================================================================
    # PHASE 2: Top-k sensitivity (how does k affect accuracy?)
    # ===================================================================
    print("\n\n" + "=" * 78)
    print("  PHASE 2: TOP-K SENSITIVITY (threshold=0.001)")
    print("=" * 78)

    fixed_thresh = 0.001
    topk_sensitivity = []

    for dist_name, weights in distributions.items():
        actual_counts = count_above_threshold(weights, fixed_thresh)
        actual_mean = float(mx.mean(actual_counts.astype(mx.float32)).item())

        print(f"\n  {dist_name} (actual={actual_mean:.0f}):")
        print(f"  {'k':>6s} {'Predicted':>10s} {'Error%':>8s}")
        print(f"  {'-'*6} {'-'*10} {'-'*8}")

        for k in TOP_K_VALUES:
            pred = predict_workload_topk(weights, fixed_thresh, k=k)
            pred_mean = float(np.mean(pred))
            err = abs(pred_mean - actual_mean) / max(actual_mean, 1) * 100

            print(f"  {k:>6d} {pred_mean:>10.0f} {err:>7.1f}%")
            topk_sensitivity.append({
                "dist": dist_name,
                "k": k,
                "actual": actual_mean,
                "predicted": pred_mean,
                "error": err,
            })

    # ===================================================================
    # PHASE 3: Dispatch decision accuracy
    # ===================================================================
    print("\n\n" + "=" * 78)
    print("  PHASE 3: DISPATCH DECISION ACCURACY")
    print("=" * 78)
    print(f"\n  Dispatch policy:")
    print(f"    SKIP:   predicted_active < {SKIP_THRESHOLD*100:.0f}% "
          f"of L_kv ({int(L_KV * SKIP_THRESHOLD)} positions)")
    print(f"    SPARSE: predicted_active < {SPARSE_THRESHOLD*100:.0f}% "
          f"of L_kv ({int(L_KV * SPARSE_THRESHOLD)} positions)")
    print(f"    DENSE:  predicted_active > {DENSE_THRESHOLD*100:.0f}% "
          f"of L_kv ({int(L_KV * DENSE_THRESHOLD)} positions)")

    dispatch_results = []

    for dist_name, weights in distributions.items():
        print(f"\n  --- {dist_name} ---")
        print(f"  {'Threshold':>10s} {'Oracle':>8s} {'TopK':>8s} "
              f"{'Match':>6s} {'Erlang':>8s} {'Match':>6s} "
              f"{'Hybrid':>8s} {'Match':>6s}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8} "
              f"{'-'*6} {'-'*8} {'-'*6}")

        for threshold in THRESHOLDS:
            actual_counts = count_above_threshold(weights, threshold)
            pred_topk = predict_workload_topk(weights, threshold, k=100)
            pred_erlang = predict_workload_utilization(weights, threshold)
            pred_hybrid = predict_workload_hybrid(weights, threshold, k=100)

            # Compute per-head dispatch decisions
            n_total = 0
            n_match_topk = 0
            n_match_erlang = 0
            n_match_hybrid = 0

            actual_np = np.array(actual_counts.reshape(-1))
            pred_topk_flat = pred_topk.reshape(-1)
            pred_erlang_flat = pred_erlang.reshape(-1)
            pred_hybrid_flat = pred_hybrid.reshape(-1)

            for i in range(len(actual_np)):
                oracle_d = oracle_dispatch(float(actual_np[i]), L_KV)
                topk_d = decide_dispatch(float(pred_topk_flat[i]), L_KV)
                erlang_d = decide_dispatch(float(pred_erlang_flat[i]), L_KV)
                hybrid_d = decide_dispatch(float(pred_hybrid_flat[i]), L_KV)

                n_total += 1
                if topk_d == oracle_d:
                    n_match_topk += 1
                if erlang_d == oracle_d:
                    n_match_erlang += 1
                if hybrid_d == oracle_d:
                    n_match_hybrid += 1

            # Most common oracle dispatch for display
            oracle_decisions = [oracle_dispatch(float(actual_np[i]), L_KV)
                                for i in range(len(actual_np))]
            oracle_mode = max(set(oracle_decisions), key=oracle_decisions.count)

            match_topk = n_match_topk / n_total * 100
            match_erlang = n_match_erlang / n_total * 100
            match_hybrid = n_match_hybrid / n_total * 100

            topk_decisions = [decide_dispatch(float(pred_topk_flat[i]), L_KV)
                              for i in range(len(pred_topk_flat))]
            topk_mode = max(set(topk_decisions), key=topk_decisions.count)

            erlang_decisions = [decide_dispatch(float(pred_erlang_flat[i]), L_KV)
                                for i in range(len(pred_erlang_flat))]
            erlang_mode = max(set(erlang_decisions), key=erlang_decisions.count)

            hybrid_decisions = [decide_dispatch(float(pred_hybrid_flat[i]), L_KV)
                                for i in range(len(pred_hybrid_flat))]
            hybrid_mode = max(set(hybrid_decisions), key=hybrid_decisions.count)

            print(f"  {threshold:>10.4f} {oracle_mode:>8s} {topk_mode:>8s} "
                  f"{match_topk:>5.0f}% {erlang_mode:>8s} {match_erlang:>5.0f}% "
                  f"{hybrid_mode:>8s} {match_hybrid:>5.0f}%")

            dispatch_results.append({
                "dist": dist_name,
                "threshold": threshold,
                "oracle_mode": oracle_mode,
                "topk_mode": topk_mode,
                "match_topk": match_topk,
                "erlang_mode": erlang_mode,
                "match_erlang": match_erlang,
                "hybrid_mode": hybrid_mode,
                "match_hybrid": match_hybrid,
            })

    # ===================================================================
    # PHASE 4: Prediction cost measurement
    # ===================================================================
    print("\n\n" + "=" * 78)
    print("  PHASE 4: PREDICTION COST (microseconds)")
    print("=" * 78)

    # Use concentrated distribution for timing (most realistic dispatch case)
    w_timing = distributions["Concentrated"]
    test_thresh = 0.001

    print(f"\n  Method               Median (us)   Mean (us)")
    print(f"  {'-'*20}  {'-'*11}   {'-'*9}")

    # Full scan
    med_scan, mean_scan = time_full_scan(w_timing, test_thresh)
    print(f"  Full scan            {med_scan:>11.1f}   {mean_scan:>9.1f}")

    # Top-k predictions at various k
    topk_times = {}
    for k in TOP_K_VALUES:
        fn = lambda w, t, _k=k: predict_workload_topk(w, t, _k)
        med, mean = time_prediction(fn, w_timing, test_thresh)
        topk_times[k] = (med, mean)
        print(f"  Top-k (k={k:<4d})       {med:>11.1f}   {mean:>9.1f}")

    # Erlang utilization
    med_erlang, mean_erlang = time_prediction(
        predict_workload_utilization, w_timing, test_thresh
    )
    print(f"  Erlang utilization   {med_erlang:>11.1f}   {mean_erlang:>9.1f}")

    # Hybrid
    fn_hybrid = lambda w, t: predict_workload_hybrid(w, t, k=100)
    med_hybrid, mean_hybrid = time_prediction(fn_hybrid, w_timing, test_thresh)
    print(f"  Hybrid (k=100)       {med_hybrid:>11.1f}   {mean_hybrid:>9.1f}")

    cost_results = {
        "full_scan": (med_scan, mean_scan),
        "topk": topk_times,
        "erlang": (med_erlang, mean_erlang),
        "hybrid": (med_hybrid, mean_hybrid),
    }

    # Break-even analysis
    print(f"\n  Break-even analysis (is prediction cheaper than scanning?):")
    for k, (med, _) in topk_times.items():
        ratio = med / med_scan if med_scan > 0 else float('inf')
        cheaper = "YES" if ratio < 1.0 else "NO"
        print(f"    Top-k (k={k}): {ratio:.2f}x of scan cost -> {cheaper}")
    erlang_ratio = med_erlang / med_scan if med_scan > 0 else float('inf')
    print(f"    Erlang: {erlang_ratio:.2f}x of scan cost "
          f"-> {'YES' if erlang_ratio < 1.0 else 'NO'}")
    hybrid_ratio = med_hybrid / med_scan if med_scan > 0 else float('inf')
    print(f"    Hybrid: {hybrid_ratio:.2f}x of scan cost "
          f"-> {'YES' if hybrid_ratio < 1.0 else 'NO'}")

    # ===================================================================
    # PHASE 5: Context length scaling
    # ===================================================================
    print("\n\n" + "=" * 78)
    print("  PHASE 5: CONTEXT LENGTH SCALING")
    print("=" * 78)
    print(f"\n  How does prediction cost scale with L_kv?")

    scaling_results = []
    context_lengths = [1024, 4096, 8192, 16384, 32768, 65536]

    print(f"\n  {'L_kv':>8s} {'Scan (us)':>12s} {'TopK-100':>12s} "
          f"{'Erlang':>12s} {'Scan/TopK':>10s} {'Scan/Erlang':>12s}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    for l_kv in context_lengths:
        shape = (B, N_HEADS, L_Q, l_kv)
        w_scale = make_concentrated_weights(rng, shape)
        mx.eval(w_scale)

        # Time full scan
        med_s, _ = time_full_scan(w_scale, test_thresh,
                                   warmup=3, trials=20)

        # Time top-k prediction
        fn_tk = lambda w, t: predict_workload_topk(w, t, k=100)
        med_tk, _ = time_prediction(fn_tk, w_scale, test_thresh,
                                     warmup=3, trials=20)

        # Time Erlang prediction
        med_er, _ = time_prediction(predict_workload_utilization,
                                     w_scale, test_thresh,
                                     warmup=3, trials=20)

        ratio_topk = med_s / med_tk if med_tk > 0 else 0
        ratio_erlang = med_s / med_er if med_er > 0 else 0

        print(f"  {l_kv:>8d} {med_s:>12.1f} {med_tk:>12.1f} "
              f"{med_er:>12.1f} {ratio_topk:>10.2f}x {ratio_erlang:>10.2f}x")

        scaling_results.append({
            "L_kv": l_kv,
            "scan_us": med_s,
            "topk_us": med_tk,
            "erlang_us": med_er,
            "ratio_topk": ratio_topk,
            "ratio_erlang": ratio_erlang,
        })

        del w_scale
        gc.collect()

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    # Accuracy summary: average error across all distributions
    print("\n  Average prediction error by model:")
    for model in ["topk", "erlang", "hybrid"]:
        errors = [r[f"err_{model}"] for r in all_results]
        print(f"    {model:>8s}: mean={np.mean(errors):.1f}%, "
              f"median={np.median(errors):.1f}%, "
              f"max={np.max(errors):.1f}%")

    # Per-distribution best model
    print("\n  Best model per distribution (lowest mean error):")
    for dist_name in distributions:
        dist_results = [r for r in all_results if r["dist"] == dist_name]
        best_err = float('inf')
        best_model = "topk"
        for model in ["topk", "erlang", "hybrid"]:
            errs = [r[f"err_{model}"] for r in dist_results]
            mean_err = np.mean(errs)
            if mean_err < best_err:
                best_err = mean_err
                best_model = model
        print(f"    {dist_name:<15s}: {best_model} "
              f"(mean error {best_err:.1f}%)")

    # Dispatch accuracy summary
    print("\n  Dispatch decision accuracy (all distributions, all thresholds):")
    for model in ["topk", "erlang", "hybrid"]:
        matches = [r[f"match_{model}"] for r in dispatch_results]
        print(f"    {model:>8s}: mean={np.mean(matches):.1f}%, "
              f"min={np.min(matches):.0f}%")

    # Key findings
    print("\n  KEY FINDINGS:")

    # 1. Can we predict within 20%?
    topk_errors = [r["err_topk"] for r in all_results]
    hybrid_errors = [r["err_hybrid"] for r in all_results]
    within_20_topk = sum(1 for e in topk_errors if e <= 20) / len(topk_errors) * 100
    within_20_hybrid = sum(1 for e in hybrid_errors if e <= 20) / len(hybrid_errors) * 100
    print(f"  1. Predictions within 20% error:")
    print(f"     Top-k:  {within_20_topk:.0f}% of cases")
    print(f"     Hybrid: {within_20_hybrid:.0f}% of cases")

    # 2. Is prediction cheaper?
    print(f"  2. Cost ratio (prediction / full scan):")
    print(f"     Top-k(100): {topk_times[100][0] / med_scan:.2f}x")
    print(f"     Erlang:     {med_erlang / med_scan:.2f}x")

    # 3. Dispatch accuracy
    topk_dispatch = [r["match_topk"] for r in dispatch_results]
    print(f"  3. Dispatch decision accuracy:")
    print(f"     Top-k:  {np.mean(topk_dispatch):.0f}% correct "
          f"(vs oracle)")
    hybrid_dispatch = [r["match_hybrid"] for r in dispatch_results]
    print(f"     Hybrid: {np.mean(hybrid_dispatch):.0f}% correct")

    # ===================================================================
    # Save results to markdown
    # ===================================================================
    md_path = os.path.join(os.path.dirname(__file__), "EXP9_RESULTS.md")
    save_results_md(md_path, all_results, topk_sensitivity, dispatch_results,
                    cost_results, scaling_results, distributions, dist_entropy)
    print(f"\n  Results saved to: {md_path}")

    print(f"\n{'='*78}")
    print("  EXPERIMENT 9 COMPLETE")
    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def save_results_md(md_path, all_results, topk_sensitivity, dispatch_results,
                    cost_results, scaling_results, distributions, dist_entropy):
    """Write comprehensive results to markdown."""
    with open(md_path, "w") as f:
        f.write("# Experiment 9: Erlang Queuing Model for Sparse V "
                "Workload Prediction\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} "
                f"(GQA 4:1), D={D}, bits={BITS}  \n")
        f.write(f"**Context:** L_kv={L_KV:,}, L_q={L_Q} (decode)  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        # Hypothesis
        f.write("## Hypothesis\n\n")
        f.write("The SV kernel iterates ALL L_kv positions to check "
                "`if (|wn_val| > threshold)` even though concentrated heads "
                "skip ~99% of positions. Attention weight distributions have "
                "predictable structure. From the top-k attention weights, we "
                "can estimate the total number of above-threshold positions "
                "using Erlang-style utilization analysis, without scanning all "
                "positions.\n\n")
        f.write("This enables:\n")
        f.write("1. Pre-computing expected workload per head\n")
        f.write("2. Choosing kernel dispatch strategy (skip / sparse / dense)\n")
        f.write("3. Skipping the kernel entirely when predicted workload ~ 0\n\n")

        # Models
        f.write("## Prediction Models\n\n")
        f.write("### Top-k Exponential Tail\n")
        f.write("Sample top-k weights (partial sort, O(L_kv)). Fit exponential "
                "decay from w_1 (largest) to w_k (k-th). Extrapolate to find rank "
                "where weight drops below threshold.\n\n")
        f.write("### Erlang Utilization\n")
        f.write("Map to queuing theory: arrival_rate = mean attention weight, "
                "service_rate = threshold. Utilization rho = mean/threshold. "
                "For exponential distribution: P(w > t) = exp(-t/mean). "
                "Expected active = L_kv * P(w > threshold).\n\n")
        f.write("### Hybrid (Top-k + Erlang)\n")
        f.write("Use top-k to calibrate the tail decay rate, then use Erlang-style "
                "CDF extrapolation for the unseen tail.\n\n")

        # Distribution characterization
        f.write("## Distributions Tested\n\n")
        f.write("| Distribution | Mean Entropy | Description |\n")
        f.write("|:-------------|:-----------:|:------------|\n")
        for name in distributions:
            ent = dist_entropy[name]
            descs = {
                "Concentrated": "~50 positions with high weight (softmax temp=8)",
                "Moderate": "~500 positions with meaningful weight (softmax temp=3)",
                "Spread": "Nearly uniform (softmax temp=0.1)",
                "Power-law": "Zipf distribution (alpha=1.2), realistic for transformers",
                "Bimodal": "Two Gaussian peaks at 1/3 and 2/3 of context",
            }
            f.write(f"| {name} | {ent:.4f} | {descs.get(name, '')} |\n")

        # Phase 1: Prediction accuracy
        f.write("\n## Prediction Accuracy\n\n")
        f.write("| Distribution | Threshold | Actual | Top-k | "
                "Err% | Erlang | Err% | Hybrid | Err% |\n")
        f.write("|:-------------|----------:|-------:|------:|"
                "----:|-------:|----:|-------:|----:|\n")
        for r in all_results:
            f.write(f"| {r['dist']} | {r['threshold']:.4f} | "
                    f"{r['actual']:.0f} | {r['pred_topk']:.0f} | "
                    f"{r['err_topk']:.1f}% | {r['pred_erlang']:.0f} | "
                    f"{r['err_erlang']:.1f}% | {r['pred_hybrid']:.0f} | "
                    f"{r['err_hybrid']:.1f}% |\n")

        # Phase 2: Top-k sensitivity
        f.write("\n## Top-k Sensitivity (threshold=0.001)\n\n")
        f.write("| Distribution | k=50 | k=100 | k=200 | k=500 |\n")
        f.write("|:-------------|-----:|------:|------:|------:|\n")

        for dist_name in distributions:
            row = f"| {dist_name}"
            for k in TOP_K_VALUES:
                match = [r for r in topk_sensitivity
                         if r["dist"] == dist_name and r["k"] == k]
                if match:
                    row += f" | {match[0]['error']:.1f}%"
                else:
                    row += " | -"
            row += " |\n"
            f.write(row)

        # Phase 3: Dispatch accuracy
        f.write("\n## Dispatch Decision Accuracy\n\n")
        f.write(f"Dispatch policy: SKIP (<{SKIP_THRESHOLD*100:.0f}%), "
                f"SPARSE (<{SPARSE_THRESHOLD*100:.0f}%), "
                f"DENSE (>{DENSE_THRESHOLD*100:.0f}%)\n\n")
        f.write("| Distribution | Threshold | Oracle | Top-k | Match | "
                "Erlang | Match | Hybrid | Match |\n")
        f.write("|:-------------|----------:|:------:|:-----:|------:|"
                ":-----:|------:|:------:|------:|\n")
        for r in dispatch_results:
            f.write(f"| {r['dist']} | {r['threshold']:.4f} | "
                    f"{r['oracle_mode']} | {r['topk_mode']} | "
                    f"{r['match_topk']:.0f}% | {r['erlang_mode']} | "
                    f"{r['match_erlang']:.0f}% | {r['hybrid_mode']} | "
                    f"{r['match_hybrid']:.0f}% |\n")

        # Phase 4: Cost
        f.write("\n## Prediction Cost\n\n")
        scan_med, scan_mean = cost_results["full_scan"]
        f.write(f"At L_kv={L_KV:,}, concentrated distribution, "
                f"threshold=0.001:\n\n")
        f.write("| Method | Median (us) | Mean (us) | vs Scan |\n")
        f.write("|:-------|:-----------:|:---------:|:-------:|\n")
        f.write(f"| Full scan | {scan_med:.1f} | {scan_mean:.1f} | 1.00x |\n")
        for k, (med, mean) in cost_results["topk"].items():
            ratio = med / scan_med if scan_med > 0 else 0
            f.write(f"| Top-k (k={k}) | {med:.1f} | {mean:.1f} | "
                    f"{ratio:.2f}x |\n")
        er_med, er_mean = cost_results["erlang"]
        ratio_er = er_med / scan_med if scan_med > 0 else 0
        f.write(f"| Erlang utilization | {er_med:.1f} | {er_mean:.1f} | "
                f"{ratio_er:.2f}x |\n")
        hy_med, hy_mean = cost_results["hybrid"]
        ratio_hy = hy_med / scan_med if scan_med > 0 else 0
        f.write(f"| Hybrid (k=100) | {hy_med:.1f} | {hy_mean:.1f} | "
                f"{ratio_hy:.2f}x |\n")

        # Phase 5: Scaling
        f.write("\n## Context Length Scaling\n\n")
        f.write("| L_kv | Scan (us) | Top-k (us) | Erlang (us) | "
                "Scan/Top-k | Scan/Erlang |\n")
        f.write("|-----:|:---------:|:----------:|:-----------:|"
                ":---------:|:-----------:|\n")
        for s in scaling_results:
            f.write(f"| {s['L_kv']:,} | {s['scan_us']:.1f} | "
                    f"{s['topk_us']:.1f} | {s['erlang_us']:.1f} | "
                    f"{s['ratio_topk']:.2f}x | "
                    f"{s['ratio_erlang']:.2f}x |\n")

        # Summary statistics
        f.write("\n## Summary\n\n")

        topk_errors = [r["err_topk"] for r in all_results]
        erlang_errors = [r["err_erlang"] for r in all_results]
        hybrid_errors = [r["err_hybrid"] for r in all_results]

        f.write("### Prediction Error (across all distributions and thresholds)\n\n")
        f.write("| Model | Mean Error | Median Error | Max Error "
                "| Cases <20% |\n")
        f.write("|:------|:---------:|:-----------:|:---------:|"
                ":---------:|\n")
        for name, errs in [("Top-k (k=100)", topk_errors),
                           ("Erlang utilization", erlang_errors),
                           ("Hybrid", hybrid_errors)]:
            within_20 = sum(1 for e in errs if e <= 20) / len(errs) * 100
            f.write(f"| {name} | {np.mean(errs):.1f}% | "
                    f"{np.median(errs):.1f}% | {np.max(errs):.1f}% | "
                    f"{within_20:.0f}% |\n")

        f.write("\n### Dispatch Decision Accuracy\n\n")
        f.write("| Model | Mean Accuracy | Min Accuracy |\n")
        f.write("|:------|:------------:|:------------:|\n")
        for model in ["topk", "erlang", "hybrid"]:
            matches = [r[f"match_{model}"] for r in dispatch_results]
            f.write(f"| {model.capitalize()} | {np.mean(matches):.1f}% | "
                    f"{np.min(matches):.0f}% |\n")

        # Analysis
        f.write("\n## Analysis\n\n")

        # Which model is best?
        mean_topk = np.mean(topk_errors)
        mean_erlang = np.mean(erlang_errors)
        mean_hybrid = np.mean(hybrid_errors)
        best_model = min([("Top-k", mean_topk),
                          ("Erlang", mean_erlang),
                          ("Hybrid", mean_hybrid)],
                         key=lambda x: x[1])

        f.write(f"**Best prediction model:** {best_model[0]} "
                f"(mean error {best_model[1]:.1f}%)\n\n")

        # Where does each model struggle?
        f.write("### Model Strengths and Weaknesses\n\n")

        for dist_name in distributions:
            dist_r = [r for r in all_results if r["dist"] == dist_name]
            best_for_dist = min(
                [("Top-k", np.mean([r["err_topk"] for r in dist_r])),
                 ("Erlang", np.mean([r["err_erlang"] for r in dist_r])),
                 ("Hybrid", np.mean([r["err_hybrid"] for r in dist_r]))],
                key=lambda x: x[1]
            )
            f.write(f"- **{dist_name}**: Best={best_for_dist[0]} "
                    f"({best_for_dist[1]:.1f}% mean error)\n")

        # Cost-effectiveness
        f.write("\n### Cost-Effectiveness\n\n")
        scan_med = cost_results["full_scan"][0]
        topk100_med = cost_results["topk"][100][0]
        erlang_med = cost_results["erlang"][0]

        if topk100_med < scan_med:
            f.write(f"Top-k (k=100) is **{scan_med/topk100_med:.1f}x faster** "
                    f"than full scan, making it a viable pre-check.\n\n")
        else:
            f.write(f"Top-k (k=100) is **{topk100_med/scan_med:.1f}x slower** "
                    f"than full scan. The overhead of prediction exceeds the "
                    f"cost of just running the threshold check.\n\n")

        if erlang_med < scan_med:
            f.write(f"Erlang utilization is **{scan_med/erlang_med:.1f}x faster** "
                    f"than full scan, offering the cheapest prediction.\n\n")
        else:
            f.write(f"Erlang utilization is **{erlang_med/scan_med:.1f}x slower** "
                    f"than full scan.\n\n")

        # Context scaling insight
        f.write("### Scaling Behavior\n\n")
        if len(scaling_results) >= 2:
            first = scaling_results[0]
            last = scaling_results[-1]
            scan_growth = last["scan_us"] / first["scan_us"]
            topk_growth = last["topk_us"] / first["topk_us"]
            lkv_growth = last["L_kv"] / first["L_kv"]

            f.write(f"Over {lkv_growth:.0f}x context length increase "
                    f"({first['L_kv']:,} -> {last['L_kv']:,}):\n")
            f.write(f"- Full scan cost grows {scan_growth:.1f}x\n")
            f.write(f"- Top-k cost grows {topk_growth:.1f}x\n")
            f.write(f"- {'Top-k scales better' if topk_growth < scan_growth else 'Both scale similarly'}\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")

        # Determine overall verdict
        topk_within_20 = sum(1 for e in topk_errors if e <= 20) / len(topk_errors) * 100
        hybrid_within_20 = sum(1 for e in hybrid_errors if e <= 20) / len(hybrid_errors) * 100

        topk_dispatch_acc = np.mean([r["match_topk"] for r in dispatch_results])
        hybrid_dispatch_acc = np.mean([r["match_hybrid"] for r in dispatch_results])

        topk_cheaper = cost_results["topk"][100][0] < cost_results["full_scan"][0]
        erlang_cheaper = cost_results["erlang"][0] < cost_results["full_scan"][0]

        best_within_20 = max(topk_within_20, hybrid_within_20)
        best_dispatch = max(topk_dispatch_acc, hybrid_dispatch_acc)

        if best_within_20 >= 70 and best_dispatch >= 80 and (topk_cheaper or erlang_cheaper):
            f.write("### Verdict: POSITIVE\n\n")
            f.write("Erlang-style workload prediction is viable for sparse V "
                    "dispatch decisions:\n\n")
            f.write(f"- **{best_within_20:.0f}% of predictions within 20% error** "
                    f"-- sufficient for dispatch routing\n")
            f.write(f"- **{best_dispatch:.0f}% dispatch decision accuracy** "
                    f"-- correctly routes skip/sparse/dense\n")
            if topk_cheaper:
                speedup = cost_results["full_scan"][0] / cost_results["topk"][100][0]
                f.write(f"- **Prediction is {speedup:.1f}x cheaper than full scan** "
                        f"-- net win for workload estimation\n\n")
            if erlang_cheaper:
                speedup = cost_results["full_scan"][0] / cost_results["erlang"][0]
                f.write(f"- **Erlang model is {speedup:.1f}x cheaper than full scan**\n\n")
            f.write("### Integration Path\n\n")
            f.write("1. Before SV kernel dispatch, run top-k workload prediction "
                    "(~microseconds)\n")
            f.write("2. If predicted_active < 1% of L_kv: skip kernel entirely "
                    "(output ~ 0)\n")
            f.write("3. If predicted_active < 10%: use current sparse threshold\n")
            f.write("4. If predicted_active > 50%: disable threshold (dense mode "
                    "avoids branch overhead)\n")
            f.write("5. This converts the O(L_kv) threshold-check overhead to "
                    "O(k + L_kv) partial sort + O(1) prediction\n")
        elif best_within_20 >= 50 and best_dispatch >= 60:
            f.write("### Verdict: PARTIAL\n\n")
            f.write("Workload prediction shows promise but is not reliable "
                    "enough for production dispatch:\n\n")
            f.write(f"- {best_within_20:.0f}% of predictions within 20% error "
                    f"(target: >70%)\n")
            f.write(f"- {best_dispatch:.0f}% dispatch accuracy "
                    f"(target: >80%)\n")
            if not topk_cheaper and not erlang_cheaper:
                f.write("- Prediction cost exceeds scan cost -- no cost benefit\n")
            f.write("\nThe approach may be viable at longer context lengths "
                    "where the scan becomes more expensive.\n")
        else:
            f.write("### Verdict: NEGATIVE\n\n")
            f.write("Workload prediction does not provide sufficient accuracy "
                    "or cost benefit for dispatch decisions:\n\n")
            f.write(f"- Only {best_within_20:.0f}% of predictions within "
                    f"20% error\n")
            f.write(f"- {best_dispatch:.0f}% dispatch accuracy\n")
            f.write("- The current approach (iterate all positions, check threshold) "
                    "remains the best strategy.\n")
            f.write("\nThe SV kernel's branch check is already very cheap on Metal "
                    "(~1 cycle per position). Prediction overhead cannot amortize "
                    "this for moderate context lengths.\n")

        f.write("\n---\n")
        f.write(f"*Generated by exp9_erlang_workload.py on "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


if __name__ == "__main__":
    main()
