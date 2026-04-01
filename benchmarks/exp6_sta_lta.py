#!/usr/bin/env python3
"""
Experiment 6: STA/LTA Change-Point Detection for Entropy Amortization

Hypothesis: Attention patterns don't change drastically between consecutive
decode tokens — they drift slowly. We can compute entropy once, cache the
per-head thresholds, and only recompute when STA/LTA detects a shift.
Between shifts, reuse cached thresholds.

Inspired by AAPM's seismic_detection.py: STA/LTA (Short-Term Average /
Long-Term Average) ratio detects sudden changes in a signal. When the ratio
is stable (~1.0), nothing has changed. When it spikes (>2.0), a significant
shift occurred.

Key question: Can we reduce entropy computations by >80% while catching
transitions within 5 steps?

Strategies compared:
  a) Always recompute: entropy every step (current Phase 2a behavior)
  b) STA/LTA gated: only recompute when change-point detected
  c) Fixed interval: recompute every N steps (N=10, 25, 50)

STA/LTA tracked statistic: mean of max-attention-per-head — O(n_heads * L_kv),
much cheaper than full Shannon entropy which is O(n_heads * L_kv * log).

Usage:
    cd ~/workspace/polarquant-metal
    python3 benchmarks/exp6_sta_lta.py
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
N_HEADS = 8          # query heads
N_KV_HEADS = 2       # kv heads (GQA 4:1)
D = 128              # head_dim
L_KV_START = 2048    # starting context length
N_STEPS = 500        # decode steps to simulate
BITS = 3
MAX_THRESHOLD = 0.01
SPARSE_V_THRESHOLD = 1e-3

REP = N_HEADS // N_KV_HEADS  # 4

# Transition points in the 500-step decode
TRANSITION_1 = 150   # concentrated -> mixed
TRANSITION_2 = 350   # mixed -> concentrated

# STA/LTA parameter grid
STA_WINDOWS = [3, 5, 10]
LTA_WINDOWS = [20, 30, 50]
TRIGGER_RATIOS = [1.5, 2.0, 3.0]

# Fixed interval baselines
FIXED_INTERVALS = [10, 25, 50]


# ---------------------------------------------------------------------------
# STA/LTA Detector
# ---------------------------------------------------------------------------

class STALTADetector:
    """Seismic-style change-point detector for attention statistics.

    Tracks a scalar statistic over time. Computes short-term and long-term
    running averages. When the ratio STA/LTA deviates significantly from 1.0,
    a change-point is flagged.

    Args:
        sta_window: window size for short-term average
        lta_window: window size for long-term average
        trigger_ratio: STA/LTA ratio threshold to trigger recomputation
    """

    def __init__(self, sta_window=5, lta_window=30, trigger_ratio=2.0):
        self.sta_window = sta_window
        self.lta_window = lta_window
        self.trigger_ratio = trigger_ratio
        self.history = []
        self.trigger_count = 0
        self.trigger_steps = []

    def update(self, stat_value):
        """Add new observation. Returns True if change detected (recompute needed)."""
        self.history.append(stat_value)

        if len(self.history) < self.lta_window:
            # Not enough data yet — always recompute
            self.trigger_count += 1
            self.trigger_steps.append(len(self.history) - 1)
            return True

        sta = np.mean(self.history[-self.sta_window:])
        lta = np.mean(self.history[-self.lta_window:])
        ratio = sta / (lta + 1e-10)

        triggered = ratio > self.trigger_ratio or ratio < 1.0 / self.trigger_ratio
        if triggered:
            self.trigger_count += 1
            self.trigger_steps.append(len(self.history) - 1)
        return triggered

    def reset(self):
        """Reset detector state."""
        self.history.clear()
        self.trigger_count = 0
        self.trigger_steps.clear()

    @property
    def ratio_history(self):
        """Compute the full STA/LTA ratio time series for analysis."""
        ratios = []
        for i in range(len(self.history)):
            if i < self.lta_window - 1:
                ratios.append(float('nan'))
            else:
                sta = np.mean(self.history[max(0, i - self.sta_window + 1):i + 1])
                lta = np.mean(self.history[max(0, i - self.lta_window + 1):i + 1])
                ratios.append(sta / (lta + 1e-10))
        return ratios


# ---------------------------------------------------------------------------
# Evolving attention weight generator
# ---------------------------------------------------------------------------

def generate_evolving_weights(step, n_heads, L_kv):
    """Generate attention weights that shift at defined change points.

    Steps 0-149:   All heads concentrated (low entropy, stable)
    Steps 150-159: Transition (pattern shifts to mixed)
    Steps 160-349: Mixed (heads 0-3 concentrated, heads 4-7 spread)
    Steps 350-359: Transition back
    Steps 360-499: All heads concentrated again

    Returns:
        weights: (1, n_heads, 1, L_kv) normalized attention weights
    """
    weights_np = np.zeros((1, n_heads, 1, L_kv), dtype=np.float32)
    positions = np.arange(L_kv, dtype=np.float32)

    if step < 150 or step >= 360:
        # Concentrated: all heads focus on ~50 positions
        for h in range(n_heads):
            hot = (step * 7 + h * 13) % L_kv
            weights_np[0, h, 0] = np.exp(-0.01 * (positions - hot) ** 2)
    elif 150 <= step < 160 or 350 <= step < 360:
        # Transition: blend concentrated and mixed
        if step < 160:
            alpha = (step - 150) / 10.0  # 0->1 over transition
        else:
            alpha = 1.0 - (step - 350) / 10.0  # 1->0 over transition
        for h in range(n_heads):
            hot = (step * 7 + h * 13) % L_kv
            conc = np.exp(-0.01 * (positions - hot) ** 2)
            if h < 4:
                weights_np[0, h, 0] = conc
            else:
                spread = np.ones(L_kv, dtype=np.float32)
                weights_np[0, h, 0] = (1 - alpha) * conc + alpha * spread
    else:
        # Mixed: heads 0-3 concentrated, heads 4-7 spread
        for h in range(n_heads):
            if h < 4:
                hot = (step * 7 + h * 13) % L_kv
                weights_np[0, h, 0] = np.exp(-0.01 * (positions - hot) ** 2)
            else:
                weights_np[0, h, 0] = np.ones(L_kv, dtype=np.float32)

    # Normalize to probability distribution
    row_sums = weights_np.sum(axis=-1, keepdims=True)
    weights_np = weights_np / (row_sums + 1e-10)
    return mx.array(weights_np)


# ---------------------------------------------------------------------------
# Entropy computation (same as turboquant_cache._compute_adaptive_threshold)
# ---------------------------------------------------------------------------

def compute_entropy_thresholds(weights, n_heads, sparse_v_threshold=SPARSE_V_THRESHOLD):
    """Compute per-head entropy-guided sparse V thresholds.

    This mirrors TurboQuantKVCache._compute_adaptive_threshold exactly.
    """
    eps = 1e-10
    log_w = mx.log(weights + eps)
    head_entropy = -(weights * log_w).sum(axis=-1).mean(axis=(0, 2))  # (n_heads,)
    max_ent = math.log(weights.shape[-1])
    mx.eval(head_entropy)

    thresholds = []
    for h in range(n_heads):
        norm_ent = float(head_entropy[h].item()) / max_ent
        t = sparse_v_threshold / (1.0 + math.exp(10.0 * (norm_ent - 0.5)))
        thresholds.append(t)

    return mx.array(thresholds, dtype=mx.float32), head_entropy


def compute_cheap_stat(weights):
    """Cheap statistic for STA/LTA: mean of max-attention-per-head.

    O(n_heads * L_kv) — just one max reduction per head.
    Much cheaper than full entropy which is O(n_heads * L_kv * log).
    """
    max_per_head = mx.max(weights, axis=-1)  # (B, n_heads, L_q)
    stat = float(max_per_head.mean().item())
    return stat


# ---------------------------------------------------------------------------
# FP16 baseline: standard matmul
# ---------------------------------------------------------------------------

def fp16_sv_matmul(weights, values):
    """Standard FP16 attention output: weights @ V (with GQA expansion)."""
    if REP > 1:
        values_exp = mx.repeat(values, REP, axis=1)
    else:
        values_exp = values
    return weights @ values_exp


# ---------------------------------------------------------------------------
# Quantized SV pipeline
# ---------------------------------------------------------------------------

def setup_quantized_v(values_raw):
    """Quantize and pack V values."""
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
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    na = mx.sqrt(mx.sum(a_flat * a_flat))
    nb = mx.sqrt(mx.sum(b_flat * b_flat))
    result = float(dot / (na * nb + 1e-10))
    return result


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_always_recompute(steps_weights, values_raw, v_packed, v_norms,
                         val_centroids, pq_val):
    """Strategy A: Compute entropy every step (current Phase 2a)."""
    n_computations = 0
    cos_sims = []
    per_step_thresholds = []

    t0 = time.perf_counter()
    for step, weights in enumerate(steps_weights):
        # FP16 baseline for this step
        fp16_out = fp16_sv_matmul(weights, values_raw)

        # Always compute entropy thresholds
        thresholds, _ = compute_entropy_thresholds(weights, N_HEADS)
        n_computations += 1
        per_step_thresholds.append(np.array(thresholds))

        # Run quantized SV with adaptive thresholds
        quant_out = run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                                  pq_val, sparse_v_threshold=thresholds)
        mx.eval(quant_out, fp16_out)

        cos = cosine_similarity(fp16_out, quant_out)
        cos_sims.append(cos)

    total_time = time.perf_counter() - t0
    return {
        "n_computations": n_computations,
        "cos_sims": cos_sims,
        "mean_cos": np.mean(cos_sims),
        "min_cos": np.min(cos_sims),
        "total_time": total_time,
        "per_step_thresholds": per_step_thresholds,
    }


def run_sta_lta_gated(steps_weights, values_raw, v_packed, v_norms,
                      val_centroids, pq_val, sta_window, lta_window,
                      trigger_ratio):
    """Strategy B: Only recompute entropy when STA/LTA triggers."""
    detector = STALTADetector(sta_window, lta_window, trigger_ratio)
    cached_thresholds = None
    n_computations = 0
    cos_sims = []
    trigger_steps = []

    t0 = time.perf_counter()
    for step, weights in enumerate(steps_weights):
        fp16_out = fp16_sv_matmul(weights, values_raw)

        # Compute cheap statistic
        stat = compute_cheap_stat(weights)

        # Check if we need to recompute
        should_recompute = detector.update(stat)

        if should_recompute or cached_thresholds is None:
            thresholds, _ = compute_entropy_thresholds(weights, N_HEADS)
            cached_thresholds = thresholds
            n_computations += 1
            if step >= lta_window:
                trigger_steps.append(step)

        # Use cached or fresh thresholds
        quant_out = run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                                  pq_val, sparse_v_threshold=cached_thresholds)
        mx.eval(quant_out, fp16_out)

        cos = cosine_similarity(fp16_out, quant_out)
        cos_sims.append(cos)

    total_time = time.perf_counter() - t0

    # Analyze transition detection latency
    detection_latency_1 = None
    detection_latency_2 = None
    for ts in trigger_steps:
        if ts >= TRANSITION_1 and detection_latency_1 is None:
            detection_latency_1 = ts - TRANSITION_1
        if ts >= TRANSITION_2 and detection_latency_2 is None:
            detection_latency_2 = ts - TRANSITION_2

    return {
        "n_computations": n_computations,
        "cos_sims": cos_sims,
        "mean_cos": np.mean(cos_sims),
        "min_cos": np.min(cos_sims),
        "total_time": total_time,
        "trigger_steps": trigger_steps,
        "detection_latency_1": detection_latency_1,
        "detection_latency_2": detection_latency_2,
        "ratio_history": detector.ratio_history,
        "sta_window": sta_window,
        "lta_window": lta_window,
        "trigger_ratio": trigger_ratio,
    }


def run_fixed_interval(steps_weights, values_raw, v_packed, v_norms,
                       val_centroids, pq_val, interval):
    """Strategy C: Recompute entropy every N steps."""
    cached_thresholds = None
    n_computations = 0
    cos_sims = []

    t0 = time.perf_counter()
    for step, weights in enumerate(steps_weights):
        fp16_out = fp16_sv_matmul(weights, values_raw)

        if step % interval == 0 or cached_thresholds is None:
            thresholds, _ = compute_entropy_thresholds(weights, N_HEADS)
            cached_thresholds = thresholds
            n_computations += 1

        quant_out = run_sv_kernel(weights, v_packed, v_norms, val_centroids,
                                  pq_val, sparse_v_threshold=cached_thresholds)
        mx.eval(quant_out, fp16_out)

        cos = cosine_similarity(fp16_out, quant_out)
        cos_sims.append(cos)

    total_time = time.perf_counter() - t0

    # Detection latency: how many steps after transition before a recompute
    detection_latency_1 = interval - (TRANSITION_1 % interval)
    if detection_latency_1 == interval:
        detection_latency_1 = 0
    detection_latency_2 = interval - (TRANSITION_2 % interval)
    if detection_latency_2 == interval:
        detection_latency_2 = 0

    return {
        "n_computations": n_computations,
        "cos_sims": cos_sims,
        "mean_cos": np.mean(cos_sims),
        "min_cos": np.min(cos_sims),
        "total_time": total_time,
        "interval": interval,
        "detection_latency_1": detection_latency_1,
        "detection_latency_2": detection_latency_2,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Experiment 6: STA/LTA Change-Point Detection for Entropy Amortization")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
          f"D={D}, bits={BITS}")
    print(f"Context:    L_kv starts at {L_KV_START}, {N_STEPS} decode steps")
    print(f"Transitions: step {TRANSITION_1} (conc->mixed), "
          f"step {TRANSITION_2} (mixed->conc)")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()

    # -----------------------------------------------------------------------
    # Setup: generate V values and pre-quantize
    # -----------------------------------------------------------------------
    rng = np.random.RandomState(42)
    L_kv = L_KV_START + N_STEPS  # full context after all steps

    values_raw = mx.array(
        rng.standard_normal((B, N_KV_HEADS, L_kv, D)).astype(np.float32)
    )
    mx.eval(values_raw)

    pq_val, v_packed, v_norms, val_centroids = setup_quantized_v(values_raw)

    # -----------------------------------------------------------------------
    # Pre-generate all 500 steps of evolving attention weights
    # -----------------------------------------------------------------------
    print("Generating evolving attention weights for 500 steps...")
    steps_weights = []
    for step in range(N_STEPS):
        w = generate_evolving_weights(step, N_HEADS, L_kv)
        mx.eval(w)
        steps_weights.append(w)
    print(f"  Done. Shape per step: {steps_weights[0].shape}")

    # Verify the pattern looks right: compute entropy at key points
    print("\n  Spot-check entropy at key steps:")
    for check_step in [0, 75, 149, 155, 200, 349, 355, 400, 499]:
        w = steps_weights[check_step]
        _, h_ent = compute_entropy_thresholds(w, N_HEADS)
        max_ent = math.log(L_kv)
        ent_list = [float(h_ent[h].item()) / max_ent for h in range(N_HEADS)]
        mean_ent = np.mean(ent_list)
        print(f"    Step {check_step:3d}: mean_norm_entropy={mean_ent:.4f}  "
              f"heads=[{', '.join(f'{e:.3f}' for e in ent_list)}]")

    # Also check the cheap stat at key points
    print("\n  Spot-check cheap stat (mean max-attention) at key steps:")
    for check_step in [0, 75, 149, 155, 200, 349, 355, 400, 499]:
        stat = compute_cheap_stat(steps_weights[check_step])
        print(f"    Step {check_step:3d}: stat={stat:.6f}")

    # ===================================================================
    # Strategy A: Always recompute
    # ===================================================================
    print("\n" + "=" * 78)
    print("  STRATEGY A: Always Recompute (current Phase 2a)")
    print("=" * 78)

    result_always = run_always_recompute(
        steps_weights, values_raw, v_packed, v_norms, val_centroids, pq_val
    )
    print(f"  Entropy computations: {result_always['n_computations']}/{N_STEPS}")
    print(f"  Mean cosine sim:      {result_always['mean_cos']:.6f}")
    print(f"  Min cosine sim:       {result_always['min_cos']:.6f}")
    print(f"  Total time:           {result_always['total_time']:.2f}s")

    # ===================================================================
    # Strategy B: STA/LTA gated (parameter sweep)
    # ===================================================================
    print("\n" + "=" * 78)
    print("  STRATEGY B: STA/LTA Gated — Parameter Sweep")
    print("=" * 78)

    sta_lta_results = []
    for sta_w in STA_WINDOWS:
        for lta_w in LTA_WINDOWS:
            if sta_w >= lta_w:
                continue  # STA must be shorter than LTA
            for trig in TRIGGER_RATIOS:
                result = run_sta_lta_gated(
                    steps_weights, values_raw, v_packed, v_norms,
                    val_centroids, pq_val, sta_w, lta_w, trig
                )
                sta_lta_results.append(result)

                reduction = (1 - result['n_computations'] / N_STEPS) * 100
                lat1 = result['detection_latency_1']
                lat2 = result['detection_latency_2']
                lat1_str = f"{lat1}" if lat1 is not None else "MISSED"
                lat2_str = f"{lat2}" if lat2 is not None else "MISSED"

                print(f"  STA={sta_w:2d} LTA={lta_w:2d} trig={trig:.1f}: "
                      f"computes={result['n_computations']:3d}/{N_STEPS} "
                      f"({reduction:5.1f}% reduction)  "
                      f"cos={result['mean_cos']:.6f}  "
                      f"lat=[{lat1_str}, {lat2_str}]  "
                      f"time={result['total_time']:.2f}s")

    # ===================================================================
    # Strategy C: Fixed interval
    # ===================================================================
    print("\n" + "=" * 78)
    print("  STRATEGY C: Fixed Interval Recomputation")
    print("=" * 78)

    fixed_results = []
    for interval in FIXED_INTERVALS:
        result = run_fixed_interval(
            steps_weights, values_raw, v_packed, v_norms,
            val_centroids, pq_val, interval
        )
        fixed_results.append(result)

        reduction = (1 - result['n_computations'] / N_STEPS) * 100
        print(f"  Interval={interval:2d}: "
              f"computes={result['n_computations']:3d}/{N_STEPS} "
              f"({reduction:5.1f}% reduction)  "
              f"cos={result['mean_cos']:.6f}  "
              f"lat=[{result['detection_latency_1']}, "
              f"{result['detection_latency_2']}]  "
              f"time={result['total_time']:.2f}s")

    # ===================================================================
    # Analysis: find best STA/LTA config
    # ===================================================================
    print("\n" + "=" * 78)
    print("  ANALYSIS")
    print("=" * 78)

    # Filter STA/LTA results that meet our criteria:
    # >80% reduction AND transition caught within 5 steps
    quality_floor = result_always['mean_cos'] - 0.001  # allow 0.001 degradation

    print(f"\n  Baseline (always recompute): "
          f"cos={result_always['mean_cos']:.6f}, "
          f"time={result_always['total_time']:.2f}s")
    print(f"  Quality floor (baseline - 0.001): {quality_floor:.6f}")

    print(f"\n  Candidates meeting >80% reduction + quality floor:")
    print(f"  {'Config':<28s} {'Computes':>9s} {'Reduction':>10s} "
          f"{'Cos Sim':>9s} {'Lat1':>5s} {'Lat2':>5s} {'Time':>7s}")
    print(f"  {'-'*28} {'-'*9} {'-'*10} {'-'*9} {'-'*5} {'-'*5} {'-'*7}")

    viable_sta_lta = []
    for r in sta_lta_results:
        reduction = (1 - r['n_computations'] / N_STEPS) * 100
        if reduction >= 80 and r['mean_cos'] >= quality_floor:
            lat1 = r['detection_latency_1']
            lat2 = r['detection_latency_2']
            caught_1 = lat1 is not None and lat1 <= 5
            caught_2 = lat2 is not None and lat2 <= 5
            label = (f"STA={r['sta_window']} LTA={r['lta_window']} "
                     f"trig={r['trigger_ratio']:.1f}")
            lat1_s = f"{lat1}" if lat1 is not None else "MISS"
            lat2_s = f"{lat2}" if lat2 is not None else "MISS"
            marker = " <<" if caught_1 and caught_2 else ""
            print(f"  {label:<28s} {r['n_computations']:>9d} "
                  f"{reduction:>9.1f}% {r['mean_cos']:>9.6f} "
                  f"{lat1_s:>5s} {lat2_s:>5s} "
                  f"{r['total_time']:>6.2f}s{marker}")
            if caught_1 and caught_2:
                viable_sta_lta.append(r)

    print(f"\n  Viable configs (>80% reduction, caught both within 5 steps): "
          f"{len(viable_sta_lta)}")

    # Also show fixed interval viability
    print(f"\n  Fixed interval comparison:")
    for r in fixed_results:
        reduction = (1 - r['n_computations'] / N_STEPS) * 100
        meets_quality = r['mean_cos'] >= quality_floor
        print(f"  Interval={r['interval']:2d}: "
              f"reduction={reduction:.1f}%  cos={r['mean_cos']:.6f}  "
              f"lat=[{r['detection_latency_1']}, {r['detection_latency_2']}]  "
              f"{'PASS' if meets_quality else 'FAIL'}")

    # Best STA/LTA config
    if viable_sta_lta:
        best = min(viable_sta_lta, key=lambda r: r['n_computations'])
        best_reduction = (1 - best['n_computations'] / N_STEPS) * 100
        print(f"\n  BEST STA/LTA CONFIG:")
        print(f"    STA={best['sta_window']}, LTA={best['lta_window']}, "
              f"trigger={best['trigger_ratio']:.1f}")
        print(f"    Entropy computations: {best['n_computations']}/{N_STEPS} "
              f"({best_reduction:.1f}% reduction)")
        print(f"    Mean cos sim: {best['mean_cos']:.6f}")
        print(f"    Transition 1 latency: {best['detection_latency_1']} steps")
        print(f"    Transition 2 latency: {best['detection_latency_2']} steps")
        print(f"    Time: {best['total_time']:.2f}s vs "
              f"{result_always['total_time']:.2f}s "
              f"({result_always['total_time']/best['total_time']:.2f}x)")

    # ===================================================================
    # Quality over time: show cos sim around transitions
    # ===================================================================
    print("\n" + "=" * 78)
    print("  QUALITY AROUND TRANSITIONS")
    print("=" * 78)

    if viable_sta_lta:
        best = min(viable_sta_lta, key=lambda r: r['n_computations'])
    elif sta_lta_results:
        # Pick the STA/LTA with best reduction that still has decent quality
        best = min(
            [r for r in sta_lta_results
             if r['mean_cos'] >= quality_floor],
            key=lambda r: r['n_computations'],
            default=sta_lta_results[0]
        )

    best_fixed = min(fixed_results, key=lambda r: r['n_computations'])

    print(f"\n  {'Step':>5s} {'Always':>10s} {'STA/LTA':>10s} "
          f"{'Fixed-{}'.format(best_fixed['interval']):>10s}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10}")

    check_windows = list(range(145, 165)) + list(range(345, 365))
    for step in check_windows:
        if step >= len(result_always['cos_sims']):
            continue
        a_cos = result_always['cos_sims'][step]
        b_cos = best['cos_sims'][step]
        c_cos = best_fixed['cos_sims'][step]
        marker = " <-- TRANSITION" if step in (TRANSITION_1, TRANSITION_2) else ""
        print(f"  {step:5d} {a_cos:10.6f} {b_cos:10.6f} {c_cos:10.6f}{marker}")

    # ===================================================================
    # Save results
    # ===================================================================
    md_path = os.path.join(os.path.dirname(__file__), "EXP6_RESULTS.md")
    save_results(md_path, result_always, sta_lta_results, fixed_results,
                 viable_sta_lta)

    print(f"\n  Results saved to: {md_path}")
    print(f"\n{'='*78}")
    print("  EXPERIMENT 6 COMPLETE")
    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def save_results(md_path, result_always, sta_lta_results, fixed_results,
                 viable_sta_lta):
    """Write experiment results to markdown."""
    with open(md_path, "w") as f:
        f.write("# Experiment 6: STA/LTA Change-Point Detection "
                "for Entropy Amortization\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} "
                f"(GQA {REP}:1), D={D}, bits={BITS}  \n")
        f.write(f"**Context:** L_kv starts at {L_KV_START}, "
                f"{N_STEPS} decode steps  \n")
        f.write(f"**Transitions:** step {TRANSITION_1} (concentrated -> mixed), "
                f"step {TRANSITION_2} (mixed -> concentrated)  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        # Hypothesis
        f.write("## Hypothesis\n\n")
        f.write("Attention patterns don't change drastically between consecutive "
                "decode tokens -- they drift slowly. We can compute entropy once, "
                "cache the per-head thresholds, and only recompute when STA/LTA "
                "(Short-Term Average / Long-Term Average) ratio detects a "
                "significant shift. Between shifts, reuse cached thresholds.\n\n")
        f.write("**Key question:** Can we reduce entropy computations by >80% "
                "while catching transitions within 5 steps?\n\n")

        # Baseline
        f.write("## Baseline: Always Recompute\n\n")
        f.write(f"- Entropy computations: {result_always['n_computations']}"
                f"/{N_STEPS}\n")
        f.write(f"- Mean cosine similarity: {result_always['mean_cos']:.6f}\n")
        f.write(f"- Min cosine similarity: {result_always['min_cos']:.6f}\n")
        f.write(f"- Total time: {result_always['total_time']:.2f}s\n\n")

        # STA/LTA parameter sweep
        f.write("## STA/LTA Parameter Sweep\n\n")
        f.write("| STA | LTA | Trigger | Computes | Reduction | "
                "Mean Cos | Lat1 | Lat2 | Time |\n")
        f.write("|----:|----:|--------:|---------:|----------:|"
                "--------:|-----:|-----:|-----:|\n")

        for r in sta_lta_results:
            reduction = (1 - r['n_computations'] / N_STEPS) * 100
            lat1 = r['detection_latency_1']
            lat2 = r['detection_latency_2']
            lat1_s = str(lat1) if lat1 is not None else "MISS"
            lat2_s = str(lat2) if lat2 is not None else "MISS"
            viable = " **" if (reduction >= 80
                               and r['mean_cos'] >= result_always['mean_cos'] - 0.001
                               and lat1 is not None and lat1 <= 5
                               and lat2 is not None and lat2 <= 5) else ""
            f.write(f"| {r['sta_window']} | {r['lta_window']} | "
                    f"{r['trigger_ratio']:.1f} | "
                    f"{r['n_computations']} | {reduction:.1f}% | "
                    f"{r['mean_cos']:.6f} | {lat1_s} | {lat2_s} | "
                    f"{r['total_time']:.2f}s |{viable}\n")

        # Fixed interval results
        f.write("\n## Fixed Interval Comparison\n\n")
        f.write("| Interval | Computes | Reduction | Mean Cos | "
                "Lat1 | Lat2 | Time |\n")
        f.write("|---------:|---------:|----------:|---------:|"
                "-----:|-----:|-----:|\n")

        for r in fixed_results:
            reduction = (1 - r['n_computations'] / N_STEPS) * 100
            f.write(f"| {r['interval']} | {r['n_computations']} | "
                    f"{reduction:.1f}% | {r['mean_cos']:.6f} | "
                    f"{r['detection_latency_1']} | "
                    f"{r['detection_latency_2']} | "
                    f"{r['total_time']:.2f}s |\n")

        # Key comparison table
        f.write("\n## Strategy Comparison Summary\n\n")
        f.write("| Strategy | Computes | Reduction | Mean Cos | "
                "Catches transitions? | Time |\n")
        f.write("|:---------|:--------:|:---------:|:--------:|"
                ":-------------------:|:----:|\n")

        f.write(f"| Always recompute | {result_always['n_computations']} | "
                f"0% | {result_always['mean_cos']:.6f} | N/A | "
                f"{result_always['total_time']:.2f}s |\n")

        if viable_sta_lta:
            best = min(viable_sta_lta, key=lambda r: r['n_computations'])
            best_reduction = (1 - best['n_computations'] / N_STEPS) * 100
            f.write(f"| **STA/LTA (best)** STA={best['sta_window']} "
                    f"LTA={best['lta_window']} t={best['trigger_ratio']:.1f} | "
                    f"{best['n_computations']} | {best_reduction:.1f}% | "
                    f"{best['mean_cos']:.6f} | "
                    f"Yes ({best['detection_latency_1']},"
                    f"{best['detection_latency_2']} steps) | "
                    f"{best['total_time']:.2f}s |\n")

        # Always include best overall STA/LTA even if it didn't meet all criteria
        quality_floor = result_always['mean_cos'] - 0.001
        all_quality = [r for r in sta_lta_results
                       if r['mean_cos'] >= quality_floor]
        if all_quality:
            most_reduced = min(all_quality, key=lambda r: r['n_computations'])
            if not viable_sta_lta or most_reduced != viable_sta_lta[0]:
                mr_reduction = (1 - most_reduced['n_computations'] / N_STEPS) * 100
                lat1 = most_reduced['detection_latency_1']
                lat2 = most_reduced['detection_latency_2']
                lat1_s = str(lat1) if lat1 is not None else "MISS"
                lat2_s = str(lat2) if lat2 is not None else "MISS"
                f.write(f"| STA/LTA (most reduced) "
                        f"STA={most_reduced['sta_window']} "
                        f"LTA={most_reduced['lta_window']} "
                        f"t={most_reduced['trigger_ratio']:.1f} | "
                        f"{most_reduced['n_computations']} | "
                        f"{mr_reduction:.1f}% | "
                        f"{most_reduced['mean_cos']:.6f} | "
                        f"({lat1_s},{lat2_s}) | "
                        f"{most_reduced['total_time']:.2f}s |\n")

        for r in fixed_results:
            reduction = (1 - r['n_computations'] / N_STEPS) * 100
            f.write(f"| Fixed interval={r['interval']} | "
                    f"{r['n_computations']} | {reduction:.1f}% | "
                    f"{r['mean_cos']:.6f} | "
                    f"Blind ({r['detection_latency_1']},"
                    f"{r['detection_latency_2']} steps) | "
                    f"{r['total_time']:.2f}s |\n")

        # Transition quality detail
        f.write("\n## Quality Around Transitions\n\n")
        f.write("Cosine similarity vs FP16 at steps around each transition point.\n\n")

        if viable_sta_lta:
            best_sl = min(viable_sta_lta, key=lambda r: r['n_computations'])
        elif all_quality:
            best_sl = min(all_quality, key=lambda r: r['n_computations'])
        else:
            best_sl = sta_lta_results[0] if sta_lta_results else None

        best_fi = min(fixed_results, key=lambda r: r['n_computations'])

        if best_sl:
            f.write(f"Best STA/LTA: STA={best_sl['sta_window']} "
                    f"LTA={best_sl['lta_window']} "
                    f"trigger={best_sl['trigger_ratio']:.1f}  \n")
        f.write(f"Best fixed: interval={best_fi['interval']}\n\n")

        f.write("### Transition 1 (step 150: concentrated -> mixed)\n\n")
        f.write("| Step | Always | STA/LTA | Fixed |\n")
        f.write("|-----:|-------:|--------:|------:|\n")
        for step in range(145, 165):
            if step >= N_STEPS:
                break
            a = result_always['cos_sims'][step]
            b = best_sl['cos_sims'][step] if best_sl else 0
            c = best_fi['cos_sims'][step]
            marker = " **" if step == TRANSITION_1 else ""
            f.write(f"| {step} | {a:.6f} | {b:.6f} | {c:.6f} |{marker}\n")

        f.write("\n### Transition 2 (step 350: mixed -> concentrated)\n\n")
        f.write("| Step | Always | STA/LTA | Fixed |\n")
        f.write("|-----:|-------:|--------:|------:|\n")
        for step in range(345, 365):
            if step >= N_STEPS:
                break
            a = result_always['cos_sims'][step]
            b = best_sl['cos_sims'][step] if best_sl else 0
            c = best_fi['cos_sims'][step]
            marker = " **" if step == TRANSITION_2 else ""
            f.write(f"| {step} | {a:.6f} | {b:.6f} | {c:.6f} |{marker}\n")

        # Analysis
        f.write("\n## Analysis\n\n")

        n_viable = len(viable_sta_lta)
        if n_viable > 0:
            f.write(f"**{n_viable} STA/LTA configurations** met all criteria "
                    f"(>80% reduction, quality within 0.001, both transitions "
                    f"caught within 5 steps).\n\n")
            best = min(viable_sta_lta, key=lambda r: r['n_computations'])
            best_red = (1 - best['n_computations'] / N_STEPS) * 100
            f.write(f"Best config achieves **{best_red:.1f}% reduction** in "
                    f"entropy computations ({best['n_computations']}/{N_STEPS} "
                    f"computes) while maintaining cos sim of "
                    f"{best['mean_cos']:.6f}.\n\n")
        else:
            f.write("**No STA/LTA configuration** met all three criteria "
                    "simultaneously. See analysis below for why.\n\n")

        # Compare STA/LTA vs fixed
        f.write("### STA/LTA vs Fixed Interval\n\n")
        if all_quality:
            best_sl_red = min(all_quality,
                              key=lambda r: r['n_computations'])
            sl_red = (1 - best_sl_red['n_computations'] / N_STEPS) * 100
            fi_red = (1 - best_fi['n_computations'] / N_STEPS) * 100
            if sl_red > fi_red:
                f.write(f"STA/LTA achieves higher reduction ({sl_red:.1f}%) "
                        f"than the best fixed interval ({fi_red:.1f}%), and "
                        f"STA/LTA is adaptive -- it fires specifically when the "
                        f"signal changes rather than on a blind schedule.\n\n")
            else:
                f.write(f"Fixed interval achieves competitive reduction "
                        f"({fi_red:.1f}%) compared to STA/LTA ({sl_red:.1f}%). "
                        f"The simpler approach may be preferable given the "
                        f"overhead of maintaining STA/LTA history.\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")

        if viable_sta_lta:
            best = min(viable_sta_lta, key=lambda r: r['n_computations'])
            best_red = (1 - best['n_computations'] / N_STEPS) * 100
            speedup = result_always['total_time'] / best['total_time']

            f.write("### Verdict: POSITIVE\n\n")
            f.write(f"STA/LTA change-point detection successfully amortizes "
                    f"entropy computation:\n\n")
            f.write(f"- **{best_red:.0f}% fewer entropy computations** "
                    f"({best['n_computations']} vs {N_STEPS})\n")
            f.write(f"- **Quality preserved:** mean cos sim "
                    f"{best['mean_cos']:.6f} vs baseline "
                    f"{result_always['mean_cos']:.6f}\n")
            f.write(f"- **Transitions detected** within "
                    f"{max(best['detection_latency_1'], best['detection_latency_2'])}"
                    f" steps\n")
            f.write(f"- **Wall-clock:** {speedup:.2f}x "
                    f"({'faster' if speedup > 1 else 'slower'})\n\n")
            f.write(f"Recommended config: STA={best['sta_window']}, "
                    f"LTA={best['lta_window']}, "
                    f"trigger={best['trigger_ratio']:.1f}\n\n")
            f.write("### Integration Path\n\n")
            f.write("1. Add `STALTADetector` to `TurboQuantKVCache.__init__`\n")
            f.write("2. In `fused_sdpa()`, before `_compute_adaptive_threshold()`:"
                    "\n")
            f.write("   - Compute cheap stat: "
                    "`mx.max(weights, axis=-1).mean().item()`\n")
            f.write("   - Feed to detector: `detector.update(stat)`\n")
            f.write("   - If no trigger: reuse `self._cached_thresholds`\n")
            f.write("   - If trigger: recompute and cache\n")
            f.write("3. Cost: one `mx.max()` per step (~0.01ms) vs full entropy "
                    "(~0.1-1ms)\n")
        else:
            # Check if any configs got close
            best_overall = None
            if all_quality:
                best_overall = min(all_quality,
                                   key=lambda r: r['n_computations'])
                best_red = (1 - best_overall['n_computations'] / N_STEPS) * 100

            if best_overall and best_red >= 50:
                f.write("### Verdict: PARTIAL\n\n")
                f.write(f"STA/LTA achieves {best_red:.0f}% reduction but "
                        f"does not meet the >80% target with transition "
                        f"detection within 5 steps. ")
                f.write(f"The fixed interval approach at interval="
                        f"{best_fi['interval']} provides a simpler "
                        f"alternative.\n\n")
                f.write("Consider:\n")
                f.write("- Using the cheap stat directly as a threshold gate "
                        "(no STA/LTA windowing)\n")
                f.write("- Combining STA/LTA with a minimum recompute interval "
                        "for guaranteed freshness\n")
                f.write("- Using a different tracked statistic (e.g., entropy "
                        "itself, with a cheap approximation)\n")
            else:
                f.write("### Verdict: NEGATIVE\n\n")
                f.write("STA/LTA change-point detection does not provide "
                        "sufficient reduction while maintaining quality and "
                        "transition responsiveness. The attention statistics "
                        "may not be amenable to this approach, or the signal "
                        "is too noisy for ratio-based detection.\n\n")
                f.write("Fixed interval recomputation (every 10-25 steps) "
                        "is the simpler, safer alternative.\n")

        f.write("\n---\n")
        f.write(f"*Generated by exp6_sta_lta.py on "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")


if __name__ == "__main__":
    main()
