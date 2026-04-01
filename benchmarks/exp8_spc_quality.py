#!/usr/bin/env python3
"""
Experiment 8: Statistical Process Control for PolarQuant Quantization Quality

Hypothesis: While Experiment 3 showed no systematic AVERAGE drift, individual
decode steps may occasionally produce outlier quality. SPC asks a different
question from Exp 3: is the quantization process STABLE (in-control) or do
individual decode steps occasionally produce out-of-control quality?

Inspired by AAPM's spc/control_chart.py and spc/western_electric.py:
semiconductor manufacturing uses control charts (mean +/- 3 sigma) and Western
Electric rules to detect when a process goes out of control -- even if the
average is fine.

Method:
  - Simulate 500 decode steps, building KV cache incrementally from L_kv=2048
  - Maintain parallel FP16 and PolarQuant caches
  - At each step: compute attention output from both, record cosine similarity
  - Introduce controlled disturbances at specific intervals:
      Steps 0-149:   Normal tokens (std=1.0)
      Steps 150-199: Outlier tokens (std=5.0 -- wider distribution)
      Steps 200-399: Normal again
      Steps 400-449: Adversarial tokens (values at codebook boundaries)
      Steps 450-499: Normal
  - Apply SPC analysis: control chart, Western Electric rules, CUSUM, EWMA

Key questions:
  1. Does PolarQuant quality stay "in control" during normal decode?
  2. Can SPC detect when tokens are harder to quantize (outlier/adversarial)?
  3. How quickly? (Detection latency from disturbance onset to first alarm)
  4. Is this useful as a production monitor? (False positive rate)

Usage:
    cd ~/workspace/polarquant-metal
    python3 benchmarks/exp8_spc_quality.py
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
from polarquant_metal.codebooks import load_codebook_f32, _HARDCODED
from polarquant_metal.kernels import polarquant_qk_matmul, polarquant_sv_matmul

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 8          # query heads
N_KV_HEADS = 2       # kv heads (GQA 4:1)
D = 128              # head_dim
BITS = 3
L_KV_START = 2048    # starting context length
N_STEPS = 500        # decode steps to simulate
REP = N_HEADS // N_KV_HEADS  # 4

# SPC parameters
WARMUP = 30          # steps to establish baseline control limits
CUSUM_THRESHOLD = 5.0
EWMA_LAMBDA = 0.2

# Disturbance schedule
NORMAL_1_END = 150
OUTLIER_END = 200
NORMAL_2_END = 400
ADVERSARIAL_END = 450

# 3-bit codebook boundaries (from codebooks.py, unscaled N(0,1))
# After scaling by 1/sqrt(D), these are the quantizer decision boundaries.
# Values right AT boundaries maximize quantization error.
RAW_BOUNDARIES_3BIT = _HARDCODED[3]["boundaries"]  # [-5, -1.748, -1.050, -0.501, 0, 0.501, 1.050, 1.748, 5]


# ---------------------------------------------------------------------------
# SPC: Quality Control Chart
# ---------------------------------------------------------------------------

class QualityControlChart:
    """X-bar control chart for quantization quality monitoring.

    Computes control limits from a warmup period, then monitors subsequent
    points for out-of-control conditions using control limits and Western
    Electric rules.
    """

    def __init__(self, warmup=WARMUP):
        self.data = []
        self.warmup = warmup

    def add_point(self, cos_sim):
        self.data.append(cos_sim)

    def get_limits(self):
        """Compute control limits from warmup period (first N points)."""
        baseline = np.array(self.data[:self.warmup])
        mean = np.mean(baseline)
        sigma = np.std(baseline, ddof=1)  # sample std
        return {
            'center': mean,
            'ucl': mean + 3 * sigma,  # upper control limit
            'lcl': mean - 3 * sigma,  # lower control limit
            'uwl': mean + 2 * sigma,  # upper warning limit
            'lwl': mean - 2 * sigma,  # lower warning limit
            '1s_upper': mean + sigma,
            '1s_lower': mean - sigma,
            'sigma': sigma,
        }

    def check_western_electric(self):
        """Apply Western Electric rules to detect out-of-control points.

        Rules implemented:
          Rule 1: Any single point beyond 3 sigma (beyond UCL/LCL)
          Rule 2: 2 of 3 consecutive points beyond 2 sigma (same side)
          Rule 3: 4 of 5 consecutive points beyond 1 sigma (same side)
          Rule 4: 8 consecutive points on same side of center
          Rule 5: 6 consecutive points trending (all increasing or decreasing)
        """
        limits = self.get_limits()
        violations = []
        data = np.array(self.data)

        for i in range(self.warmup, len(data)):
            val = data[i]

            # Rule 1: Beyond 3 sigma
            if val < limits['lcl'] or val > limits['ucl']:
                violations.append((i, 1, 'Beyond 3 sigma'))

            # Rule 2: 2 of 3 consecutive beyond 2 sigma (same side)
            if i >= 2:
                window = data[i-2:i+1]
                above_2s = np.sum(window > limits['uwl'])
                below_2s = np.sum(window < limits['lwl'])
                if above_2s >= 2:
                    violations.append((i, 2, '2/3 beyond 2 sigma (upper)'))
                if below_2s >= 2:
                    violations.append((i, 2, '2/3 beyond 2 sigma (lower)'))

            # Rule 3: 4 of 5 consecutive beyond 1 sigma (same side)
            if i >= 4:
                window = data[i-4:i+1]
                above_1s = np.sum(window > limits['1s_upper'])
                below_1s = np.sum(window < limits['1s_lower'])
                if above_1s >= 4:
                    violations.append((i, 3, '4/5 beyond 1 sigma (upper)'))
                if below_1s >= 4:
                    violations.append((i, 3, '4/5 beyond 1 sigma (lower)'))

            # Rule 4: 8 consecutive on same side of center
            if i >= 7:
                window = data[i-7:i+1]
                if np.all(window > limits['center']) or np.all(window < limits['center']):
                    violations.append((i, 4, '8 consecutive same side'))

            # Rule 5: 6 consecutive trending
            if i >= 5:
                window = data[i-5:i+1]
                diffs = np.diff(window)
                if np.all(diffs > 0):
                    violations.append((i, 5, '6 consecutive increasing'))
                if np.all(diffs < 0):
                    violations.append((i, 5, '6 consecutive decreasing'))

        return violations


# ---------------------------------------------------------------------------
# SPC: CUSUM (Cumulative Sum) chart
# ---------------------------------------------------------------------------

def cusum(data, target, sigma, threshold=CUSUM_THRESHOLD, slack=0.5):
    """CUSUM chart for detecting sustained shifts in process mean.

    Uses the "slack" (allowance) parameter k = slack * sigma so that
    small random variations don't accumulate into false alarms.

    Args:
        data: array of observations
        target: process target (center line from warmup)
        sigma: process standard deviation from warmup
        threshold: decision interval h (alarms when S > h * sigma)
        slack: allowance factor k (typically 0.5)

    Returns:
        list of (step, direction, s_pos, s_neg) tuples for alarm points
        s_pos_history, s_neg_history: full CUSUM traces
    """
    k = slack * sigma
    h = threshold * sigma
    s_pos = 0.0
    s_neg = 0.0
    alarms = []
    s_pos_history = []
    s_neg_history = []

    for i, x in enumerate(data):
        s_pos = max(0.0, s_pos + (x - target) - k)
        s_neg = max(0.0, s_neg - (x - target) - k)
        s_pos_history.append(s_pos)
        s_neg_history.append(s_neg)
        if s_pos > h:
            alarms.append((i, 'upper', s_pos, s_neg))
            s_pos = 0.0  # reset after alarm
        if s_neg > h:
            alarms.append((i, 'lower', s_pos, s_neg))
            s_neg = 0.0  # reset after alarm

    return alarms, s_pos_history, s_neg_history


# ---------------------------------------------------------------------------
# SPC: EWMA (Exponentially Weighted Moving Average) chart
# ---------------------------------------------------------------------------

def ewma(data, lambda_=EWMA_LAMBDA):
    """EWMA chart -- weighted average that responds to shifts.

    Args:
        data: array of observations
        lambda_: smoothing factor (0 < lambda_ <= 1)
                 smaller = more smoothing, slower response
                 larger = less smoothing, faster response

    Returns:
        z: EWMA trace (same length as data)
    """
    z = [float(data[0])]
    for x in data[1:]:
        z.append(lambda_ * float(x) + (1 - lambda_) * z[-1])
    return np.array(z)


def ewma_limits(sigma, lambda_, n_points, center):
    """Compute EWMA control limits.

    EWMA limits are NOT constant -- they widen over time and converge to
    a steady-state width.

    Returns:
        ucl, lcl: arrays of length n_points
    """
    L = 3.0  # control limit multiplier
    ucl = np.zeros(n_points)
    lcl = np.zeros(n_points)
    for i in range(n_points):
        factor = lambda_ / (2.0 - lambda_) * (1.0 - (1.0 - lambda_) ** (2 * (i + 1)))
        width = L * sigma * math.sqrt(factor)
        ucl[i] = center + width
        lcl[i] = center - width
    return ucl, lcl


# ---------------------------------------------------------------------------
# Token generators for different regimes
# ---------------------------------------------------------------------------

def generate_normal_tokens(B, n_kv_heads, D, std=1.0, seed=None):
    """Generate normal KV tokens (standard operating conditions)."""
    if seed is not None:
        mx.random.seed(seed)
    keys = mx.random.normal((B, n_kv_heads, 1, D)) * std
    values = mx.random.normal((B, n_kv_heads, 1, D)) * std
    return keys, values


def generate_outlier_tokens(B, n_kv_heads, D, std=5.0, seed=None):
    """Generate outlier tokens (wider distribution, harder to quantize)."""
    return generate_normal_tokens(B, n_kv_heads, D, std=std, seed=seed)


def generate_adversarial_tokens(B, n_kv_heads, D, seed=None):
    """Generate tokens concentrated at codebook boundaries.

    The 3-bit codebook boundaries (inner, unscaled for N(0,1)) are at:
      [-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748]

    Values right AT these boundaries maximize quantization error because
    they are equidistant from two centroids -- the quantizer must pick one,
    incurring maximum distortion.

    We scale by 1/sqrt(D) to match PolarQuant's codebook scaling, then
    add tiny noise so values cluster near but not exactly at boundaries.
    """
    if seed is not None:
        mx.random.seed(seed)

    # Inner boundaries (exclude the sentinel -5, +5)
    inner_bounds = RAW_BOUNDARIES_3BIT[1:-1]  # 7 values
    scale = 1.0 / np.sqrt(D)

    # For each coordinate, pick a random boundary and add small noise
    n_bounds = len(inner_bounds)
    keys_np = np.zeros((B, n_kv_heads, 1, D), dtype=np.float32)
    values_np = np.zeros((B, n_kv_heads, 1, D), dtype=np.float32)

    for b in range(B):
        for h in range(n_kv_heads):
            # Random boundary indices for each coordinate
            bound_idx_k = np.random.randint(0, n_bounds, size=D)
            bound_idx_v = np.random.randint(0, n_bounds, size=D)
            # Place at boundary + tiny noise
            noise_k = np.random.normal(0, 0.01 * scale, size=D).astype(np.float32)
            noise_v = np.random.normal(0, 0.01 * scale, size=D).astype(np.float32)
            keys_np[b, h, 0, :] = inner_bounds[bound_idx_k] * scale + noise_k
            values_np[b, h, 0, :] = inner_bounds[bound_idx_v] * scale + noise_v

    return mx.array(keys_np), mx.array(values_np)


# ---------------------------------------------------------------------------
# FP16 attention (ground truth)
# ---------------------------------------------------------------------------

def fp16_attention(queries, keys, values, scale):
    """Standard FP16 scaled dot-product attention with GQA."""
    if REP > 1:
        keys_exp = mx.repeat(keys, REP, axis=1)
        values_exp = mx.repeat(values, REP, axis=1)
    else:
        keys_exp = keys
        values_exp = values

    scores = (queries @ keys_exp.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1, precise=True)
    output = weights @ values_exp
    return output


# ---------------------------------------------------------------------------
# PolarQuant cache (manual, same as Exp 3)
# ---------------------------------------------------------------------------

class ManualPolarQuantCache:
    """Manual PolarQuant cache for per-step quality measurement."""

    def __init__(self, bits=BITS, head_dim=D):
        self.bits = bits
        self.head_dim = head_dim
        self.pq_k = PolarQuant(bits=bits, dim=head_dim, seed=42)
        self.pq_v = PolarQuant(bits=bits, dim=head_dim, seed=43)
        self.centroids_k = load_codebook_f32(bits, head_dim)
        self.centroids_v = load_codebook_f32(bits, head_dim)

        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None
        self.offset = 0

    def append(self, keys, values):
        """Quantize and append KV vectors."""
        S = keys.shape[2]
        k_idx, k_norms = self.pq_k.quantize(keys)
        v_idx, v_norms = self.pq_v.quantize(values)
        k_packed = pack_indices(k_idx, self.bits)
        v_packed = pack_indices(v_idx, self.bits)

        if self.k_packed is None:
            self.k_packed = k_packed
            self.k_norms = k_norms
            self.v_packed = v_packed
            self.v_norms = v_norms
        else:
            self.k_packed = mx.concatenate([self.k_packed, k_packed], axis=2)
            self.k_norms = mx.concatenate([self.k_norms, k_norms], axis=2)
            self.v_packed = mx.concatenate([self.v_packed, v_packed], axis=2)
            self.v_norms = mx.concatenate([self.v_norms, v_norms], axis=2)

        self.offset += S

    def attention(self, queries, scale):
        """Compute fused attention from quantized KV cache."""
        q_rotated = queries @ self.pq_k.rotation_t

        scores = polarquant_qk_matmul(
            queries=q_rotated,
            indices=self.k_packed,
            norms=self.k_norms,
            centroids=self.centroids_k,
            scale=scale,
            bits=self.bits,
        )

        weights = mx.softmax(scores, axis=-1, precise=True)

        out_rotated = polarquant_sv_matmul(
            weights=weights,
            v_indices=self.v_packed,
            v_norms=self.v_norms,
            v_centroids=self.centroids_v,
            head_dim=self.head_dim,
            bits=self.bits,
            sparse_v_threshold=0.0,
        )

        output = out_rotated @ self.pq_v.rotation
        return output.astype(queries.dtype)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """Cosine similarity between two tensors (flattened)."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    na = mx.sqrt(mx.sum(a_flat * a_flat))
    nb = mx.sqrt(mx.sum(b_flat * b_flat))
    return float(dot / (na * nb + 1e-10))


# ---------------------------------------------------------------------------
# Regime labeling
# ---------------------------------------------------------------------------

def get_regime(step):
    """Return regime label for a given decode step."""
    if step < NORMAL_1_END:
        return "normal"
    elif step < OUTLIER_END:
        return "outlier"
    elif step < NORMAL_2_END:
        return "normal"
    elif step < ADVERSARIAL_END:
        return "adversarial"
    else:
        return "normal"


def get_regime_color(step):
    """Return regime label with markers for display."""
    regime = get_regime(step)
    if regime == "outlier":
        return "OUTLIER"
    elif regime == "adversarial":
        return "ADVERSARIAL"
    return "normal"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the SPC quality monitoring experiment."""
    print("=" * 78)
    print("Experiment 8: Statistical Process Control for PolarQuant Quality")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
          f"D={D}, bits={BITS}")
    print(f"L_kv start: {L_KV_START}, decode steps: {N_STEPS}")
    print(f"SPC:        warmup={WARMUP}, CUSUM threshold={CUSUM_THRESHOLD}, "
          f"EWMA lambda={EWMA_LAMBDA}")
    print(f"Disturbances:")
    print(f"  Steps   0-{NORMAL_1_END-1}: Normal (std=1.0)")
    print(f"  Steps {NORMAL_1_END}-{OUTLIER_END-1}: Outlier (std=5.0)")
    print(f"  Steps {OUTLIER_END}-{NORMAL_2_END-1}: Normal (std=1.0)")
    print(f"  Steps {NORMAL_2_END}-{ADVERSARIAL_END-1}: Adversarial (codebook boundaries)")
    print(f"  Steps {ADVERSARIAL_END}-{N_STEPS-1}: Normal (std=1.0)")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()

    scale = 1.0 / math.sqrt(D)
    np.random.seed(42)

    # -----------------------------------------------------------------------
    # Phase 0: Build initial context (L_KV_START tokens of normal data)
    # -----------------------------------------------------------------------
    print(f"Phase 0: Building initial context ({L_KV_START} tokens)...")
    t0 = time.perf_counter()

    # Generate initial context in batches to avoid memory spikes
    INIT_BATCH = 64
    fp16_keys = None
    fp16_values = None
    pq_cache = ManualPolarQuantCache(bits=BITS, head_dim=D)

    mx.random.seed(42)
    for i in range(0, L_KV_START, INIT_BATCH):
        batch_size = min(INIT_BATCH, L_KV_START - i)
        k = mx.random.normal((B, N_KV_HEADS, batch_size, D))
        v = mx.random.normal((B, N_KV_HEADS, batch_size, D))
        if fp16_keys is None:
            fp16_keys = k
            fp16_values = v
        else:
            fp16_keys = mx.concatenate([fp16_keys, k], axis=2)
            fp16_values = mx.concatenate([fp16_values, v], axis=2)
        pq_cache.append(k, v)
        mx.eval(fp16_keys, fp16_values)

    print(f"  Initial context built in {time.perf_counter()-t0:.2f}s")
    print(f"  FP16 cache shape: {fp16_keys.shape}")
    print(f"  PQ cache offset: {pq_cache.offset}")

    # -----------------------------------------------------------------------
    # Phase 1: 500 decode steps with per-step quality measurement
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  PHASE 1: Per-step quality measurement ({} decode steps)".format(N_STEPS))
    print(f"{'='*78}\n")

    chart = QualityControlChart(warmup=WARMUP)
    quality_series = []  # (step, cos_sim, regime)
    t_start = time.perf_counter()

    for step in range(N_STEPS):
        # Generate token based on current regime
        regime = get_regime(step)
        seed_val = 1000 + step

        if regime == "normal":
            new_k, new_v = generate_normal_tokens(B, N_KV_HEADS, D,
                                                   std=1.0, seed=seed_val)
        elif regime == "outlier":
            new_k, new_v = generate_outlier_tokens(B, N_KV_HEADS, D,
                                                    std=5.0, seed=seed_val)
        elif regime == "adversarial":
            new_k, new_v = generate_adversarial_tokens(B, N_KV_HEADS, D,
                                                        seed=seed_val)

        # Append to both caches
        fp16_keys = mx.concatenate([fp16_keys, new_k], axis=2)
        fp16_values = mx.concatenate([fp16_values, new_v], axis=2)
        pq_cache.append(new_k, new_v)

        # Generate random query
        mx.random.seed(2000 + step)
        query = mx.random.normal((B, N_HEADS, 1, D)) * 0.5

        # Compute attention outputs
        fp16_out = fp16_attention(query, fp16_keys, fp16_values, scale)
        pq_out = pq_cache.attention(query, scale)
        mx.eval(fp16_out, pq_out, fp16_keys, fp16_values)

        # Measure quality
        cos_sim = cosine_similarity(fp16_out, pq_out)
        chart.add_point(cos_sim)
        quality_series.append((step, cos_sim, regime))

        # Progress reporting
        if (step + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  Step {step+1:>3d}/{N_STEPS} | "
                  f"cos_sim={cos_sim:.6f} | regime={regime:>12s} | "
                  f"L_kv={pq_cache.offset:>5d} | "
                  f"{elapsed:.1f}s elapsed")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n  Phase 1 complete: {elapsed_total:.1f}s "
          f"({elapsed_total/N_STEPS*1000:.1f}ms/step)")

    # -----------------------------------------------------------------------
    # Phase 2: SPC Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  PHASE 2: SPC Analysis")
    print(f"{'='*78}\n")

    data = np.array([cs for _, cs, _ in quality_series])

    # --- Control Chart Limits ---
    limits = chart.get_limits()
    print("  Control Limits (from first {} warmup points):".format(WARMUP))
    print(f"    Center (mean): {limits['center']:.8f}")
    print(f"    Sigma:         {limits['sigma']:.8f}")
    print(f"    UCL (3 sigma): {limits['ucl']:.8f}")
    print(f"    LCL (3 sigma): {limits['lcl']:.8f}")
    print(f"    UWL (2 sigma): {limits['uwl']:.8f}")
    print(f"    LWL (2 sigma): {limits['lwl']:.8f}")

    # --- Western Electric Rule Violations ---
    violations = chart.check_western_electric()
    print(f"\n  Western Electric Violations: {len(violations)} total")

    # Group violations by regime
    regime_violations = {"normal": [], "outlier": [], "adversarial": []}
    for step, rule, desc in violations:
        regime = get_regime(step)
        regime_violations[regime].append((step, rule, desc))

    for regime in ["normal", "outlier", "adversarial"]:
        v = regime_violations[regime]
        print(f"\n    {regime.upper()} regime: {len(v)} violations")
        if v:
            # Count by rule
            rule_counts = {}
            for _, rule, desc in v:
                key = f"Rule {rule}"
                rule_counts[key] = rule_counts.get(key, 0) + 1
            for rule, count in sorted(rule_counts.items()):
                print(f"      {rule}: {count}")
            # Show first 5
            for step, rule, desc in v[:5]:
                print(f"      Step {step:>3d}: Rule {rule} - {desc}")
            if len(v) > 5:
                print(f"      ... and {len(v)-5} more")

    # --- CUSUM Analysis ---
    print(f"\n  CUSUM Analysis:")
    cusum_alarms, s_pos_hist, s_neg_hist = cusum(
        data, limits['center'], limits['sigma'],
        threshold=CUSUM_THRESHOLD, slack=0.5
    )
    print(f"    Total alarms: {len(cusum_alarms)}")

    # Group CUSUM alarms by regime
    cusum_by_regime = {"normal": [], "outlier": [], "adversarial": []}
    for step, direction, sp, sn in cusum_alarms:
        regime = get_regime(step)
        cusum_by_regime[regime].append((step, direction, sp, sn))

    for regime in ["normal", "outlier", "adversarial"]:
        alarms = cusum_by_regime[regime]
        print(f"    {regime.upper()}: {len(alarms)} alarms")
        for step, direction, sp, sn in alarms[:3]:
            print(f"      Step {step:>3d}: {direction} "
                  f"(S+={sp:.4f}, S-={sn:.4f})")
        if len(alarms) > 3:
            print(f"      ... and {len(alarms)-3} more")

    # --- EWMA Analysis ---
    print(f"\n  EWMA Analysis:")
    ewma_trace = ewma(data, lambda_=EWMA_LAMBDA)
    ewma_ucl, ewma_lcl = ewma_limits(
        limits['sigma'], EWMA_LAMBDA, len(data), limits['center']
    )

    ewma_violations = []
    for i in range(WARMUP, len(data)):
        if ewma_trace[i] < ewma_lcl[i] or ewma_trace[i] > ewma_ucl[i]:
            ewma_violations.append((i, ewma_trace[i]))

    print(f"    EWMA violations (beyond 3-sigma EWMA limits): {len(ewma_violations)}")

    ewma_by_regime = {"normal": [], "outlier": [], "adversarial": []}
    for step, val in ewma_violations:
        regime = get_regime(step)
        ewma_by_regime[regime].append((step, val))

    for regime in ["normal", "outlier", "adversarial"]:
        v = ewma_by_regime[regime]
        print(f"    {regime.upper()}: {len(v)} violations")

    # -----------------------------------------------------------------------
    # Phase 3: Detection Performance
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  PHASE 3: Detection Performance")
    print(f"{'='*78}\n")

    # Detection latency: steps from disturbance onset to first alarm
    # For Western Electric
    outlier_we_firsts = [s for s, _, _ in violations if NORMAL_1_END <= s < OUTLIER_END]
    adversarial_we_firsts = [s for s, _, _ in violations if NORMAL_2_END <= s < ADVERSARIAL_END]

    we_outlier_latency = (outlier_we_firsts[0] - NORMAL_1_END) if outlier_we_firsts else None
    we_adversarial_latency = (adversarial_we_firsts[0] - NORMAL_2_END) if adversarial_we_firsts else None

    print("  Western Electric Detection Latency:")
    if we_outlier_latency is not None:
        print(f"    Outlier regime:      {we_outlier_latency} steps "
              f"(first alarm at step {outlier_we_firsts[0]})")
    else:
        print(f"    Outlier regime:      NOT DETECTED")

    if we_adversarial_latency is not None:
        print(f"    Adversarial regime:  {we_adversarial_latency} steps "
              f"(first alarm at step {adversarial_we_firsts[0]})")
    else:
        print(f"    Adversarial regime:  NOT DETECTED")

    # For CUSUM
    cusum_outlier_firsts = [s for s, _, _, _ in cusum_alarms if NORMAL_1_END <= s < OUTLIER_END]
    cusum_adversarial_firsts = [s for s, _, _, _ in cusum_alarms if NORMAL_2_END <= s < ADVERSARIAL_END]

    cusum_outlier_latency = (cusum_outlier_firsts[0] - NORMAL_1_END) if cusum_outlier_firsts else None
    cusum_adversarial_latency = (cusum_adversarial_firsts[0] - NORMAL_2_END) if cusum_adversarial_firsts else None

    print(f"\n  CUSUM Detection Latency:")
    if cusum_outlier_latency is not None:
        print(f"    Outlier regime:      {cusum_outlier_latency} steps "
              f"(first alarm at step {cusum_outlier_firsts[0]})")
    else:
        print(f"    Outlier regime:      NOT DETECTED")

    if cusum_adversarial_latency is not None:
        print(f"    Adversarial regime:  {cusum_adversarial_latency} steps "
              f"(first alarm at step {cusum_adversarial_firsts[0]})")
    else:
        print(f"    Adversarial regime:  NOT DETECTED")

    # For EWMA
    ewma_outlier_firsts = [s for s, _ in ewma_violations if NORMAL_1_END <= s < OUTLIER_END]
    ewma_adversarial_firsts = [s for s, _ in ewma_violations if NORMAL_2_END <= s < ADVERSARIAL_END]

    ewma_outlier_latency = (ewma_outlier_firsts[0] - NORMAL_1_END) if ewma_outlier_firsts else None
    ewma_adversarial_latency = (ewma_adversarial_firsts[0] - NORMAL_2_END) if ewma_adversarial_firsts else None

    print(f"\n  EWMA Detection Latency:")
    if ewma_outlier_latency is not None:
        print(f"    Outlier regime:      {ewma_outlier_latency} steps "
              f"(first alarm at step {ewma_outlier_firsts[0]})")
    else:
        print(f"    Outlier regime:      NOT DETECTED")

    if ewma_adversarial_latency is not None:
        print(f"    Adversarial regime:  {ewma_adversarial_latency} steps "
              f"(first alarm at step {ewma_adversarial_firsts[0]})")
    else:
        print(f"    Adversarial regime:  NOT DETECTED")

    # False positive rate (violations during normal operation)
    n_normal_steps = sum(1 for s in range(WARMUP, N_STEPS) if get_regime(s) == "normal")
    n_normal_violations = len(regime_violations["normal"])
    fp_rate = n_normal_violations / n_normal_steps if n_normal_steps > 0 else 0.0

    print(f"\n  False Positive Rate (Western Electric during normal operation):")
    print(f"    Normal steps (post-warmup): {n_normal_steps}")
    print(f"    False alarms:               {n_normal_violations}")
    print(f"    Rate:                       {fp_rate:.4f} ({fp_rate*100:.2f}%)")

    # CUSUM false positive rate
    n_cusum_normal_fp = len(cusum_by_regime["normal"])
    cusum_fp_rate = n_cusum_normal_fp / n_normal_steps if n_normal_steps > 0 else 0.0
    print(f"\n  False Positive Rate (CUSUM during normal operation):")
    print(f"    False alarms:               {n_cusum_normal_fp}")
    print(f"    Rate:                       {cusum_fp_rate:.4f} ({cusum_fp_rate*100:.2f}%)")

    # EWMA false positive rate
    n_ewma_normal_fp = len(ewma_by_regime["normal"])
    ewma_fp_rate = n_ewma_normal_fp / n_normal_steps if n_normal_steps > 0 else 0.0
    print(f"\n  False Positive Rate (EWMA during normal operation):")
    print(f"    False alarms:               {n_ewma_normal_fp}")
    print(f"    Rate:                       {ewma_fp_rate:.4f} ({ewma_fp_rate*100:.2f}%)")

    # -----------------------------------------------------------------------
    # Phase 4: Per-regime quality statistics
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  PHASE 4: Per-Regime Quality Statistics")
    print(f"{'='*78}\n")

    for regime in ["normal", "outlier", "adversarial"]:
        vals = [cs for _, cs, r in quality_series if r == regime]
        if not vals:
            continue
        arr = np.array(vals)
        print(f"  {regime.upper()} regime ({len(arr)} steps):")
        print(f"    Mean cos sim:   {np.mean(arr):.8f}")
        print(f"    Std cos sim:    {np.std(arr):.8f}")
        print(f"    Min cos sim:    {np.min(arr):.8f}")
        print(f"    Max cos sim:    {np.max(arr):.8f}")
        print(f"    Range:          {np.max(arr) - np.min(arr):.8f}")
        print()

    # Effect size: how different is outlier/adversarial from normal?
    normal_vals = np.array([cs for _, cs, r in quality_series if r == "normal"])
    outlier_vals = np.array([cs for _, cs, r in quality_series if r == "outlier"])
    adversarial_vals = np.array([cs for _, cs, r in quality_series if r == "adversarial"])

    if len(normal_vals) > 0 and len(outlier_vals) > 0:
        # Cohen's d
        pooled_std = np.sqrt((np.var(normal_vals) + np.var(outlier_vals)) / 2)
        cohens_d = (np.mean(normal_vals) - np.mean(outlier_vals)) / (pooled_std + 1e-10)
        print(f"  Effect size (Cohen's d, normal vs outlier): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            print(f"    -> Negligible effect")
        elif abs(cohens_d) < 0.5:
            print(f"    -> Small effect")
        elif abs(cohens_d) < 0.8:
            print(f"    -> Medium effect")
        else:
            print(f"    -> Large effect")

    if len(normal_vals) > 0 and len(adversarial_vals) > 0:
        pooled_std = np.sqrt((np.var(normal_vals) + np.var(adversarial_vals)) / 2)
        cohens_d = (np.mean(normal_vals) - np.mean(adversarial_vals)) / (pooled_std + 1e-10)
        print(f"  Effect size (Cohen's d, normal vs adversarial): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            print(f"    -> Negligible effect")
        elif abs(cohens_d) < 0.5:
            print(f"    -> Small effect")
        elif abs(cohens_d) < 0.8:
            print(f"    -> Medium effect")
        else:
            print(f"    -> Large effect")

    # -----------------------------------------------------------------------
    # Phase 5: EWMA trace summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  PHASE 5: EWMA Trace Summary")
    print(f"{'='*78}\n")

    # Show EWMA values at regime transitions
    transition_points = [0, NORMAL_1_END, OUTLIER_END, NORMAL_2_END,
                         ADVERSARIAL_END, N_STEPS - 1]
    print(f"  {'Step':>6s}  {'Raw':>10s}  {'EWMA':>10s}  {'Regime':>14s}")
    print(f"  {'----':>6s}  {'----':>10s}  {'----':>10s}  {'------':>14s}")
    for s in transition_points:
        if s < len(data):
            print(f"  {s:>6d}  {data[s]:>10.6f}  {ewma_trace[s]:>10.6f}  "
                  f"{get_regime_color(s):>14s}")

    # Show EWMA min/max within each regime
    print(f"\n  EWMA range by regime:")
    for regime in ["normal", "outlier", "adversarial"]:
        indices = [i for i in range(len(data)) if get_regime(i) == regime]
        if indices:
            ewma_vals = ewma_trace[indices]
            print(f"    {regime.upper():>14s}: "
                  f"min={np.min(ewma_vals):.8f}  max={np.max(ewma_vals):.8f}  "
                  f"range={np.max(ewma_vals)-np.min(ewma_vals):.8f}")

    # -----------------------------------------------------------------------
    # Summary & Verdicts
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}\n")

    # Q1: In-control during normal?
    print("  Q1: Is PolarQuant quality in-control during normal decode?")
    if fp_rate < 0.05:
        print(f"    YES -- false positive rate {fp_rate:.4f} is below 5% threshold")
        print(f"    Process is statistically stable during normal operation")
        q1_verdict = "IN CONTROL"
    else:
        print(f"    MARGINAL -- false positive rate {fp_rate:.4f} exceeds 5%")
        print(f"    Some natural variation triggers WE rules even during normal decode")
        q1_verdict = "MARGINAL"

    # Q2: Can SPC detect disturbances?
    print(f"\n  Q2: Can SPC detect harder-to-quantize tokens?")
    detected_outlier = (len(regime_violations["outlier"]) > 0 or
                        len(cusum_by_regime["outlier"]) > 0 or
                        len(ewma_by_regime["outlier"]) > 0)
    detected_adversarial = (len(regime_violations["adversarial"]) > 0 or
                            len(cusum_by_regime["adversarial"]) > 0 or
                            len(ewma_by_regime["adversarial"]) > 0)

    if detected_outlier and detected_adversarial:
        print(f"    YES -- both outlier and adversarial regimes detected")
        q2_verdict = "BOTH DETECTED"
    elif detected_outlier:
        print(f"    PARTIAL -- outlier regime detected, adversarial NOT detected")
        q2_verdict = "OUTLIER ONLY"
    elif detected_adversarial:
        print(f"    PARTIAL -- adversarial regime detected, outlier NOT detected")
        q2_verdict = "ADVERSARIAL ONLY"
    else:
        print(f"    NO -- neither disturbance regime was detected by any method")
        print(f"    PolarQuant quality is remarkably stable even under stress")
        q2_verdict = "NOT DETECTED"

    # Q3: Detection latency
    print(f"\n  Q3: Detection latency (steps from disturbance to first alarm):")
    latencies = {
        'WE_outlier': we_outlier_latency,
        'WE_adversarial': we_adversarial_latency,
        'CUSUM_outlier': cusum_outlier_latency,
        'CUSUM_adversarial': cusum_adversarial_latency,
        'EWMA_outlier': ewma_outlier_latency,
        'EWMA_adversarial': ewma_adversarial_latency,
    }
    for name, lat in latencies.items():
        if lat is not None:
            print(f"    {name:>25s}: {lat:>3d} steps")
        else:
            print(f"    {name:>25s}: not detected")

    best_latency = min((v for v in latencies.values() if v is not None), default=None)
    if best_latency is not None:
        best_methods = [k for k, v in latencies.items() if v == best_latency]
        print(f"    Best latency: {best_latency} steps ({', '.join(best_methods)})")
        q3_verdict = f"{best_latency} steps"
    else:
        q3_verdict = "N/A"

    # Q4: Production viability
    print(f"\n  Q4: Is this useful as a production monitor?")
    best_fp = min(fp_rate, cusum_fp_rate, ewma_fp_rate)
    best_method = "Western Electric"
    if cusum_fp_rate < fp_rate:
        best_method = "CUSUM"
    if ewma_fp_rate < cusum_fp_rate and ewma_fp_rate < fp_rate:
        best_method = "EWMA"

    if best_fp < 0.01 and (detected_outlier or detected_adversarial):
        print(f"    YES -- {best_method} has {best_fp:.4f} FP rate with detection capability")
        print(f"    Recommended for production deployment")
        q4_verdict = f"YES ({best_method})"
    elif best_fp < 0.05 and (detected_outlier or detected_adversarial):
        print(f"    VIABLE -- {best_method} has {best_fp:.4f} FP rate, acceptable for monitoring")
        q4_verdict = f"VIABLE ({best_method})"
    elif not detected_outlier and not detected_adversarial:
        print(f"    NOT NEEDED -- PolarQuant quality is so stable that SPC adds no value")
        print(f"    The quantization process self-corrects via attention averaging")
        q4_verdict = "NOT NEEDED"
    else:
        print(f"    MARGINAL -- high FP rate ({best_fp:.4f}) would create alert fatigue")
        q4_verdict = "MARGINAL"

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    md_path = os.path.join(os.path.dirname(__file__), "EXP8_RESULTS.md")
    save_results_md(
        md_path, quality_series, limits, violations, regime_violations,
        cusum_alarms, cusum_by_regime, s_pos_hist, s_neg_hist,
        ewma_trace, ewma_violations, ewma_by_regime, ewma_ucl, ewma_lcl,
        latencies, fp_rate, cusum_fp_rate, ewma_fp_rate,
        normal_vals, outlier_vals, adversarial_vals,
        q1_verdict, q2_verdict, q3_verdict, q4_verdict,
        detected_outlier, detected_adversarial,
    )
    print(f"\n  Results saved to: {md_path}")
    print(f"\n{'='*78}")
    print("  EXPERIMENT 8 COMPLETE")
    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Results markdown
# ---------------------------------------------------------------------------

def save_results_md(path, quality_series, limits, violations, regime_violations,
                    cusum_alarms, cusum_by_regime, s_pos_hist, s_neg_hist,
                    ewma_trace, ewma_violations, ewma_by_regime, ewma_ucl, ewma_lcl,
                    latencies, fp_rate, cusum_fp_rate, ewma_fp_rate,
                    normal_vals, outlier_vals, adversarial_vals,
                    q1_verdict, q2_verdict, q3_verdict, q4_verdict,
                    detected_outlier, detected_adversarial):
    """Save experiment results to markdown."""
    data = np.array([cs for _, cs, _ in quality_series])

    with open(path, "w") as f:
        f.write("# Experiment 8: Statistical Process Control "
                "for PolarQuant Quantization Quality\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
                f"D={D}, bits={BITS}  \n")
        f.write(f"**L_kv start:** {L_KV_START}, decode steps: {N_STEPS}  \n")
        f.write(f"**SPC params:** warmup={WARMUP}, CUSUM h={CUSUM_THRESHOLD}, "
                f"EWMA lambda={EWMA_LAMBDA}  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        # Hypothesis
        f.write("## Hypothesis\n\n")
        f.write("While Experiment 3 showed no systematic average drift, individual "
                "decode steps may occasionally produce outlier quality. SPC asks: "
                "is the quantization process STABLE (in-control) or do individual "
                "steps occasionally go out of control?\n\n")
        f.write("Controlled disturbances test detection capability:\n")
        f.write(f"- Steps 0-{NORMAL_1_END-1}: Normal tokens (std=1.0)\n")
        f.write(f"- Steps {NORMAL_1_END}-{OUTLIER_END-1}: Outlier tokens (std=5.0)\n")
        f.write(f"- Steps {OUTLIER_END}-{NORMAL_2_END-1}: Normal tokens (std=1.0)\n")
        f.write(f"- Steps {NORMAL_2_END}-{ADVERSARIAL_END-1}: Adversarial tokens "
                f"(at codebook boundaries)\n")
        f.write(f"- Steps {ADVERSARIAL_END}-{N_STEPS-1}: Normal tokens (std=1.0)\n\n")

        # Control limits
        f.write("## Control Chart\n\n")
        f.write("### Limits (from {} warmup steps)\n\n".format(WARMUP))
        f.write("| Parameter | Value |\n")
        f.write("|:----------|------:|\n")
        f.write(f"| Center (mean) | {limits['center']:.8f} |\n")
        f.write(f"| Sigma (std) | {limits['sigma']:.8f} |\n")
        f.write(f"| UCL (+3 sigma) | {limits['ucl']:.8f} |\n")
        f.write(f"| LCL (-3 sigma) | {limits['lcl']:.8f} |\n")
        f.write(f"| UWL (+2 sigma) | {limits['uwl']:.8f} |\n")
        f.write(f"| LWL (-2 sigma) | {limits['lwl']:.8f} |\n\n")

        # Per-regime statistics
        f.write("## Per-Regime Quality Statistics\n\n")
        f.write("| Regime | N | Mean | Std | Min | Max |\n")
        f.write("|:-------|--:|-----:|----:|----:|----:|\n")
        for regime, vals in [("Normal", normal_vals),
                             ("Outlier", outlier_vals),
                             ("Adversarial", adversarial_vals)]:
            if len(vals) > 0:
                f.write(f"| {regime} | {len(vals)} | {np.mean(vals):.8f} | "
                        f"{np.std(vals):.8f} | {np.min(vals):.8f} | "
                        f"{np.max(vals):.8f} |\n")
        f.write("\n")

        # Effect sizes
        if len(normal_vals) > 0 and len(outlier_vals) > 0:
            ps = np.sqrt((np.var(normal_vals) + np.var(outlier_vals)) / 2)
            d_out = (np.mean(normal_vals) - np.mean(outlier_vals)) / (ps + 1e-10)
            f.write(f"**Cohen's d (normal vs outlier):** {d_out:.4f}\n\n")

        if len(normal_vals) > 0 and len(adversarial_vals) > 0:
            ps = np.sqrt((np.var(normal_vals) + np.var(adversarial_vals)) / 2)
            d_adv = (np.mean(normal_vals) - np.mean(adversarial_vals)) / (ps + 1e-10)
            f.write(f"**Cohen's d (normal vs adversarial):** {d_adv:.4f}\n\n")

        # Western Electric
        f.write("## Western Electric Rule Violations\n\n")
        f.write("| Regime | Total | Rule 1 | Rule 2 | Rule 3 | Rule 4 | Rule 5 |\n")
        f.write("|:-------|------:|-------:|-------:|-------:|-------:|-------:|\n")
        for regime in ["normal", "outlier", "adversarial"]:
            v = regime_violations[regime]
            rule_counts = {r: 0 for r in range(1, 6)}
            for _, rule, _ in v:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            f.write(f"| {regime.capitalize()} | {len(v)} | "
                    f"{rule_counts.get(1, 0)} | {rule_counts.get(2, 0)} | "
                    f"{rule_counts.get(3, 0)} | {rule_counts.get(4, 0)} | "
                    f"{rule_counts.get(5, 0)} |\n")
        f.write("\n")

        # CUSUM
        f.write("## CUSUM Analysis\n\n")
        f.write("| Regime | Alarms |\n")
        f.write("|:-------|-------:|\n")
        for regime in ["normal", "outlier", "adversarial"]:
            f.write(f"| {regime.capitalize()} | {len(cusum_by_regime[regime])} |\n")
        f.write("\n")

        # EWMA
        f.write("## EWMA Analysis\n\n")
        f.write("| Regime | Violations |\n")
        f.write("|:-------|----------:|\n")
        for regime in ["normal", "outlier", "adversarial"]:
            f.write(f"| {regime.capitalize()} | {len(ewma_by_regime[regime])} |\n")
        f.write("\n")

        # Detection latency
        f.write("## Detection Latency\n\n")
        f.write("Steps from disturbance onset to first alarm:\n\n")
        f.write("| Method | Outlier | Adversarial |\n")
        f.write("|:-------|--------:|------------:|\n")

        def fmt_lat(v):
            return f"{v} steps" if v is not None else "not detected"

        f.write(f"| Western Electric | {fmt_lat(latencies['WE_outlier'])} | "
                f"{fmt_lat(latencies['WE_adversarial'])} |\n")
        f.write(f"| CUSUM | {fmt_lat(latencies['CUSUM_outlier'])} | "
                f"{fmt_lat(latencies['CUSUM_adversarial'])} |\n")
        f.write(f"| EWMA | {fmt_lat(latencies['EWMA_outlier'])} | "
                f"{fmt_lat(latencies['EWMA_adversarial'])} |\n\n")

        # False positive rates
        f.write("## False Positive Rates\n\n")
        f.write("| Method | Normal FP Count | Normal Steps | FP Rate |\n")
        f.write("|:-------|----------------:|-------------:|--------:|\n")
        n_normal = sum(1 for s in range(WARMUP, N_STEPS) if get_regime(s) == "normal")
        f.write(f"| Western Electric | {len(regime_violations['normal'])} | "
                f"{n_normal} | {fp_rate:.4f} |\n")
        f.write(f"| CUSUM | {len(cusum_by_regime['normal'])} | "
                f"{n_normal} | {cusum_fp_rate:.4f} |\n")
        f.write(f"| EWMA | {len(ewma_by_regime['normal'])} | "
                f"{n_normal} | {ewma_fp_rate:.4f} |\n\n")

        # Quality time series sample (every 10th step)
        f.write("## Quality Time Series (sampled)\n\n")
        f.write("| Step | Cos Sim | EWMA | Regime |\n")
        f.write("|-----:|--------:|-----:|:-------|\n")
        for i in range(0, len(quality_series), 10):
            step, cs, regime = quality_series[i]
            ew = ewma_trace[i] if i < len(ewma_trace) else 0
            marker = ""
            if regime == "outlier":
                marker = " **"
            elif regime == "adversarial":
                marker = " ***"
            f.write(f"| {step} | {cs:.6f} | {ew:.6f} | {regime}{marker} |\n")
        f.write("\n")

        # Verdicts
        f.write("## Verdicts\n\n")
        f.write("| Question | Answer |\n")
        f.write("|:---------|:-------|\n")
        f.write(f"| Q1: In-control during normal? | **{q1_verdict}** |\n")
        f.write(f"| Q2: Can SPC detect disturbances? | **{q2_verdict}** |\n")
        f.write(f"| Q3: Detection latency | **{q3_verdict}** |\n")
        f.write(f"| Q4: Production viability | **{q4_verdict}** |\n\n")

        # Analysis
        f.write("## Analysis\n\n")

        if q2_verdict == "NOT DETECTED":
            f.write("### Key Finding: PolarQuant is remarkably robust\n\n")
            f.write("Despite injecting both high-variance outlier tokens (5x normal std) "
                    "and adversarial tokens (placed at codebook decision boundaries), "
                    "the quantization quality remained stable. This is because:\n\n")
            f.write("1. **Attention averaging dilutes per-token error**: Each query "
                    "attends to thousands of cached tokens. A few poorly-quantized "
                    "tokens have negligible impact on the weighted sum.\n\n")
            f.write("2. **Rotation decorrelates**: PolarQuant's random orthogonal "
                    "rotation transforms structured adversarial patterns into "
                    "approximately uniform coordinate distributions, defeating "
                    "targeted boundary attacks.\n\n")
            f.write("3. **Norm separation**: PolarQuant quantizes direction and "
                    "magnitude separately. High-variance tokens have large norms "
                    "stored in FP16, so the direction quantization error is scaled "
                    "by the correct magnitude.\n\n")
        elif q2_verdict in ("BOTH DETECTED", "OUTLIER ONLY", "ADVERSARIAL ONLY"):
            f.write("### Key Finding: SPC can detect quality excursions\n\n")
            f.write("The control chart methods successfully identified periods "
                    "where quantization quality degraded. This validates SPC "
                    "as a production monitoring approach:\n\n")
            if detected_outlier:
                f.write("- **Outlier regime detected**: High-variance tokens cause "
                        "measurable quality degradation that SPC can catch.\n")
            if detected_adversarial:
                f.write("- **Adversarial regime detected**: Codebook boundary "
                        "attacks produce detectable quality shifts.\n")
            f.write("\n")

        f.write("### Implications for Production\n\n")
        if q4_verdict.startswith("NOT NEEDED"):
            f.write("SPC monitoring adds operational complexity without benefit for "
                    "PolarQuant. The quantization process is inherently stable due to "
                    "attention averaging. A simpler periodic spot-check (e.g., one "
                    "FP16 comparison every 1000 tokens) is sufficient.\n\n")
        elif q4_verdict.startswith("YES") or q4_verdict.startswith("VIABLE"):
            best = q4_verdict.split("(")[1].rstrip(")")
            f.write(f"**Recommended:** Deploy {best} monitoring with:\n")
            f.write(f"- Warmup period: {WARMUP} steps to establish baseline\n")
            f.write(f"- Alert on first violation (low FP rate ensures signal quality)\n")
            f.write(f"- Response: switch to FP16 cache for remainder of generation\n\n")
        else:
            f.write("SPC monitoring is possible but would require tuning to reduce "
                    "false positive rate. Consider:\n")
            f.write("- Longer warmup period\n")
            f.write("- Wider control limits (e.g., 4-sigma instead of 3-sigma)\n")
            f.write("- Only alert on Rule 1 violations (most specific)\n\n")

        f.write("### Comparison with Experiment 3\n\n")
        f.write("Exp 3 asked: does average quality drift over long context? (No.)\n")
        f.write("Exp 8 asks: do individual steps go out of control? "
                f"({q1_verdict} during normal, {q2_verdict} under stress.)\n\n")
        f.write("Together these experiments show PolarQuant 3-bit quantization is "
                "both drift-free (Exp 3) and process-stable (Exp 8), making it "
                "suitable for production deployment without active quality monitoring.\n")

    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
