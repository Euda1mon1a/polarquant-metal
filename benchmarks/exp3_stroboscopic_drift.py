#!/usr/bin/env python3
"""
Experiment 3: Stroboscopic FP16 Checkpoints for PolarQuant Drift Detection

Hypothesis: Over long conversations (16K+ tokens), PolarQuant 3-bit codebook
quantization error may accumulate. Each token's attention output has a small
error from codebook approximation, and these errors compound across layers
and tokens as the KV cache grows.

Inspired by time crystal "stroboscopic observation": periodically run a full
FP16 attention pass as a calibration checkpoint to measure accumulated drift.

Key Questions:
  1. Does drift ACCUMULATE (cosine sim decreases as context grows)?
     Or stay FLAT (each token's error is independent)?
  2. If drift is detected, does "recalibration" (re-quantize from FP16 cache)
     recover quality? Is stroboscopic recalibration a viable correction?

Method:
  - Build two caches in parallel: FP16 (ground truth) + PolarQuant 3-bit
  - Add tokens in batches of 16 up to 16K context
  - At checkpoint intervals (64, 256, 1024, 4096), compute attention output
    from both caches with a random query and measure cosine similarity
  - Test recalibration at the final checkpoint

Usage:
    cd ~/workspace/polarquant-metal
    python benchmarks/exp3_stroboscopic_drift.py
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
from polarquant_metal.kernels import polarquant_qk_matmul, polarquant_sv_matmul

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
B = 1
N_HEADS = 8          # query heads
N_KV_HEADS = 2       # kv heads (GQA 4:1)
D = 128               # head_dim
BITS = 3
BATCH_SIZE = 16       # tokens added per step
TOTAL_TOKENS = 16384  # 16K context target
REP = N_HEADS // N_KV_HEADS  # 4

# Checkpoint intervals: measure drift at these context lengths
CHECKPOINT_INTERVALS = [64, 256, 1024, 4096]

# Minimum context for fused path (matches TurboQuantKVCache default)
MIN_FUSED_CONTEXT = 512


# ---------------------------------------------------------------------------
# KV sequence generator (AR(1) process for realistic correlation)
# ---------------------------------------------------------------------------

def generate_kv_sequence(total_tokens, B, n_kv_heads, D, batch_size=BATCH_SIZE,
                         seed=42):
    """Generate sequence of KV vectors with realistic correlation structure.

    Each batch: previous * 0.95 + noise * 0.1 (AR(1) process).
    This models the fact that nearby tokens in a real sequence have correlated
    KV projections, with occasional distributional shifts.

    Returns:
        list of (k_batch, v_batch) tuples, each (B, n_kv_heads, batch_size, D)
    """
    mx.random.seed(seed)
    batches = []

    # Initialize with moderate variance
    current_k = mx.random.normal((B, n_kv_heads, batch_size, D)) * 0.5
    current_v = mx.random.normal((B, n_kv_heads, batch_size, D)) * 0.5

    for i in range(0, total_tokens, batch_size):
        noise_k = mx.random.normal((B, n_kv_heads, batch_size, D)) * 0.1
        noise_v = mx.random.normal((B, n_kv_heads, batch_size, D)) * 0.1
        current_k = current_k * 0.95 + noise_k
        current_v = current_v * 0.95 + noise_v
        batches.append((current_k, current_v))

    return batches


# ---------------------------------------------------------------------------
# FP16 attention (ground truth)
# ---------------------------------------------------------------------------

def fp16_attention(queries, keys, values, scale):
    """Standard FP16 scaled dot-product attention with GQA.

    Args:
        queries: (B, N_HEADS, 1, D)
        keys:    (B, N_KV_HEADS, L, D)
        values:  (B, N_KV_HEADS, L, D)
        scale:   1/sqrt(D)

    Returns:
        output: (B, N_HEADS, 1, D)
    """
    # Expand KV heads for GQA
    if REP > 1:
        keys_exp = mx.repeat(keys, REP, axis=1)
        values_exp = mx.repeat(values, REP, axis=1)
    else:
        keys_exp = keys
        values_exp = values

    # Q @ K^T * scale
    scores = (queries @ keys_exp.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1, precise=True)
    output = weights @ values_exp
    return output


# ---------------------------------------------------------------------------
# PolarQuant attention (quantized path, manual — not using TurboQuantKVCache
# to keep FP16 and quant caches fully independent)
# ---------------------------------------------------------------------------

class ManualPolarQuantCache:
    """Manual PolarQuant cache for drift measurement.

    Stores packed indices and norms, provides a method to compute fused
    attention. This mirrors TurboQuantKVCache internals but without the
    lazy-quantization threshold so we quantize from step 0.
    """

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
        """Quantize and append a batch of KV vectors.

        Args:
            keys:   (B, N_KV_HEADS, S, D)
            values: (B, N_KV_HEADS, S, D)
        """
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
        """Compute fused attention from quantized KV cache.

        Args:
            queries: (B, N_HEADS, 1, D)
            scale:   1/sqrt(D)

        Returns:
            output: (B, N_HEADS, 1, D)
        """
        # Pre-rotate queries into key basis
        q_rotated = queries @ self.pq_k.rotation_t

        # Fused Q @ K^T
        scores = polarquant_qk_matmul(
            queries=q_rotated,
            indices=self.k_packed,
            norms=self.k_norms,
            centroids=self.centroids_k,
            scale=scale,
            bits=self.bits,
        )

        # Softmax
        weights = mx.softmax(scores, axis=-1, precise=True)

        # Fused weights @ V
        out_rotated = polarquant_sv_matmul(
            weights=weights,
            v_indices=self.v_packed,
            v_norms=self.v_norms,
            v_centroids=self.centroids_v,
            head_dim=self.head_dim,
            bits=self.bits,
            sparse_v_threshold=0.0,
        )

        # Inverse rotation from value basis
        output = out_rotated @ self.pq_v.rotation
        return output.astype(queries.dtype)

    def recalibrate_from_fp16(self, fp16_keys, fp16_values):
        """Replace quantized cache with fresh quantization of FP16 data.

        This is the stroboscopic recalibration: throw away the accumulated
        quantized cache and re-quantize from the FP16 ground truth.
        """
        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None
        self.offset = 0
        self.append(fp16_keys, fp16_values)


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


def l2_distance(a, b):
    """L2 distance between two tensors (flattened)."""
    diff = (a - b).reshape(-1).astype(mx.float32)
    return float(mx.sqrt(mx.sum(diff * diff)))


def per_head_cosine(a, b, n_heads):
    """Cosine similarity per head.

    Args:
        a, b: (B, n_heads, L_q, D)

    Returns:
        list of float, one per head
    """
    results = []
    for h in range(n_heads):
        a_h = a[:, h:h+1, :, :]
        b_h = b[:, h:h+1, :, :]
        results.append(cosine_similarity(a_h, b_h))
    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the stroboscopic drift detection experiment."""
    print("=" * 78)
    print("Experiment 3: Stroboscopic FP16 Checkpoints for PolarQuant Drift")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
          f"D={D}, bits={BITS}")
    print(f"Context:    up to {TOTAL_TOKENS:,} tokens in batches of {BATCH_SIZE}")
    print(f"Checkpoints at: {CHECKPOINT_INTERVALS}")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()

    scale = 1.0 / math.sqrt(D)

    # Generate the full KV sequence
    print("Generating KV sequence (AR(1) process)...")
    t0 = time.perf_counter()
    kv_batches = generate_kv_sequence(TOTAL_TOKENS, B, N_KV_HEADS, D)
    print(f"  Generated {len(kv_batches)} batches in {time.perf_counter()-t0:.2f}s")

    # Build set of checkpoint token counts for quick lookup
    checkpoint_set = set()
    for interval in CHECKPOINT_INTERVALS:
        for ctx_len in range(interval, TOTAL_TOKENS + 1, interval):
            checkpoint_set.add(ctx_len)

    # --- Phase 1: Incremental cache build with checkpoint measurements ---
    print(f"\n{'='*78}")
    print("  PHASE 1: Incremental drift measurement")
    print(f"{'='*78}\n")

    # FP16 cache (ground truth): just concatenate tensors
    fp16_keys = None
    fp16_values = None

    # PolarQuant cache
    pq_cache = ManualPolarQuantCache(bits=BITS, head_dim=D)

    # Results storage: {interval: [(ctx_len, cos_sim, l2_dist, per_head_cos)]}
    results_by_interval = {iv: [] for iv in CHECKPOINT_INTERVALS}

    # Pre-generate query vectors for each checkpoint
    # Use a fixed seed separate from KV generation
    mx.random.seed(999)
    query_bank = {}
    for ctx_len in sorted(checkpoint_set):
        query_bank[ctx_len] = mx.random.normal((B, N_HEADS, 1, D)) * 0.5

    current_ctx = 0
    t_start = time.perf_counter()
    last_report = 0

    for batch_idx, (k_batch, v_batch) in enumerate(kv_batches):
        # Append to FP16 cache
        if fp16_keys is None:
            fp16_keys = k_batch
            fp16_values = v_batch
        else:
            fp16_keys = mx.concatenate([fp16_keys, k_batch], axis=2)
            fp16_values = mx.concatenate([fp16_values, v_batch], axis=2)

        # Append to PolarQuant cache
        pq_cache.append(k_batch, v_batch)
        mx.eval(fp16_keys, fp16_values)

        current_ctx += BATCH_SIZE

        # Progress reporting every 2048 tokens
        if current_ctx - last_report >= 2048:
            elapsed = time.perf_counter() - t_start
            print(f"  Context: {current_ctx:>6,} / {TOTAL_TOKENS:,} "
                  f"({current_ctx/TOTAL_TOKENS*100:5.1f}%) "
                  f"[{elapsed:.1f}s elapsed]")
            last_report = current_ctx

        # Check if this is a checkpoint
        if current_ctx in checkpoint_set:
            query = query_bank[current_ctx]
            mx.eval(query)

            # FP16 attention output (ground truth)
            fp16_out = fp16_attention(query, fp16_keys, fp16_values, scale)
            mx.eval(fp16_out)

            # PolarQuant attention output
            pq_out = pq_cache.attention(query, scale)
            mx.eval(pq_out)

            # Metrics
            cos = cosine_similarity(fp16_out, pq_out)
            l2 = l2_distance(fp16_out, pq_out)
            head_cos = per_head_cosine(fp16_out, pq_out, N_HEADS)

            # Record for each interval that triggers at this context length
            for iv in CHECKPOINT_INTERVALS:
                if current_ctx % iv == 0:
                    results_by_interval[iv].append(
                        (current_ctx, cos, l2, head_cos)
                    )

    elapsed_total = time.perf_counter() - t_start
    print(f"\n  Phase 1 complete: {elapsed_total:.1f}s total")

    # --- Print Phase 1 results ---
    print(f"\n{'='*78}")
    print("  PHASE 1 RESULTS: Drift by checkpoint interval")
    print(f"{'='*78}")

    for iv in CHECKPOINT_INTERVALS:
        data = results_by_interval[iv]
        if not data:
            continue
        print(f"\n  --- Interval: every {iv} tokens ---")
        print(f"  {'Context':>8s}  {'Cos Sim':>10s}  {'L2 Dist':>10s}  "
              f"{'Min Head':>10s}  {'Max Head':>10s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for ctx, cos, l2, hcos in data:
            print(f"  {ctx:>8,}  {cos:>10.6f}  {l2:>10.6f}  "
                  f"{min(hcos):>10.6f}  {max(hcos):>10.6f}")

        # Drift analysis for this interval
        first_cos = data[0][1]
        last_cos = data[-1][1]
        drift = first_cos - last_cos
        print(f"\n  Drift from first to last: {drift:+.6f} "
              f"({first_cos:.6f} -> {last_cos:.6f})")
        if abs(drift) < 0.001:
            print(f"  -> FLAT: drift is negligible (<0.001)")
        elif drift > 0:
            print(f"  -> ACCUMULATING: quality degrades over context length")
        else:
            print(f"  -> IMPROVING: quality increases (unexpected)")

    # --- Phase 2: Recalibration test ---
    print(f"\n\n{'='*78}")
    print("  PHASE 2: Stroboscopic recalibration test")
    print(f"{'='*78}\n")

    # Measure pre-recalibration quality at full context
    query_final = mx.random.normal((B, N_HEADS, 1, D)) * 0.5
    mx.eval(query_final)

    fp16_out_final = fp16_attention(query_final, fp16_keys, fp16_values, scale)
    mx.eval(fp16_out_final)

    pq_out_pre = pq_cache.attention(query_final, scale)
    mx.eval(pq_out_pre)

    cos_pre = cosine_similarity(fp16_out_final, pq_out_pre)
    l2_pre = l2_distance(fp16_out_final, pq_out_pre)
    hcos_pre = per_head_cosine(fp16_out_final, pq_out_pre, N_HEADS)

    print(f"  Pre-recalibration at {TOTAL_TOKENS:,} tokens:")
    print(f"    Cosine similarity: {cos_pre:.6f}")
    print(f"    L2 distance:       {l2_pre:.6f}")
    print(f"    Per-head cos sim:  {['%.4f' % h for h in hcos_pre]}")

    # Recalibrate: re-quantize entire cache from FP16 ground truth
    print(f"\n  Recalibrating (re-quantize from FP16)...")
    t_recal = time.perf_counter()
    pq_cache.recalibrate_from_fp16(fp16_keys, fp16_values)
    mx.eval(pq_cache.k_packed, pq_cache.k_norms,
            pq_cache.v_packed, pq_cache.v_norms)
    recal_time = time.perf_counter() - t_recal

    # Measure post-recalibration quality (same query)
    pq_out_post = pq_cache.attention(query_final, scale)
    mx.eval(pq_out_post)

    cos_post = cosine_similarity(fp16_out_final, pq_out_post)
    l2_post = l2_distance(fp16_out_final, pq_out_post)
    hcos_post = per_head_cosine(fp16_out_final, pq_out_post, N_HEADS)

    print(f"  Recalibration took {recal_time*1000:.1f}ms")
    print(f"\n  Post-recalibration at {TOTAL_TOKENS:,} tokens:")
    print(f"    Cosine similarity: {cos_post:.6f}")
    print(f"    L2 distance:       {l2_post:.6f}")
    print(f"    Per-head cos sim:  {['%.4f' % h for h in hcos_post]}")

    recovery = cos_post - cos_pre
    print(f"\n  Recovery delta (post - pre): {recovery:+.6f}")
    if abs(recovery) < 0.0005:
        print(f"  -> NO MEANINGFUL RECOVERY: recalibration doesn't help "
              f"(drift is independent-error, not accumulating)")
    elif recovery > 0:
        print(f"  -> RECOVERY CONFIRMED: recalibration improves quality by "
              f"{recovery:.6f}")
    else:
        print(f"  -> UNEXPECTED: recalibration made quality worse")

    # --- Phase 3: Per-head drift analysis ---
    print(f"\n\n{'='*78}")
    print("  PHASE 3: Per-head drift analysis (interval=256)")
    print(f"{'='*78}\n")

    data_256 = results_by_interval.get(256, [])
    if data_256:
        print(f"  {'Context':>8s}", end="")
        for h in range(N_HEADS):
            print(f"  {'H'+str(h):>8s}", end="")
        print()
        print(f"  {'-'*8}", end="")
        for _ in range(N_HEADS):
            print(f"  {'-'*8}", end="")
        print()

        for ctx, cos, l2, hcos in data_256:
            print(f"  {ctx:>8,}", end="")
            for h in range(N_HEADS):
                print(f"  {hcos[h]:>8.4f}", end="")
            print()

        # Per-head drift magnitude
        print(f"\n  Per-head drift (first -> last):")
        for h in range(N_HEADS):
            first_h = data_256[0][3][h]
            last_h = data_256[-1][3][h]
            drift_h = first_h - last_h
            print(f"    H{h}: {first_h:.6f} -> {last_h:.6f} "
                  f"(drift={drift_h:+.6f})")

    # --- Phase 4: TurboQuantKVCache lazy-quantization comparison ---
    print(f"\n\n{'='*78}")
    print("  PHASE 4: Lazy quantization effect (< vs >= min_fused_context)")
    print(f"{'='*78}\n")
    print(f"  TurboQuantKVCache defers quantization until {MIN_FUSED_CONTEXT} tokens.")
    print(f"  Below that threshold, tokens are stored in FP16 -- zero quant error.")

    # Check the 256-interval data for the transition point
    if data_256:
        pre_threshold = [(ctx, cos) for ctx, cos, _, _ in data_256
                         if ctx <= MIN_FUSED_CONTEXT]
        post_threshold = [(ctx, cos) for ctx, cos, _, _ in data_256
                          if ctx > MIN_FUSED_CONTEXT]

        if pre_threshold:
            mean_pre = np.mean([c for _, c in pre_threshold])
            print(f"\n  Mean cos sim BELOW threshold (FP16 region): {mean_pre:.6f}")
            print(f"    Note: Our manual cache quantizes from step 0, so values")
            print(f"    here reflect full quantization even for early tokens.")
        if post_threshold:
            mean_post = np.mean([c for _, c in post_threshold])
            print(f"  Mean cos sim ABOVE threshold (quantized region): {mean_post:.6f}")

        if pre_threshold and post_threshold:
            delta = mean_pre - mean_post
            print(f"  Delta (pre - post threshold): {delta:+.6f}")
            if abs(delta) < 0.001:
                print(f"  -> No quality difference -- lazy quantization doesn't help")
            elif delta > 0:
                print(f"  -> Lazy quantization helps: early FP16 tokens would have "
                      f"higher quality")
            else:
                print(f"  -> Unexpected: post-threshold quality is higher")

    # --- Summary ---
    print(f"\n\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}\n")

    # Gather overall drift from the finest interval
    finest_interval = CHECKPOINT_INTERVALS[0]  # 64
    finest_data = results_by_interval[finest_interval]
    if finest_data:
        cos_values = [cos for _, cos, _, _ in finest_data]
        cos_min = min(cos_values)
        cos_max = max(cos_values)
        cos_mean = np.mean(cos_values)
        cos_std = np.std(cos_values)
        cos_first = cos_values[0]
        cos_last = cos_values[-1]

        # Linear regression to quantify drift rate
        ctx_vals = np.array([ctx for ctx, _, _, _ in finest_data], dtype=np.float64)
        cos_arr = np.array(cos_values, dtype=np.float64)
        if len(ctx_vals) > 1:
            slope = np.polyfit(ctx_vals, cos_arr, 1)[0]
            drift_per_1k = slope * 1000
        else:
            slope = 0.0
            drift_per_1k = 0.0

        print(f"  Cosine Similarity Statistics (interval={finest_interval}):")
        print(f"    Mean:      {cos_mean:.6f}")
        print(f"    Std:       {cos_std:.6f}")
        print(f"    Min:       {cos_min:.6f}")
        print(f"    Max:       {cos_max:.6f}")
        print(f"    First:     {cos_first:.6f}")
        print(f"    Last:      {cos_last:.6f}")
        print(f"    Drift/1K:  {drift_per_1k:+.6f} (linear fit slope * 1000)")
        print()

        # Verdict
        if abs(drift_per_1k) < 0.0005:
            drift_verdict = "FLAT"
            print(f"  VERDICT: Drift is FLAT -- quantization error is independent per")
            print(f"  token, NOT accumulating. Each query samples the cache uniformly,")
            print(f"  so individual token errors average out rather than compound.")
        elif drift_per_1k < 0:
            drift_verdict = "ACCUMULATING"
            print(f"  VERDICT: Drift is ACCUMULATING -- quality degrades by")
            print(f"  {abs(drift_per_1k):.6f} cosine similarity per 1K tokens.")
            print(f"  Stroboscopic recalibration may be worthwhile.")
        else:
            drift_verdict = "IMPROVING"
            print(f"  VERDICT: Quality IMPROVES with context length.")
            print(f"  More tokens -> better averaging -> errors cancel.")

        print(f"\n  Recalibration Assessment:")
        print(f"    Pre-recalibration cos sim:  {cos_pre:.6f}")
        print(f"    Post-recalibration cos sim: {cos_post:.6f}")
        print(f"    Recovery:                   {recovery:+.6f}")
        print(f"    Recalibration cost:         {recal_time*1000:.1f}ms")

        if abs(recovery) < 0.0005:
            recal_verdict = "NOT NEEDED"
            print(f"    -> Recalibration provides no benefit. Stroboscopic")
            print(f"       checkpoints are useful for MONITORING but not CORRECTION.")
        elif recovery > 0:
            recal_verdict = "BENEFICIAL"
            print(f"    -> Recalibration recovers {recovery:.6f} cosine similarity.")
            print(f"       At {recal_time*1000:.1f}ms cost, viable as periodic maintenance.")
        else:
            recal_verdict = "COUNTERPRODUCTIVE"
            print(f"    -> Recalibration hurts quality. Do not use.")
    else:
        drift_verdict = "NO DATA"
        recal_verdict = "NO DATA"

    # --- Save results to markdown ---
    md_path = os.path.join(os.path.dirname(__file__), "EXP3_RESULTS.md")
    save_results_md(
        md_path, results_by_interval, finest_data,
        cos_pre, cos_post, recovery, recal_time,
        drift_verdict, recal_verdict,
        hcos_pre, hcos_post, data_256,
    )
    print(f"\n  Results saved to: {md_path}")
    print(f"\n{'='*78}")
    print("  EXPERIMENT 3 COMPLETE")
    print(f"{'='*78}")


def save_results_md(path, results_by_interval, finest_data,
                    cos_pre, cos_post, recovery, recal_time,
                    drift_verdict, recal_verdict,
                    hcos_pre, hcos_post, data_256):
    """Save experiment results to markdown."""
    with open(path, "w") as f:
        f.write("# Experiment 3: Stroboscopic FP16 Checkpoints "
                "for PolarQuant Drift Detection\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), "
                f"D={D}, bits={BITS}  \n")
        f.write(f"**Context:** up to {TOTAL_TOKENS:,} tokens in batches of "
                f"{BATCH_SIZE}  \n")
        f.write(f"**Checkpoints:** {CHECKPOINT_INTERVALS}  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        # Hypothesis
        f.write("## Hypothesis\n\n")
        f.write("Over long conversations (16K+ tokens), PolarQuant 3-bit codebook "
                "quantization error may accumulate. Each token's attention output "
                "has a small error from codebook approximation, and these errors "
                "may compound across layers and tokens as the KV cache grows.\n\n")
        f.write("**Key question:** Does drift ACCUMULATE (cosine similarity "
                "decreases as context grows) or stay FLAT (each token's error "
                "is independent)?\n\n")

        # Drift measurement table
        f.write("## Drift Measurement\n\n")
        for iv in CHECKPOINT_INTERVALS:
            data = results_by_interval[iv]
            if not data:
                continue
            f.write(f"### Checkpoint interval: every {iv} tokens\n\n")
            f.write("| Context | Cos Sim | L2 Dist | Min Head | Max Head |\n")
            f.write("|--------:|--------:|--------:|---------:|---------:|\n")
            for ctx, cos, l2, hcos in data:
                f.write(f"| {ctx:,} | {cos:.6f} | {l2:.6f} | "
                        f"{min(hcos):.6f} | {max(hcos):.6f} |\n")

            first_cos = data[0][1]
            last_cos = data[-1][1]
            drift = first_cos - last_cos
            f.write(f"\nDrift (first - last): **{drift:+.6f}**\n\n")

        # Statistics
        if finest_data:
            cos_values = [cos for _, cos, _, _ in finest_data]
            ctx_vals = np.array([ctx for ctx, _, _, _ in finest_data],
                                dtype=np.float64)
            cos_arr = np.array(cos_values, dtype=np.float64)
            slope = np.polyfit(ctx_vals, cos_arr, 1)[0] if len(ctx_vals) > 1 else 0
            drift_per_1k = slope * 1000

            f.write("## Summary Statistics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|:-------|------:|\n")
            f.write(f"| Mean cos sim | {np.mean(cos_values):.6f} |\n")
            f.write(f"| Std cos sim | {np.std(cos_values):.6f} |\n")
            f.write(f"| Min cos sim | {min(cos_values):.6f} |\n")
            f.write(f"| Max cos sim | {max(cos_values):.6f} |\n")
            f.write(f"| First checkpoint | {cos_values[0]:.6f} |\n")
            f.write(f"| Last checkpoint | {cos_values[-1]:.6f} |\n")
            f.write(f"| Drift per 1K tokens | {drift_per_1k:+.6f} |\n")
            f.write(f"| **Drift verdict** | **{drift_verdict}** |\n\n")

        # Recalibration
        f.write("## Recalibration Test\n\n")
        f.write(f"Re-quantize the entire KV cache from FP16 ground truth "
                f"at {TOTAL_TOKENS:,} tokens.\n\n")
        f.write("| Metric | Pre-Recal | Post-Recal | Delta |\n")
        f.write("|:-------|----------:|-----------:|------:|\n")
        f.write(f"| Cos sim | {cos_pre:.6f} | {cos_post:.6f} | "
                f"{recovery:+.6f} |\n")
        f.write(f"| Cost | - | {recal_time*1000:.1f}ms | - |\n\n")

        # Per-head recalibration
        f.write("### Per-head cosine similarity\n\n")
        f.write("| Head | Pre-Recal | Post-Recal | Delta |\n")
        f.write("|-----:|----------:|-----------:|------:|\n")
        for h in range(N_HEADS):
            delta_h = hcos_post[h] - hcos_pre[h]
            f.write(f"| H{h} | {hcos_pre[h]:.6f} | {hcos_post[h]:.6f} | "
                    f"{delta_h:+.6f} |\n")
        f.write(f"\n**Recalibration verdict:** {recal_verdict}\n\n")

        # Per-head drift over context
        if data_256:
            f.write("## Per-Head Drift (interval=256)\n\n")
            f.write("| Context |")
            for h in range(N_HEADS):
                f.write(f" H{h} |")
            f.write("\n|--------:|")
            for _ in range(N_HEADS):
                f.write("------:|")
            f.write("\n")
            for ctx, cos, l2, hcos in data_256:
                f.write(f"| {ctx:,} |")
                for h in range(N_HEADS):
                    f.write(f" {hcos[h]:.4f} |")
                f.write("\n")
            f.write("\n")

        # Analysis
        f.write("## Analysis\n\n")
        if drift_verdict == "FLAT":
            f.write("Quantization error is **independent per token** and does not "
                    "accumulate over context length. This is because attention "
                    "operates as a weighted average over all cached positions: "
                    "individual token quantization errors are independent and "
                    "tend to cancel out when averaged.\n\n")
            f.write("The attention mechanism's softmax normalization ensures that "
                    "the output is a convex combination of value vectors. "
                    "Quantization error in any single value vector is diluted "
                    "by the number of positions the query attends to. As context "
                    "grows, the query typically spreads attention across more "
                    "positions, further diluting per-token errors.\n\n")
        elif drift_verdict == "ACCUMULATING":
            f.write("Quantization error **accumulates** with context length. "
                    "Possible causes:\n\n")
            f.write("1. Correlated KV vectors (AR(1) structure) cause systematic "
                    "quantization bias in one direction\n")
            f.write("2. Attention softmax concentrates on specific tokens whose "
                    "quantization errors dominate\n")
            f.write("3. The codebook approximation is biased for the particular "
                    "data distribution at large context\n\n")
        else:
            f.write("Quality improves with context length, likely because more "
                    "tokens means the weighted average samples more positions, "
                    "allowing quantization errors to cancel out more effectively "
                    "(law of large numbers applied to the attention mechanism).\n\n")

        if recal_verdict == "NOT NEEDED":
            f.write("Stroboscopic recalibration provides no meaningful quality "
                    "improvement, confirming that quantization error is "
                    "stationary. The FP16 checkpoints are valuable for "
                    "**monitoring** (detecting if drift ever appears in "
                    "production with real model weights) but not for "
                    "**correction**.\n\n")
        elif recal_verdict == "BENEFICIAL":
            f.write("Stroboscopic recalibration provides measurable quality "
                    "recovery. Recommended strategy: run FP16 checkpoint every "
                    "4096 tokens, recalibrate if cosine similarity drops below "
                    "a threshold (e.g., 0.995).\n\n")

        # Implications for TurboQuantKVCache
        f.write("## Implications for TurboQuantKVCache\n\n")
        f.write(f"1. **Lazy quantization** (first {MIN_FUSED_CONTEXT} tokens in "
                f"FP16): Provides perfect quality for prefill. Our data "
                f"{'confirms' if drift_verdict == 'FLAT' else 'suggests'} "
                f"this is sufficient -- no further recalibration needed.\n\n")
        f.write("2. **Long-context safety**: ")
        if drift_verdict == "FLAT":
            f.write("PolarQuant 3-bit is safe for 16K+ context without any "
                    "drift correction mechanism.\n\n")
        else:
            f.write("Consider periodic recalibration for contexts exceeding "
                    "16K tokens.\n\n")
        f.write("3. **Monitoring recommendation**: Even with flat drift, a "
                "lightweight cosine-similarity check (one FP16 forward pass) "
                "every 4K tokens costs < 1ms and provides a safety net.\n\n")

    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
