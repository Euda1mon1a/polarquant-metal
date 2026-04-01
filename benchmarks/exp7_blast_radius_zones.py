#!/usr/bin/env python3
"""
Experiment 7: Blast Radius Positional Zone Containment

Hypothesis: The KV cache can be divided into position-based zones with
independent precision levels. System prompt tokens (positions 0-N) are
accessed by every query and contain critical instructions -- they should
get higher precision. Recent tokens matter most for generation quality.
Mid-context bulk tokens can tolerate cheaper quantization. When error is
high in one zone, it stays contained there.

Inspired by AAPM's blast_radius.py: divide the system into independent
zones. Each zone has its own precision level. Failures (quantization
error) in one zone cannot propagate to affect others.

Key difference from Experiment 4 (spectral bit-width):
  Exp4 tried per-HEAD adaptive bits but failed because PolarQuant's
  rotation makes error pattern-independent.
Key difference from Experiment 5 (hub tokens):
  Exp5 tried attention-based position identification but failed because
  hub-ness is query-dependent.

This experiment uses FIXED STRUCTURAL BOUNDARIES based on position range.
No attention analysis, no content analysis -- pure position-based zoning.

Usage:
    cd ~/workspace/polarquant-metal
    python3 benchmarks/exp7_blast_radius_zones.py
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
L_KV = 16384         # 16K context
L_Q = 1              # decode mode
REP = N_HEADS // N_KV_HEADS  # 4

# Position zones for 16K context
ZONES = {
    "system_prompt": (0, 200),        # First 200 tokens -- critical instructions
    "early_context":  (200, 2048),    # Conversation setup
    "mid_context":    (2048, 14336),  # Bulk conversation (biggest zone)
    "recent":         (14336, 16384), # Last ~2K tokens -- most query-relevant
}

# Strategy definitions: zone_name -> bits (16 = FP16, no quantization)
STRATEGIES = {
    "uniform_3bit": {
        "system_prompt": 3,
        "early_context":  3,
        "mid_context":    3,
        "recent":         3,
    },
    "tiered_conservative": {
        "system_prompt": 4,
        "early_context":  3,
        "mid_context":    3,
        "recent":         4,
    },
    "tiered_aggressive": {
        "system_prompt": 16,  # FP16 -- keep critical tokens pristine
        "early_context":  3,
        "mid_context":    2,  # Save memory on bulk
        "recent":         16, # FP16 -- keep recent tokens pristine
    },
}

N_TIMING_TRIALS = 5  # timing warmup + measurement trials


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two arrays."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    na = mx.sqrt(mx.sum(a_flat * a_flat))
    nb = mx.sqrt(mx.sum(b_flat * b_flat))
    result = dot / (na * nb + 1e-10)
    mx.eval(result)
    return float(result)


# ---------------------------------------------------------------------------
# Realistic attention weight generation
# ---------------------------------------------------------------------------

def create_realistic_weights(rng, B, n_heads, L_q, L_kv):
    """Create realistic attention weights with structure across positions.

    Simulates a decode step where attention has:
    - Moderate mass on system prompt (positions 0-200)
    - Some spread over early context
    - Light mass on mid-context
    - Strong concentration on recent tokens

    This mimics real transformer behavior where recent tokens and
    system prompt tokens get disproportionate attention.
    """
    weights = np.zeros((B, n_heads, L_q, L_kv), dtype=np.float32)

    for h in range(n_heads):
        # System prompt: moderate attention (varies by head)
        sys_mass = rng.uniform(0.05, 0.20)
        sys_len = min(200, L_kv)
        weights[0, h, 0, :sys_len] = rng.dirichlet(
            np.ones(sys_len) * 2.0
        ).astype(np.float32) * sys_mass

        # Early context: light spread
        early_start, early_end = 200, min(2048, L_kv)
        early_len = early_end - early_start
        if early_len > 0:
            early_mass = rng.uniform(0.05, 0.15)
            weights[0, h, 0, early_start:early_end] = rng.dirichlet(
                np.ones(early_len) * 1.0
            ).astype(np.float32) * early_mass

        # Mid context: very light spread
        mid_start, mid_end = 2048, min(14336, L_kv)
        mid_len = mid_end - mid_start
        if mid_len > 0:
            mid_mass = rng.uniform(0.05, 0.15)
            weights[0, h, 0, mid_start:mid_end] = rng.dirichlet(
                np.ones(mid_len) * 0.5
            ).astype(np.float32) * mid_mass

        # Recent tokens: strong concentration
        recent_start = min(14336, L_kv)
        recent_len = L_kv - recent_start
        if recent_len > 0:
            recent_mass = 1.0 - sys_mass - rng.uniform(0.10, 0.30)
            recent_mass = max(recent_mass, 0.3)
            weights[0, h, 0, recent_start:] = rng.dirichlet(
                np.ones(recent_len) * 5.0
            ).astype(np.float32) * recent_mass

        # Normalize to sum to 1
        total = weights[0, h, 0, :].sum()
        if total > 0:
            weights[0, h, 0, :] /= total

    return mx.array(weights)


# ---------------------------------------------------------------------------
# Memory calculation
# ---------------------------------------------------------------------------

def zone_memory_bytes(zones, zone_config, n_kv_heads, D):
    """Compute total V cache memory for a zone configuration.

    For quantized zones: packed indices (uint32) + norms (float16)
    For FP16 zones: raw float16 values
    """
    total = 0
    for zone_name, (start, end) in zones.items():
        L = end - start
        bits = zone_config[zone_name]
        if bits == 16:
            # FP16: n_kv_heads * L * D * 2 bytes
            total += n_kv_heads * L * D * 2
        else:
            vals_per_int = 32 // bits
            D_packed = (D + vals_per_int - 1) // vals_per_int
            # Packed indices: n_kv_heads * L * D_packed * 4 bytes (uint32)
            total += n_kv_heads * L * D_packed * 4
            # Norms: n_kv_heads * L * 2 bytes (float16)
            total += n_kv_heads * L * 2
    return total


def fp16_total_bytes(n_kv_heads, L_kv, D):
    """Full FP16 V cache memory."""
    return n_kv_heads * L_kv * D * 2


# ---------------------------------------------------------------------------
# Zone quantization
# ---------------------------------------------------------------------------

def quantize_zones(V, zones, zone_config, D):
    """Quantize V vectors per zone, return reconstructed V.

    For each zone:
    - FP16 zones: keep original values
    - Quantized zones: PolarQuant quantize + dequantize

    Args:
        V: (B, n_kv_heads, L_kv, D) original value vectors
        zones: dict of zone_name -> (start, end)
        zone_config: dict of zone_name -> bits
        D: head dimension

    Returns:
        V_zoned: (B, n_kv_heads, L_kv, D) reconstructed values
        zone_pqs: dict of zone_name -> PolarQuant (or None for FP16)
    """
    V_zoned = mx.zeros_like(V)
    zone_pqs = {}

    for zone_idx, (zone_name, (start, end)) in enumerate(zones.items()):
        bits = zone_config[zone_name]

        if bits == 16:
            # FP16: no quantization
            V_zoned = V_zoned.at[:, :, start:end, :].add(V[:, :, start:end, :])
            zone_pqs[zone_name] = None
        else:
            # PolarQuant quantize + dequantize
            # Use seed=42 for all zones at same bit-width for consistency;
            # different bit-widths use different codebooks so seed is fine
            pq = PolarQuant(bits=bits, dim=D, seed=42)
            zone_V = V[:, :, start:end, :]
            idx, norms = pq.quantize(zone_V)
            recon = pq.dequantize(idx, norms)
            mx.eval(recon)
            V_zoned = V_zoned.at[:, :, start:end, :].add(recon)
            zone_pqs[zone_name] = pq

    mx.eval(V_zoned)
    return V_zoned, zone_pqs


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 78)
    print("Experiment 7: Blast Radius Positional Zone Containment")
    print("=" * 78)
    print(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:     B={B}, heads={N_HEADS}/{N_KV_HEADS} (GQA {REP}:1), D={D}")
    print(f"Context:    L_kv={L_KV:,}, L_q={L_Q} (decode)")
    print(f"Device:     {mx.default_device()}, Metal={mx.metal.is_available()}")
    print()
    print("Zone layout:")
    for zname, (zstart, zend) in ZONES.items():
        print(f"  {zname:20s}  positions [{zstart:>5d}, {zend:>5d})  "
              f"length={zend-zstart:>5d}")
    print()

    rng = np.random.RandomState(42)

    # Generate V vectors (shared across all strategies)
    V = mx.array(
        rng.standard_normal((B, N_KV_HEADS, L_KV, D)).astype(np.float32)
    )
    mx.eval(V)

    # Generate realistic attention weights
    weights = create_realistic_weights(rng, B, N_HEADS, L_Q, L_KV)
    mx.eval(weights)

    # Expand V for GQA: (B, N_KV_HEADS, L_KV, D) -> (B, N_HEADS, L_KV, D)
    V_expanded = mx.repeat(V, REP, axis=1)
    mx.eval(V_expanded)

    # FP16 baseline output
    baseline_output = weights @ V_expanded  # (B, N_HEADS, L_Q, D)
    mx.eval(baseline_output)

    # Compute per-zone attention mass (how much attention falls in each zone)
    print("Attention mass distribution (averaged across heads):")
    for zname, (zstart, zend) in ZONES.items():
        zone_mass = float(weights[:, :, :, zstart:zend].sum(axis=-1).mean())
        print(f"  {zname:20s}  mass={zone_mass:.4f}")
    print()

    # ==================================================================
    # PART 1: Strategy Comparison
    # ==================================================================
    print("=" * 78)
    print("  PART 1: Strategy Comparison (Overall Quality + Memory)")
    print("=" * 78)

    results = {}
    uniform_memory = None

    for strategy_name, zone_config in STRATEGIES.items():
        print(f"\n--- {strategy_name} ---")
        print(f"  Config: {zone_config}")

        # Quantize zones
        t0 = time.perf_counter()
        V_zoned, zone_pqs = quantize_zones(V, ZONES, zone_config, D)
        t_quant = (time.perf_counter() - t0) * 1000

        # Expand for GQA
        V_zoned_expanded = mx.repeat(V_zoned, REP, axis=1)
        mx.eval(V_zoned_expanded)

        # Compute output
        t0 = time.perf_counter()
        zoned_output = weights @ V_zoned_expanded
        mx.eval(zoned_output)
        t_matmul = (time.perf_counter() - t0) * 1000

        # Overall cosine similarity
        overall_cos = cosine_similarity(baseline_output, zoned_output)

        # Memory
        mem_bytes = zone_memory_bytes(ZONES, zone_config, N_KV_HEADS, D)
        fp16_bytes = fp16_total_bytes(N_KV_HEADS, L_KV, D)

        if strategy_name == "uniform_3bit":
            uniform_memory = mem_bytes

        mem_vs_fp16 = (1.0 - mem_bytes / fp16_bytes) * 100
        mem_vs_uniform = (
            (1.0 - mem_bytes / uniform_memory) * 100
            if uniform_memory else 0.0
        )

        print(f"  Overall cos_sim:      {overall_cos:.8f}")
        print(f"  Memory:               {mem_bytes:>10,d} bytes "
              f"({mem_bytes / 1024:.1f} KB)")
        print(f"  vs FP16:              {mem_vs_fp16:+.1f}%")
        print(f"  vs uniform 3-bit:     {mem_vs_uniform:+.1f}%")
        print(f"  Quantize time:        {t_quant:.1f} ms")
        print(f"  Matmul time:          {t_matmul:.1f} ms")

        results[strategy_name] = {
            "zone_config": zone_config,
            "overall_cos_sim": overall_cos,
            "memory_bytes": mem_bytes,
            "mem_vs_fp16_pct": mem_vs_fp16,
            "mem_vs_uniform_pct": mem_vs_uniform,
            "quant_time_ms": t_quant,
            "matmul_time_ms": t_matmul,
            "V_zoned": V_zoned,
            "V_zoned_expanded": V_zoned_expanded,
            "zoned_output": zoned_output,
        }

    gc.collect()

    # ==================================================================
    # PART 2: Per-Zone Quality Analysis
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 2: Per-Zone Quality (Cosine Similarity by Zone)")
    print("=" * 78)
    print()
    print("  For each zone, we compute quality using ONLY that zone's positions.")
    print("  This isolates the blast radius -- does protecting system_prompt help?")
    print("  Does downgrading mid_context to 2-bit hurt within that zone?")
    print()

    zone_results = {}

    for strategy_name in STRATEGIES:
        V_zoned_expanded = results[strategy_name]["V_zoned_expanded"]
        zone_results[strategy_name] = {}

        for zname, (zstart, zend) in ZONES.items():
            # Extract zone-specific weights and renormalize
            zone_weights = weights[:, :, :, zstart:zend]
            zone_sum = zone_weights.sum(axis=-1, keepdims=True)
            # Avoid division by zero for zones with no attention
            zone_sum = mx.maximum(zone_sum, mx.array(1e-10))
            zone_weights_norm = zone_weights / zone_sum

            # Compute zone-specific outputs
            zone_baseline = zone_weights_norm @ V_expanded[:, :, zstart:zend, :]
            zone_test = zone_weights_norm @ V_zoned_expanded[:, :, zstart:zend, :]
            mx.eval(zone_baseline, zone_test)

            zone_cos = cosine_similarity(zone_baseline, zone_test)
            zone_results[strategy_name][zname] = zone_cos

    # Print zone quality table
    header = f"  {'Zone':<20s}"
    for sname in STRATEGIES:
        header += f" | {sname:>22s}"
    print(header)
    print(f"  {'-'*20}" + "".join(f"-+-{'-'*22}" for _ in STRATEGIES))

    for zname in ZONES:
        row = f"  {zname:<20s}"
        for sname in STRATEGIES:
            cs = zone_results[sname][zname]
            bits = STRATEGIES[sname][zname]
            bits_label = "FP16" if bits == 16 else f"{bits}b"
            row += f" | {cs:.8f} ({bits_label:>4s})"
        print(row)

    # Also show overall
    row = f"  {'OVERALL':<20s}"
    for sname in STRATEGIES:
        cs = results[sname]["overall_cos_sim"]
        row += f" |           {cs:.8f}"
    print(row)

    gc.collect()

    # ==================================================================
    # PART 3: Reconstruction Error by Zone (Without Attention Weighting)
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 3: Raw Reconstruction Quality by Zone (No Attention)")
    print("=" * 78)
    print()
    print("  Direct cosine similarity between V and V_zoned per zone,")
    print("  without attention weights. This shows raw quantization quality.")
    print()

    raw_zone_results = {}

    for strategy_name in STRATEGIES:
        V_zoned = results[strategy_name]["V_zoned"]
        raw_zone_results[strategy_name] = {}

        for zname, (zstart, zend) in ZONES.items():
            raw_cos = cosine_similarity(
                V[:, :, zstart:zend, :],
                V_zoned[:, :, zstart:zend, :],
            )
            raw_zone_results[strategy_name][zname] = raw_cos

    header = f"  {'Zone':<20s}"
    for sname in STRATEGIES:
        header += f" | {sname:>22s}"
    print(header)
    print(f"  {'-'*20}" + "".join(f"-+-{'-'*22}" for _ in STRATEGIES))

    for zname in ZONES:
        row = f"  {zname:<20s}"
        for sname in STRATEGIES:
            cs = raw_zone_results[sname][zname]
            bits = STRATEGIES[sname][zname]
            bits_label = "FP16" if bits == 16 else f"{bits}b"
            row += f" | {cs:.8f} ({bits_label:>4s})"
        print(row)

    gc.collect()

    # ==================================================================
    # PART 4: Memory Budget Analysis
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 4: Memory Budget Analysis")
    print("=" * 78)
    print()

    fp16_bytes = fp16_total_bytes(N_KV_HEADS, L_KV, D)
    print(f"  FP16 baseline: {fp16_bytes:>10,d} bytes ({fp16_bytes/1024:.1f} KB)")
    print()

    for sname, sconfig in STRATEGIES.items():
        total = results[sname]["memory_bytes"]
        print(f"  {sname}:")
        for zname, (zstart, zend) in ZONES.items():
            L = zend - zstart
            bits = sconfig[zname]
            if bits == 16:
                zbytes = N_KV_HEADS * L * D * 2
            else:
                vals_per_int = 32 // bits
                D_packed = (D + vals_per_int - 1) // vals_per_int
                zbytes = N_KV_HEADS * L * D_packed * 4 + N_KV_HEADS * L * 2
            bits_label = "FP16" if bits == 16 else f"{bits}-bit"
            pct = zbytes / total * 100
            print(f"    {zname:20s}  {bits_label:>5s}  "
                  f"{zbytes:>8,d} bytes  ({pct:5.1f}% of total)")
        print(f"    {'TOTAL':20s}         {total:>8,d} bytes  "
              f"({total/1024:.1f} KB)  "
              f"[{results[sname]['mem_vs_fp16_pct']:+.1f}% vs FP16]")
        print()

    gc.collect()

    # ==================================================================
    # PART 5: Fused Kernel Dispatch Timing (Per-Zone)
    # ==================================================================
    print(f"\n\n{'='*78}")
    print("  PART 5: Fused SV Kernel Dispatch Timing (Per-Zone)")
    print("=" * 78)
    print()
    print("  If using fused kernels per zone, each zone needs its own dispatch.")
    print("  Measuring overhead of per-zone kernel dispatch vs single dispatch.")
    print()

    # Single dispatch (uniform 3-bit): quantize all, single kernel call
    pq3 = PolarQuant(bits=3, dim=D, seed=42)
    idx3, norms3 = pq3.quantize(V)
    packed3 = pack_indices(idx3, 3)
    centroids3 = load_codebook_f32(3, D)
    mx.eval(packed3, norms3, centroids3)

    # Warmup
    for _ in range(2):
        out = polarquant_sv_matmul(
            weights=weights, v_indices=packed3, v_norms=norms3,
            v_centroids=centroids3, head_dim=D, bits=3,
        )
        mx.eval(out)

    # Time single dispatch
    times_single = []
    for _ in range(N_TIMING_TRIALS):
        t0 = time.perf_counter()
        out = polarquant_sv_matmul(
            weights=weights, v_indices=packed3, v_norms=norms3,
            v_centroids=centroids3, head_dim=D, bits=3,
        )
        mx.eval(out)
        times_single.append((time.perf_counter() - t0) * 1000)

    avg_single = sum(times_single) / len(times_single)
    print(f"  Single dispatch (uniform 3-bit):  {avg_single:.2f} ms "
          f"(avg of {N_TIMING_TRIALS})")

    # Per-zone dispatch for tiered_conservative (3-bit and 4-bit zones)
    zone_packed = {}
    zone_norms = {}
    zone_centroids = {}
    zone_bits = {}

    for zname, (zstart, zend) in ZONES.items():
        bits = STRATEGIES["tiered_conservative"][zname]
        pq = PolarQuant(bits=bits, dim=D, seed=42)
        zone_V = V[:, :, zstart:zend, :]
        idx, norms = pq.quantize(zone_V)
        packed = pack_indices(idx, bits)
        mx.eval(packed, norms)
        zone_packed[zname] = packed
        zone_norms[zname] = norms
        zone_centroids[zname] = load_codebook_f32(bits, D)
        zone_bits[zname] = bits

    # Warmup per-zone dispatch
    for _ in range(2):
        for zname, (zstart, zend) in ZONES.items():
            zone_w = weights[:, :, :, zstart:zend]
            out = polarquant_sv_matmul(
                weights=zone_w,
                v_indices=zone_packed[zname],
                v_norms=zone_norms[zname],
                v_centroids=zone_centroids[zname],
                head_dim=D, bits=zone_bits[zname],
            )
            mx.eval(out)

    # Time per-zone dispatch
    times_multi = []
    for _ in range(N_TIMING_TRIALS):
        t0 = time.perf_counter()
        zone_outputs = []
        for zname, (zstart, zend) in ZONES.items():
            zone_w = weights[:, :, :, zstart:zend]
            out = polarquant_sv_matmul(
                weights=zone_w,
                v_indices=zone_packed[zname],
                v_norms=zone_norms[zname],
                v_centroids=zone_centroids[zname],
                head_dim=D, bits=zone_bits[zname],
            )
            zone_outputs.append(out)
        # Sum zone outputs (each zone contributes its weighted portion)
        combined = zone_outputs[0]
        for zo in zone_outputs[1:]:
            combined = combined + zo
        mx.eval(combined)
        times_multi.append((time.perf_counter() - t0) * 1000)

    avg_multi = sum(times_multi) / len(times_multi)
    overhead = avg_multi - avg_single
    overhead_pct = (overhead / avg_single) * 100 if avg_single > 0 else 0

    print(f"  Per-zone dispatch (conservative):  {avg_multi:.2f} ms "
          f"(avg of {N_TIMING_TRIALS})")
    print(f"  Overhead:                          {overhead:+.2f} ms "
          f"({overhead_pct:+.1f}%)")
    print()

    gc.collect()

    # ==================================================================
    # PART 6: Key Questions Answered
    # ==================================================================
    print(f"\n{'='*78}")
    print("  PART 6: Key Questions")
    print("=" * 78)

    # Q1: Does protecting system_prompt (4-bit or FP16) improve overall quality?
    u_cos = results["uniform_3bit"]["overall_cos_sim"]
    c_cos = results["tiered_conservative"]["overall_cos_sim"]
    a_cos = results["tiered_aggressive"]["overall_cos_sim"]

    delta_conservative = c_cos - u_cos
    delta_aggressive = a_cos - u_cos

    print(f"\n  Q1: Does protecting system_prompt zone improve overall quality?")
    print(f"      Uniform 3-bit:        {u_cos:.8f}")
    print(f"      Conservative (4b sys): {c_cos:.8f}  "
          f"delta={delta_conservative:+.8f}")
    print(f"      Aggressive (FP16 sys): {a_cos:.8f}  "
          f"delta={delta_aggressive:+.8f}")

    if delta_conservative > 0.0001:
        print(f"      -> YES, measurable improvement from protecting system_prompt")
    elif delta_conservative > 0:
        print(f"      -> MARGINAL, tiny improvement (may not matter in practice)")
    else:
        print(f"      -> NO, protecting system_prompt does not help overall quality")

    # Q2: Does downgrading mid_context to 2-bit hurt?
    u_mid = zone_results["uniform_3bit"]["mid_context"]
    a_mid = zone_results["tiered_aggressive"]["mid_context"]
    delta_mid = a_mid - u_mid

    print(f"\n  Q2: Does downgrading mid_context to 2-bit hurt?")
    print(f"      Uniform 3-bit mid_context:    {u_mid:.8f}")
    print(f"      Aggressive 2-bit mid_context: {a_mid:.8f}  "
          f"delta={delta_mid:+.8f}")

    if abs(delta_mid) < 0.001:
        print(f"      -> NEGLIGIBLE impact on zone quality")
    elif delta_mid < -0.01:
        print(f"      -> YES, significant quality loss in mid_context zone")
    else:
        print(f"      -> SMALL impact ({delta_mid:+.6f})")

    # Q3: Per-zone quality isolation -- does error stay contained?
    print(f"\n  Q3: Is error contained within zones (blast radius isolation)?")

    # In aggressive strategy, system_prompt is FP16 (perfect) and
    # mid_context is 2-bit (worse). Check that mid_context degradation
    # does not bleed into system_prompt quality.
    a_sys = zone_results["tiered_aggressive"]["system_prompt"]
    a_recent = zone_results["tiered_aggressive"]["recent"]
    print(f"      Aggressive system_prompt (FP16): {a_sys:.8f}")
    print(f"      Aggressive recent (FP16):        {a_recent:.8f}")
    print(f"      Aggressive mid_context (2-bit):  {a_mid:.8f}")

    if a_sys > 0.9999 and a_recent > 0.9999:
        print(f"      -> YES, FP16 zones maintain perfect quality despite "
              f"2-bit mid_context")
        print(f"         Blast radius containment WORKS.")
    elif a_sys > a_mid and a_recent > a_mid:
        print(f"      -> YES, zone quality correlates with zone precision.")
        print(f"         Error is contained within zone boundaries.")
    else:
        print(f"      -> MIXED, zone boundaries provide some but not full "
              f"isolation")

    # Q4: Memory tradeoff
    print(f"\n  Q4: Memory efficiency of tiered strategies?")
    for sname in STRATEGIES:
        mem = results[sname]["memory_bytes"]
        cos = results[sname]["overall_cos_sim"]
        quality_per_kb = cos / (mem / 1024)
        print(f"      {sname:25s}  {mem/1024:>7.1f} KB  "
              f"cos={cos:.8f}  quality/KB={quality_per_kb:.6f}")

    print()

    # ==================================================================
    # Save results
    # ==================================================================
    md_path = os.path.join(os.path.dirname(__file__), "EXP7_RESULTS.md")
    _write_results_md(
        md_path, results, zone_results, raw_zone_results,
        avg_single, avg_multi, overhead, overhead_pct,
        delta_conservative, delta_aggressive, delta_mid,
    )
    print(f"  Results saved to: {md_path}")

    print(f"\n{'='*78}")
    print("  EXPERIMENT 7 COMPLETE")
    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _write_results_md(
    md_path, results, zone_results, raw_zone_results,
    avg_single, avg_multi, overhead, overhead_pct,
    delta_conservative, delta_aggressive, delta_mid,
):
    with open(md_path, "w") as f:
        f.write("# Experiment 7: Blast Radius Positional Zone Containment\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config:** B={B}, heads={N_HEADS}/{N_KV_HEADS} "
                f"(GQA {REP}:1), D={D}  \n")
        f.write(f"**Context:** L_kv={L_KV:,}, L_q={L_Q} (decode)  \n")
        f.write(f"**Device:** {mx.default_device()}, "
                f"Metal={mx.metal.is_available()}  \n\n")

        # Hypothesis
        f.write("## Hypothesis\n\n")
        f.write("The KV cache can be divided into position-based zones with "
                "independent precision levels. System prompt tokens get higher "
                "precision because they are accessed by every query. Recent "
                "tokens get higher precision because they are most relevant. "
                "Mid-context bulk tokens can tolerate cheaper quantization. "
                "Quantization error in one zone stays contained within that "
                "zone's boundaries (blast radius isolation).\n\n")
        f.write("Inspired by AAPM's `blast_radius.py`: divide the system into "
                "independent zones where failures cannot propagate.\n\n")

        # Zone layout
        f.write("## Zone Layout (16K Context)\n\n")
        f.write("| Zone | Positions | Length | Role |\n")
        f.write("|:-----|----------:|-------:|:-----|\n")
        for zname, (zstart, zend) in ZONES.items():
            length = zend - zstart
            roles = {
                "system_prompt": "Critical instructions, accessed by every query",
                "early_context": "Conversation setup, moderate importance",
                "mid_context": "Bulk conversation, largest zone",
                "recent": "Last ~2K tokens, most query-relevant",
            }
            f.write(f"| {zname} | [{zstart}, {zend}) | {length:,} | "
                    f"{roles.get(zname, '')} |\n")

        # Strategies
        f.write("\n## Strategies\n\n")
        f.write("| Zone | Uniform 3-bit | Conservative | Aggressive |\n")
        f.write("|:-----|:-------------:|:------------:|:----------:|\n")
        for zname in ZONES:
            u = STRATEGIES["uniform_3bit"][zname]
            c = STRATEGIES["tiered_conservative"][zname]
            a = STRATEGIES["tiered_aggressive"][zname]
            u_label = "FP16" if u == 16 else f"{u}-bit"
            c_label = "FP16" if c == 16 else f"{c}-bit"
            a_label = "FP16" if a == 16 else f"{a}-bit"
            f.write(f"| {zname} | {u_label} | {c_label} | {a_label} |\n")

        # Part 1: Overall quality + memory
        f.write("\n## Part 1: Overall Quality and Memory\n\n")
        f.write("| Strategy | Cosine Sim | Memory (KB) | vs FP16 | "
                "vs Uniform 3-bit |\n")
        f.write("|:---------|----------:|-----------:|-------:|"
                "----------------:|\n")
        for sname in STRATEGIES:
            r = results[sname]
            f.write(f"| {sname} | {r['overall_cos_sim']:.8f} | "
                    f"{r['memory_bytes']/1024:.1f} | "
                    f"{r['mem_vs_fp16_pct']:+.1f}% | "
                    f"{r['mem_vs_uniform_pct']:+.1f}% |\n")

        # Part 2: Per-zone quality
        f.write("\n## Part 2: Per-Zone Cosine Similarity (Attention-Weighted)\n\n")
        header = "| Zone |"
        for sname in STRATEGIES:
            header += f" {sname} |"
        f.write(header + "\n")
        f.write("|:-----|" + "".join("-----------:|" for _ in STRATEGIES) + "\n")
        for zname in ZONES:
            row = f"| {zname} |"
            for sname in STRATEGIES:
                cs = zone_results[sname][zname]
                bits = STRATEGIES[sname][zname]
                bits_label = "FP16" if bits == 16 else f"{bits}b"
                row += f" {cs:.8f} ({bits_label}) |"
            f.write(row + "\n")

        # Part 3: Raw reconstruction quality
        f.write("\n## Part 3: Raw Reconstruction Quality (No Attention)\n\n")
        header = "| Zone |"
        for sname in STRATEGIES:
            header += f" {sname} |"
        f.write(header + "\n")
        f.write("|:-----|" + "".join("-----------:|" for _ in STRATEGIES) + "\n")
        for zname in ZONES:
            row = f"| {zname} |"
            for sname in STRATEGIES:
                cs = raw_zone_results[sname][zname]
                bits = STRATEGIES[sname][zname]
                bits_label = "FP16" if bits == 16 else f"{bits}b"
                row += f" {cs:.8f} ({bits_label}) |"
            f.write(row + "\n")

        # Part 4: Memory breakdown
        f.write("\n## Part 4: Memory Breakdown by Zone\n\n")
        for sname, sconfig in STRATEGIES.items():
            total = results[sname]["memory_bytes"]
            f.write(f"### {sname} ({total/1024:.1f} KB total)\n\n")
            f.write("| Zone | Bits | Bytes | % of Total |\n")
            f.write("|:-----|:----:|------:|-----------:|\n")
            for zname, (zstart, zend) in ZONES.items():
                L = zend - zstart
                bits = sconfig[zname]
                if bits == 16:
                    zbytes = N_KV_HEADS * L * D * 2
                else:
                    vals_per_int = 32 // bits
                    D_packed = (D + vals_per_int - 1) // vals_per_int
                    zbytes = N_KV_HEADS * L * D_packed * 4 + N_KV_HEADS * L * 2
                bits_label = "FP16" if bits == 16 else f"{bits}-bit"
                pct = zbytes / total * 100
                f.write(f"| {zname} | {bits_label} | {zbytes:,} | "
                        f"{pct:.1f}% |\n")
            f.write("\n")

        # Part 5: Timing
        f.write("## Part 5: Kernel Dispatch Timing\n\n")
        f.write(f"| Approach | Time (ms) | Overhead |\n")
        f.write(f"|:---------|----------:|---------:|\n")
        f.write(f"| Single dispatch (uniform 3-bit) | {avg_single:.2f} | "
                f"baseline |\n")
        f.write(f"| Per-zone dispatch (4 zones) | {avg_multi:.2f} | "
                f"{overhead:+.2f} ms ({overhead_pct:+.1f}%) |\n\n")

        # Conclusion
        f.write("## Conclusions\n\n")
        f.write("### Key Questions Answered\n\n")

        # Q1
        f.write("**Q1: Does protecting system_prompt improve overall quality?**\n\n")
        if delta_conservative > 0.0001:
            f.write(f"YES. Conservative tiering (4-bit system_prompt) improves "
                    f"overall cosine similarity by {delta_conservative:+.8f}. "
                    f"Aggressive tiering (FP16 system_prompt + recent) improves "
                    f"by {delta_aggressive:+.8f}.\n\n")
        elif delta_conservative > 0:
            f.write(f"MARGINAL. Conservative tiering shows a tiny improvement "
                    f"of {delta_conservative:+.8f}. The system prompt zone is "
                    f"small (200 tokens out of 16K) so its impact on overall "
                    f"quality is proportionally small.\n\n")
        else:
            f.write(f"NO. Protecting the system prompt zone does not measurably "
                    f"improve overall quality. The zone is too small relative to "
                    f"the full context to make a difference in aggregate cosine "
                    f"similarity.\n\n")

        # Q2
        f.write("**Q2: Does downgrading mid_context to 2-bit hurt?**\n\n")
        if abs(delta_mid) < 0.001:
            f.write(f"NEGLIGIBLE. The mid_context zone quality changes by "
                    f"only {delta_mid:+.8f} when dropping from 3-bit to 2-bit. "
                    f"PolarQuant's rotation decorrelation provides robust "
                    f"quality even at 2-bit.\n\n")
        elif delta_mid < -0.01:
            f.write(f"YES. Significant quality loss of {delta_mid:+.8f} in "
                    f"the mid_context zone. 2-bit quantization degrades "
                    f"reconstruction fidelity.\n\n")
        else:
            f.write(f"SMALL. Quality change of {delta_mid:+.8f} -- modest "
                    f"degradation but may be acceptable for the memory savings.\n\n")

        # Q3
        a_sys = zone_results["tiered_aggressive"]["system_prompt"]
        a_mid = zone_results["tiered_aggressive"]["mid_context"]
        a_recent = zone_results["tiered_aggressive"]["recent"]

        f.write("**Q3: Is quantization error contained within zones "
                "(blast radius isolation)?**\n\n")
        if a_sys > 0.9999 and a_recent > 0.9999:
            f.write(f"YES -- STRONG CONTAINMENT. FP16 zones (system_prompt: "
                    f"{a_sys:.8f}, recent: {a_recent:.8f}) maintain near-perfect "
                    f"quality despite 2-bit mid_context ({a_mid:.8f}). "
                    f"Error is fully contained within zone boundaries.\n\n")
        elif a_sys > a_mid and a_recent > a_mid:
            f.write(f"YES. Zone quality correlates strongly with zone precision "
                    f"(system_prompt FP16: {a_sys:.8f}, mid_context 2-bit: "
                    f"{a_mid:.8f}). Error stays within zone boundaries.\n\n")
        else:
            f.write(f"PARTIAL. Zone boundaries provide some isolation but "
                    f"error partially propagates through attention weighting.\n\n")

        # Verdict
        f.write("### Verdict\n\n")

        u_cos = results["uniform_3bit"]["overall_cos_sim"]
        c_cos = results["tiered_conservative"]["overall_cos_sim"]
        a_cos = results["tiered_aggressive"]["overall_cos_sim"]
        a_mem = results["tiered_aggressive"]["mem_vs_uniform_pct"]
        c_mem = results["tiered_conservative"]["mem_vs_uniform_pct"]

        if a_cos > u_cos and a_mem < -5:
            f.write("**POSITIVE**: Aggressive tiered zoning improves quality "
                    f"({a_cos:.8f} vs {u_cos:.8f}) while saving "
                    f"{abs(a_mem):.1f}% memory vs uniform 3-bit. The blast "
                    f"radius pattern successfully contains quantization error "
                    f"within zone boundaries. This approach is immediately "
                    f"deployable: protect critical zones (system prompt, recent) "
                    f"with higher precision and save memory on bulk mid-context "
                    f"with 2-bit.\n\n")
        elif a_cos >= u_cos * 0.9999:
            f.write("**MIXED-POSITIVE**: Aggressive tiered zoning maintains "
                    f"comparable quality ({a_cos:.8f} vs {u_cos:.8f}) while "
                    f"the memory impact is {a_mem:+.1f}% vs uniform 3-bit. "
                    f"Blast radius containment works (per-zone quality tracks "
                    f"precision), but the overall impact depends on whether "
                    f"the memory savings from 2-bit mid-context outweigh the "
                    f"FP16 cost of system_prompt + recent zones.\n\n")
        else:
            f.write("**NEGATIVE**: Tiered zoning does not provide a clear "
                    f"advantage over uniform 3-bit quantization. The overhead "
                    f"of per-zone dispatch and the memory cost of FP16 zones "
                    f"may not be justified.\n\n")

        f.write("### Implementation Notes\n\n")
        f.write("1. Zone boundaries are fixed at cache allocation time -- no "
                "runtime overhead for boundary detection.\n")
        f.write("2. Each zone uses its own PolarQuant quantizer (same seed, "
                "different bit-width).\n")
        f.write("3. Per-zone kernel dispatch adds modest overhead "
                f"({overhead:+.2f} ms / {overhead_pct:+.1f}%) but could be "
                f"mitigated by batching zones of the same bit-width.\n")
        f.write("4. The system_prompt zone length should be configurable -- "
                "200 tokens is a reasonable default but some models use longer "
                "system prompts.\n")
        f.write("5. Integration with TurboQuantKVCache would require "
                "per-zone packed arrays and norms, dispatched from "
                "`fused_sdpa()`.\n")


if __name__ == "__main__":
    run_experiment()
