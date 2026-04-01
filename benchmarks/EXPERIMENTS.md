# PolarQuant Optimization Experiments — Consolidated Findings

> **Date:** 2026-03-31
> **Platform:** Mac Mini M4 Pro 64GB, Metal GPU
> **Model config:** B=1, 8 query heads, 2 KV heads (GQA 4:1), D=128, 3-bit PolarQuant
> **Origin:** Cross-pollination from AAPM exotic resilience math (entropy, time crystal, foam topology, catastrophe theory)

---

## Overview

Four experiments tested whether physics/information-theory concepts from the AAPM
scheduling project could optimize PolarQuant's KV cache compression on Apple Silicon.

| # | Experiment | AAPM Source | Target | Verdict |
|---|---|---|---|---|
| 1 | Entropy-guided sparse V | `thermodynamics/entropy.py` | SV kernel (88% of decode time) | **Strong win** |
| 2 | Anti-churn rigidity gate | `periodicity/anti_churn.py` | Quantization overhead | **Signal validated** |
| 3 | Stroboscopic drift detection | `periodicity/subharmonic_detector.py` | Cumulative quality drift | **No drift found** |
| 4 | Spectral bit-width selection | `periodicity/subharmonic_detector.py` | Memory (per-head bit-width) | **Hypothesis rejected** |

---

## Experiment 1: Entropy-Guided Adaptive Sparse V Threshold

**Concept:** Compute Shannon entropy of attention weights per head after softmax. Low-entropy heads (concentrated attention) tolerate aggressive pruning in the SV kernel. High-entropy heads (spread attention) must not be pruned.

**Key results (16K context):**

| Distribution | Fixed t=0.01 | Entropy-guided |
|---|---|---|
| Concentrated | cos=0.984, 3.7x speedup, 99.7% skip | cos=0.984, 3.0x speedup, 99.7% skip |
| Spread | **cos=0.000 (destroyed)** | cos=0.984, 0% skip (correctly disabled) |
| Realistic mix | cos=0.984 (misleading*) | cos=0.984, 50% skip |

*Fixed threshold's aggregate cos_sim hides per-head catastrophe: spread heads H4-H7 all at cos=0.000.

**Mechanism:** `threshold = max_threshold × sigmoid(-10 × (entropy - 0.5))`
- Entropy <0.4 → threshold ~0.008-0.010 (aggressive prune)
- Entropy >0.8 → threshold ~0.0005 (effectively no prune)

**Status:** Ready for Phase 1 integration (Python-level per-head dispatch).

---

## Experiment 2: Anti-Churn Rigidity Gate for Re-quantization

**Concept:** Consecutive decode tokens produce similar KV projections. Measure Hamming distance between consecutive codebook index vectors. If rigidity (1 - hamming_distance) exceeds threshold, reuse previous indices and only update the norm.

**Key results (1000 tokens):**

| Pattern | Mean Rigidity | Skip % (t=0.90) | Cos Sim | Speedup |
|---|---|---|---|---|
| Smooth | 0.803 | 78.2% | 0.991 | 1.01x* |
| Random | 0.150 | 0.0% | 1.000 | 0.69x* |
| Mixed | 0.790 | 62.7% | 0.992 | 0.60x* |

*Python-level overhead of Hamming check exceeds pack_indices savings. A Metal-kernel
implementation that fuses the check would skip both quantize() AND pack_indices(),
roughly doubling the savings.

**Signal:** Strong — 80% index overlap on smooth sequences, 15% on random (near chance for 8-level codebook). The gate correctly identifies reusable tokens.

**Caveat:** Only worth implementing as a Metal kernel guard, not in Python.

**Status:** Validated but requires Metal kernel work for real speedup.

---

## Experiment 3: Stroboscopic FP16 Checkpoints for Drift Detection

**Concept:** Does PolarQuant quantization error accumulate over long conversations? Periodically run full FP16 attention as a calibration checkpoint and compare to quantized output.

**Key results (16K context, checkpoints every 64-4096 tokens):**

| Context | Cos Sim (PQ vs FP16) | L2 Distance | Trend |
|---|---|---|---|
| 64 tokens | 0.992 | 0.464 | — |
| 512 tokens | 0.998 | 0.098 | Improving |
| 4K tokens | 0.998 | 0.044 | Flat |
| 8K tokens | 0.998 | 0.035 | Flat |
| 16K tokens | 0.998 | 0.030 | Flat |

**Drift rate:** +0.000020 cos_sim / 1K tokens (effectively zero — quality *improves* slightly as softmax averaging dilutes per-token errors).

**Recalibration test:** Re-quantizing entire cache from FP16 ground truth produced byte-identical packed representation. Delta = 0.000000.

**Conclusion:** No drift correction needed. PolarQuant 3-bit is safe for 16K+ context. The lazy quantization threshold (first 512 tokens FP16) is sufficient. Stroboscopic checkpoints useful for production monitoring, not correction.

**Implication for foam T1:** Event-driven recalibration (the unique value foam topology would add) has no target to fix. Foam T1 can be dropped.

**Status:** Complete — no action needed.

---

## Experiment 4: Spectral Concentration for Per-Head Bit-Width Selection

**Concept:** Heads with periodic attention patterns (high spectral concentration) might tolerate cheaper 2-bit quantization. Use periodogram to classify heads and assign bit-widths adaptively.

**Key results (8K context):**

| Pattern | Spectral Conc. | 2-bit Cos Sim | 3-bit Cos Sim | 2-bit Safe? |
|---|---|---|---|---|
| Periodic-64 | 0.031 | 0.946 | 0.981 | No (< 0.95) |
| Periodic-128 | 0.016 | 0.940 | 0.981 | No |
| Sparse-20 | 0.002 | 0.943 | 0.986 | No |
| Sparse-100 | 0.002 | 0.914 | 0.977 | No |
| Diffuse | 0.002 | 0.940 | 0.983 | No |

**Why it failed:** PolarQuant's random orthogonal rotation decorrelates the signal before quantization. This makes quantization error relatively pattern-independent — the attention weight shape doesn't matter much because the V vectors are rotated into a random basis first. This is actually *good news* for the current uniform 3-bit approach.

**Memory:** Adaptive bit-width actually *increased* memory (+22%) because the conservative classifier correctly recommended 4-bit for most heads, and GQA forces KV heads to the max bit-width of any sharing query head.

**Status:** Hypothesis rejected. Uniform 3-bit is the correct choice.

---

## Combination Analysis

### What works together

| Combination | Viable? | Rationale |
|---|---|---|
| Entropy (1) + Rigidity (2) | **Yes** | Different pipeline stages, no interference |
| Entropy (1) + Stroboscopic (3) | Monitoring only | No drift to correct, but useful as safety net |
| Rigidity (2) + Stroboscopic (3) | No | Rigidity saves quantization work; no drift means no checkpoints needed |
| Spectral (4) + anything | **No** | Hypothesis rejected; rotation decorrelation makes pattern-based bit selection ineffective |
| Foam T1 + anything | **No** | Drift doesn't exist; event-driven correction has no target |

### Implemented (2026-03-31)

1. **Phase 1: Entropy-guided sparse V** in `fused_sdpa()` — SHIPPED
   - `_compute_adaptive_threshold()` returns per-head threshold array
   - Sigmoid mapping: low entropy → aggressive pruning, high entropy → no pruning

2. **Phase 2a: Per-head threshold array** in SV Metal kernel — SHIPPED
   - `sparse_thresh[h_idx]` replaces `sparse_thresh[0]`
   - Single kernel dispatch with per-head adaptive thresholds

3. **Phase 2b: Rigidity gate** in `update_and_fetch()` — SHIPPED
   - Cosine similarity of rotated unit vectors vs previous token
   - 78% skip rate on smooth sequences, 0% on random (correctly gated)
   - `rigidity_stats` property for observability

---

## Files

| File | Description |
|---|---|
| `benchmarks/exp1_entropy_sparse_v.py` | Experiment 1 script |
| `benchmarks/EXP1_RESULTS.md` | Experiment 1 detailed results |
| `benchmarks/exp2_rigidity_gate.py` | Experiment 2 script |
| `benchmarks/EXP2_RESULTS.md` | Experiment 2 detailed results |
| `benchmarks/exp3_stroboscopic_drift.py` | Experiment 3 script |
| `benchmarks/EXP3_RESULTS.md` | Experiment 3 detailed results |
| `benchmarks/exp4_spectral_bitwidth.py` | Experiment 4 script |
| `benchmarks/EXP4_RESULTS.md` | Experiment 4 detailed results |
| `benchmarks/stress_test_long_context.py` | 16K-32K stress test |
| `benchmarks/STRESS_RESULTS.md` | Stress test results |
| `benchmarks/EXPERIMENTS.md` | This consolidated document |
