# PolarQuant Optimization Experiments — Consolidated Findings

> **Updated:** 2026-03-31
> **Platform:** Mac Mini M4 Pro 64GB, Metal GPU
> **Model config:** B=1, 8 query heads, 2 KV heads (GQA 4:1), D=128, 3-bit PolarQuant
> **Origin:** Cross-pollination from AAPM exotic resilience math (entropy, time crystal, foam topology, catastrophe theory)

---

## Overview

Nine experiments tested whether physics/information-theory/queuing-theory concepts
from the AAPM scheduling project could optimize PolarQuant's KV cache compression
on Apple Silicon.

| # | Experiment | AAPM Source | Target | Verdict |
|---|---|---|---|---|
| 1 | Entropy-guided sparse V | `thermodynamics/entropy.py` | SV kernel (88% of decode time) | **Strong win** |
| 2 | Anti-churn rigidity gate | `periodicity/anti_churn.py` | Quantization overhead | **Signal validated** |
| 3 | Stroboscopic drift detection | `periodicity/subharmonic_detector.py` | Cumulative quality drift | **No drift found** |
| 4 | Spectral bit-width selection | `periodicity/subharmonic_detector.py` | Memory (per-head bit-width) | **Hypothesis rejected** |
| 5 | Hub token protection | Scale-free network analysis | Sparse V quality | **Confirmed safe** |
| 6 | STA/LTA entropy amortization | `resilience/seismic_detection.py` | Entropy computation overhead | **Strong win (fixed interval)** |
| 7 | Blast radius zones | `resilience/blast_radius.py` | Fault isolation | See EXP7_RESULTS.md |
| 8 | SPC/Western Electric quality | `resilience/spc.py` | Quality monitoring | See EXP8_RESULTS.md |
| 9 | Erlang workload prediction | `queuing/erlang_c.py` | SV kernel dispatch | **Partial (accurate but too costly)** |
| 10 | **Lazy prefill** | Engineering fix | Prefill O(N²) overhead | **NEXT — priority** |

---

## Experiment 10: Lazy Prefill (FP16 Through Entire Prefill)

**Status:** Implemented, pending benchmark on Mini (2026-04-15)

**Problem:** Chunked prefill with quantized KV is O(N²). Current code bulk-quantizes
FP16→PQ at `min_fused_context=512` tokens. Every subsequent prefill chunk must
dequantize all prior PQ tokens for standard SDPA. At 64K context: >112 min, killed.

**Root cause identified empirically (2026-04-15):** 64K prefill with PQ ran >112 min
on M5 Max MBP (128GB). FP16 baseline: ~2 min. Ratio: >56x. The threshold mismatch
(trigger at 512 tokens, not at end of prefill) is the cause.

**Fix (two changes, minimal):**

1. `turboquant_cache.py` — Remove mid-prefill bulk-quantize trigger:
   - Before: `if self.offset >= self.min_fused_context: self._bulk_quantize()`
   - After: FP16 storage accumulates through entire prefill, no trigger

2. `integration.py` — Trigger bulk-quantize at first decode step:
   - Before: no such logic
   - After: `if L_q == 1 and not cache._quantized and cache._fp16_keys is not None: cache._bulk_quantize()`

**Expected outcome:**
- Prefill: O(N) — FP16 SDPA throughout, no dequantize step
- First decode step: one-shot bulk-quantize (adds ~0.5s, amortized over generation)
- Decode tps: unchanged (same fused PQ SDPA path as before)
- 64K prefill: ~2-3 min vs >112 min (>50x improvement)

**Benchmark:** `benchmarks/exp10_lazy_prefill.py`
- Measures prefill time + decode tps at 8K / 32K / 64K
- Runs FP16 baseline and PQ-lazy back-to-back
- Target: Mini (Qwen3.5-35B-A3B-4bit, M4 Pro 64GB)

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

## Experiment 6: STA/LTA Change-Point Detection for Entropy Amortization

**Concept:** Phase 2a computes per-head Shannon entropy on EVERY decode step to set adaptive sparse V thresholds. Inspired by AAPM's `seismic_detection.py`, use STA/LTA ratio to detect when attention patterns shift, and only recompute entropy at those points.

**Key results (2K context, 500 decode steps, 2 regime transitions):**

| Strategy | Entropy Computes | Reduction | Cos Sim | Time | Speedup |
|---|---|---|---|---|---|
| Always recompute | 500/500 | 0% | 0.983016 | 1.01s | 1.00x |
| STA/LTA (best) STA=3 LTA=20 t=1.5 | 31/500 | 93.8% | 0.983016 | 0.52s | 1.94x |
| Fixed interval=25 | 20/500 | 96.0% | 0.983016 | 0.45s | 2.24x |
| **Fixed interval=50** | **10/500** | **98.0%** | **0.983016** | **0.44s** | **2.30x** |

**Critical finding:** All strategies produce **identical quality** (cos=0.983016). The entropy-to-threshold sigmoid mapping is regime-robust — cached thresholds work across both concentrated and mixed attention regimes without degradation.

**Why STA/LTA lost to fixed interval:**
- STA/LTA with trigger >= 2.0 misses transitions entirely (stat changes by 2x, not enough for ratio-based detection)
- STA/LTA with trigger 1.5 catches transitions but fires too often (more computes)
- Quality is identical regardless of strategy, making transition detection irrelevant
- Fixed interval is simpler, faster (no per-step stat computation), and equally effective

**Status:** Ready for integration. Use fixed interval=50 recomputation in `fused_sdpa()`.

---

## Experiment 9: Erlang Queuing Model for Sparse V Workload Prediction

**Concept:** The SV kernel iterates ALL L_kv positions to check `if (|wn_val| > threshold)` even when concentrated heads skip ~99%. Inspired by AAPM's `queuing/erlang_c.py`: can we predict HOW MANY positions will exceed the threshold from a cheap sample, enabling dispatch decisions (skip/sparse/dense) without scanning everything?

**Three models tested:**
1. **Top-k exponential tail:** Sample top-k weights (O(L_kv) partial sort), fit exponential decay, extrapolate to predict count above threshold
2. **Erlang utilization:** Map mean weight as arrival rate, threshold as service rate; P(active) = exp(-threshold/mean)
3. **Hybrid:** Top-k calibrates Erlang tail model

**Key results (16K context, 5 distributions, 5 thresholds):**

| Model | Mean Error | Cases <20% Error | Dispatch Accuracy |
|---|---|---|---|
| Top-k (k=100) | 32.2% | 68% | 89.5% |
| Erlang utilization | 13,166% | 16% | 79.5% |
| **Hybrid** | **32.1%** | **68%** | **90.0%** |

**Prediction accuracy is strong for concentrated/spread distributions (0% error) but poor for bimodal (131-246% error).** The exponential tail assumption breaks when the distribution has multiple modes.

**Critical finding: prediction is 8-12x MORE EXPENSIVE than full scan.** At L_kv=16K, the full threshold check takes ~141us on Metal. Top-k prediction takes ~1,752us. Even at L_kv=65K, the scan stays ~134us while prediction grows to ~2,146us. The Metal GPU's branch check is so cheap that prediction overhead cannot amortize it.

**Why it failed on cost:** The SV kernel's `if (wn_val > threshold)` check is a single branch instruction per position, executed entirely on the GPU. The prediction models require CPU-side numpy/MLX ops (topk, sort, mean) with Python loop overhead. The CPU-GPU data transfer alone exceeds the GPU's scan cost.

**Where it WOULD help:** If dispatch decisions could be made in a Metal kernel preamble (no CPU round-trip), or at extreme context lengths (>1M tokens) where scan cost grows linearly but tail estimation stays O(k).

**Status:** Partial. Accurate for dispatch decisions (90%) but not cost-effective at current context lengths.

---

## Unexplored Avenues (from AAPM exotic library)

Documented for future investigation. Each has a plausible PolarQuant angle but was not tested.

| AAPM Module | Concept | PolarQuant Angle | Priority |
|---|---|---|---|
| `soc_predictor.py` | Power-law distributions, critical slowing down | Attention weights follow approximate power laws. Estimate exponent α from top-k to predict optimal sparse threshold analytically — could replace entropy entirely with a cheaper estimator. | **Medium** — Exp 9 showed top-k prediction is accurate (90% dispatch), just too expensive in Python. A Metal kernel preamble version could work. |
| `propulsion_zones.py` | Negative viscosity, constraint alignment | Identify contiguous position RANGES where attention is concentrated (recency bias). Process as vectorized blocks instead of per-position checks — better memory coalescing in SV kernel. | **Medium** — directly attacks the 88% SV bottleneck via memory access pattern, not pruning. Different angle from all other experiments. |
| `creep_fatigue.py` | Larson-Miller parameter, Miner's cumulative damage | Track cumulative quantization error over conversation lifetime. Predict time-to-failure under sustained load. | **Low** — Exp 3 showed no drift. Cumulative damage model has no target. |
| `defense_in_depth.py` | 5-level nuclear safety, N+2 redundancy | Formalize existing quality gates (entropy → threshold → rigidity) as layered defense levels. Ensure each operates independently. | **Low** — we already have layered gates. Formalizing adds indirection, not capability. |
| `recovery_distance.py` | Min-edit graph distance to feasibility | After aggressive quantization (2-bit zone), compute minimum number of position upgrades needed to restore target quality. Could inform zone boundary placement. | **Low** — interesting for Exp 7 zone tuning but incremental. |
| `burnout_epidemiology.py` / `contagion_model.py` | SIR epidemic model | Model quantization error "spreading" through transformer layers. If one layer's error infects downstream layers, early layers need higher precision. | **Low** — Exp 3 found no error accumulation. Layer-to-layer infection hypothesis unsupported. |
| `circadian_model.py` | Circadian rhythms | Time-of-day adaptive precision: clinic hours → higher quality (4-bit), sleeping → aggressive compression (2-bit). Governor already handles presence-based shedding. | **Low** — governor + context-update already does this at the service level. Per-bit-width adaptation is novel but marginal. |

**Most promising unexplored:** SOC power-law (as Metal kernel preamble) and propulsion zones (contiguous-range vectorized access). Both attack the SV kernel bottleneck from angles not covered by existing experiments.

---

## Combination Analysis

### What works together

| Combination | Viable? | Rationale |
|---|---|---|
| Entropy (1) + Rigidity (2) | **Yes — SHIPPED** | Different pipeline stages, no interference |
| Entropy (1) + Amortization (6) | **Yes — ready** | 6 reduces cost of 1 by 98%; direct integration path |
| Entropy (1) + Zones (7) | **Yes — ready** | 7 changes quantization policy, 1 changes attention kernel. Independent. |
| Amortization (6) + Zones (7) | **Yes** | Both ready, no overlap. 6 is attention-side, 7 is storage-side. |
| Entropy (1) + Stroboscopic (3) | Monitoring only | No drift to correct, useful as safety net |
| Hub (5) + anything | **No** | Hub-ness is query-dependent, not stable |
| Spectral (4) + anything | **No** | Rotation decorrelation makes pattern-based bit selection ineffective |
| SPC (8) + anything | **No** | Per-step noise from query randomness makes control charts impractical |
| Erlang (9) + Metal preamble | **Maybe (future)** | Prediction is accurate but CPU cost kills it. Metal-native version could work at >1M context |
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
| `benchmarks/exp5_hub_tokens.py` | Experiment 5 script |
| `benchmarks/EXP5_RESULTS.md` | Experiment 5 detailed results |
| `benchmarks/exp6_sta_lta.py` | Experiment 6 script |
| `benchmarks/EXP6_RESULTS.md` | Experiment 6 detailed results |
| `benchmarks/exp7_blast_radius_zones.py` | Experiment 7 script |
| `benchmarks/EXP7_RESULTS.md` | Experiment 7 detailed results |
| `benchmarks/exp8_spc_quality.py` | Experiment 8 script |
| `benchmarks/EXP8_RESULTS.md` | Experiment 8 detailed results |
| `benchmarks/exp9_erlang_workload.py` | Experiment 9 script |
| `benchmarks/EXP9_RESULTS.md` | Experiment 9 detailed results |
| `benchmarks/stress_test_long_context.py` | 16K-32K stress test |
| `benchmarks/STRESS_RESULTS.md` | Stress test results |
| `benchmarks/EXPERIMENTS.md` | This consolidated document |
