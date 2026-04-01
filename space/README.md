---
title: PolarQuant Metal
emoji: ⚡
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: "6.10.0"
app_file: app.py
pinned: false
license: mit
short_description: Fused Metal kernels for PolarQuant KV cache on Apple Silicon
---

# PolarQuant Metal

Fused Metal kernels that eliminate the dequantize-on-fetch bottleneck in PolarQuant (TurboQuant) KV cache on Apple Silicon.

**75.3 tok/s vs 71.4 tok/s standard** — 5% faster than FP16 with 8x KV cache compression on Qwen3.5-35B (M4 Pro, 3-bit).

[GitHub Repository](https://github.com/Euda1mon1a/polarquant-metal)

---

## Phase 3 Update: Compact-Index Sparse SV + Experiments 5–9

### The Problem Phase 2 Left Unsolved

Phase 2 introduced entropy-guided Sparse V, skipping 99% of value lookups on concentrated attention heads. But the Metal kernel still *iterated* all L_kv positions to find them:

```metal
for (uint k = 0; k < L_kv; k++) {   // 16,384 iterations at 16K
    if (abs(wn_val) > threshold) {   // branch per position
        acc += wn_val * centroid;    // actual work: ~164 positions
    }
}
```

At 32K context, 32,600 iterations execute only the branch. This is the same bottleneck Block Sparse FlashAttention (arXiv:2512.07011) was designed to solve on CUDA — wasted loop overhead when sparsity is high.

### Two-Phase Compact-Index Dispatch

The fix is a well-established GPU pattern on CUDA, ported here to Metal for the first time.

**Kernel 1 — Index build** (`polarquant_sv_index_build`): One pass over the combined weight×norm array. Per-thread atomic-write into a compact `active_indices` buffer. Cost: ~130µs flat, independent of attention pattern. At 16K: 64 threadgroups, one compare + conditional atomic per thread.

**Kernel 2 — Sparse SV** (`polarquant_sv_sparse`): Iterate only `active_count` positions:

```metal
uint count = active_count[b * n_heads + h];
for (uint i = 0; i < count; i++) {
    uint k = active_indices[base + i];   // O(active) not O(L_kv)
    acc += wn_combined[k] * centroids[unpack_index(v_indices, k, d)];
}
```

### Benchmark Results (B=1, GQA 4:1, 3-bit, M4 Pro)

| L_kv | Pattern | Speedup | Active % | Cos Sim |
|-----:|:--------|--------:|--------:|--------:|
| 2,048 | concentrated | 2.16x | 3.4% | 1.000000 |
| 8,192 | concentrated | 2.49x | 1.8% | 1.000000 |
| 16,384 | concentrated | 3.39x | 1.1% | 1.000000 |
| 32,768 | concentrated | **5.61x** | 0.8% | 1.000000 |
| 32,768 | moderate | 2.92x | 5.9% | 1.000000 |
| 8,192 | spread | 1.25x | 99.6% | 1.000000 |

Never slower than dense across all 15 tested configurations. Sparse beats dense even at 99.6% active (8K spread), because the flat index-build cost is dominated by savings in the main loop at any sparsity level.

---

## What Shipped in Phase 3

### Entropy Amortization (Exp 6)

Entropy-guided thresholds are now amortized: recomputed once every 50 decode steps, cached between. Inspired by STA/LTA change-point detection from seismology — attention patterns drift slowly between consecutive tokens, and per-head entropy stays in the same regime until a topic shift occurs.

Result: **98% reduction in entropy compute overhead**, zero quality impact (cosine sim unchanged across 500 decode steps with two forced regime transitions). The deterministic 50-step interval was simpler and more robust than full STA/LTA; STA/LTA is available for adversarial inputs.

### Zone-Tiered Quantization (Exp 7)

System prompt and recent tokens receive 4-bit precision; early and mid-context receive 3-bit. Inspired by the "blast radius isolation" pattern: quantization error in one zone stays contained within that zone.

| Strategy | Cosine Sim | Memory | vs uniform 3-bit |
|:---------|----------:|-----------:|----------------:|
| uniform 3-bit | 0.98183 | 1728 KB | baseline |
| tiered conservative | 0.99540 | 1781 KB | +3% memory |

System-prompt zone specifically: 0.981 → 0.996 cosine sim. The +3% memory overhead is correct for a ~700-token protected zone at 16K context (700/16384 × 1.33× premium ≈ +53KB).

This is positional zoning (within a single context window), distinct from depth-layer strategies like PM-KVQ (ICLR 2026), which assign bit-widths per transformer block. Per-layer zoning in MLX is the closest prior we found.

### Zone Priors in the Index Build

System prompt and recent token positions are always added to the `active_indices` compact index, regardless of threshold. This ensures critical positions are never pruned while preserving sparsity on bulk mid-context tokens.

---

## Experiments That Didn't Ship

**Exp 5 — Hub Token Protection** (dropped): Entropy-guided thresholds already protect attention sinks (system prompt tokens) naturally — they maintain high attention weight across all heads and would never be below threshold anyway. Forced positional locks reduced effective sparsity without quality benefit (2.21x vs 3.37x). Zone priors handle the edge case correctly.

**Exp 8 — SPC Quality Control Charts** (abandoned): Statistical process control requires a stationary process. Attention weights are query-dependent and non-stationary by design — control limits would need per-query recalibration, costing as much as the quality measurement itself.

**Exp 9 — Workload Prediction** (deferred): Top-k sampling achieves 0% prediction error on concentrated/spread distributions and 90% dispatch accuracy on power-law (realistic transformer distributions). Problem: Python dispatch overhead (~1ms) exceeds the kernel cost at current context lengths. A Metal-native kernel preamble — a few threads sampling top-k before the main dispatch — is the sound path forward at context lengths >1M tokens.

---

## Current Performance (Qwen3.5-35B-A3B-4bit, M4 Pro, 3-bit KV)

| Stack | Speed | Memory |
|:------|------:|-------:|
| Standard mlx-lm | 71.4 tok/s | FP16 |
| Phase 1 (fused QK+SV) | 75.3 tok/s | 4.6x compressed |
| Phase 2 (+ entropy sparse V + rigidity gate) | ~79 tok/s* | 4.6x compressed |
| Phase 3 (+ compact-index SV + amortization + zone tiers) | ~82 tok/s* | 4.6x compressed |

*Decode-path estimates at long context; end-to-end Phase 2+3 benchmark pending.

---

## Architecture Overview

```
polarquant_metal/
├── kernels.py              # Metal kernels: QK, SV dense, SV index-build, SV sparse
├── turboquant_cache.py     # TurboQuantKVCache: entropy amortization, rigidity gate,
│                           #   zone-tiered quantization, Phase 3 sparse dispatch
├── polar_quant.py          # PolarQuant quantizer (rotation + Lloyd-Max codebooks)
├── integration.py          # mlx-lm SDPA patch
└── ...

benchmarks/
├── EXPERIMENTS.md          # All 9 experiments: findings, combinations, what to try next
├── PHASE3_RESULTS.md       # Compact-index sparse SV full results table
├── EXP[1-9]_RESULTS.md     # Individual experiment data
└── ...
```

## Key References

- [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) — Zandieh et al., ICLR 2026
- [Block Sparse FlashAttention (arXiv:2512.07011)](https://arxiv.org/abs/2512.07011) — two-stage CUDA analog for compact-index sparse dispatch
- [SpargeAttn](https://github.com/thu-ml/SpargeAttn) — ICML 2025, sparse warp online softmax (CUDA)
- [KIVI (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750) — ICML 2024, asymmetric K/V quantization
- [PM-KVQ](https://arxiv.org/abs/2502.01709) — ICLR 2026, layer-depth zone quantization (different axis than this work)
- [KVQuant](https://arxiv.org/abs/2401.18079) — per-channel key quantization with positional sensitivity
- [QuaRot (arXiv:2404.00456)](https://arxiv.org/abs/2404.00456) — NeurIPS 2024, rotation-based outlier elimination
- [oMLX v0.2.21](https://github.com/jundot/omlx) — concurrent fused 2-pass Metal attention
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — concurrent sparse-V on Metal (March 2026)
