# PolarQuant Metal: Fused Metal Kernels for PolarQuant KV Cache

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/3ud41mon14/polarquant-metal)

Custom Metal kernels that eliminate the dequantize-on-fetch bottleneck in PolarQuant (TurboQuant) KV cache implementations on Apple Silicon.

## The Problem

PolarQuant compresses KV cache vectors using random orthogonal rotation + Lloyd-Max codebook quantization. The current MLX implementations (rachittshah/mlx-turboquant, ml-explore/mlx-lm PR #1059) achieve excellent compression (~4.6x at 3-bit) but suffer a **0.5x decode speed penalty** because they dequantize the entire KV cache on every attention step before computing Q@K^T and weights@V.

## The Solution

Two fused Metal kernels that compute attention scores and output **directly from packed quantized indices**, avoiding full dequantization:

1. **`polarquant_qk_matmul`**: Fused Q @ K^T
   - Pre-rotates queries into PolarQuant basis
   - Iterates over head_dim, unpacking indices and doing codebook lookups on-the-fly
   - Accumulates the dot product in float32 for precision
   - Scales by key norms and attention scale

2. **`polarquant_sv_matmul`**: Fused softmax(scores) @ V
   - Same approach for value side: codebook lookup during weighted sum
   - Output is in the rotated value basis; inverse-rotated after

### Why This Works

The key insight: if queries are pre-rotated into the PolarQuant key basis (`Q_rot = Q @ R^T`), then:

```
score[i,j] = Q_rot[i] · centroids[indices[j]] * norm[j] * scale
```

No inverse rotation of keys needed. The codebook has only 2^bits entries (8 for 3-bit), so lookups are trivially cheap. The kernel fuses unpack → lookup → multiply-accumulate into a single pass per output element.

## Install

```bash
git clone http://100.69.127.98:3030/aaron/polarquant-metal.git
cd polarquant-metal
python3 -m venv .venv
./.venv/bin/pip install -e '.[dev]'
```

## Quick Start

```python
import mlx.core as mx
import numpy as np
from polarquant_metal import FusedPolarQuantKVCache

# Create cache
cache = FusedPolarQuantKVCache(bits=3, head_dim=128)

# Store keys and values
keys = mx.random.normal((1, 8, 32, 128))     # B, n_kv_heads, L_kv, D
values = mx.random.normal((1, 8, 32, 128))
cache.update_and_fetch(keys, values)

# Compute attention with fused kernels (no dequantization!)
queries = mx.random.normal((1, 32, 1, 128))   # B, n_heads, L_q, D
output = cache.fused_attention(queries)
```

## Integration with mlx-lm

```python
import mlx_lm
from polarquant_metal.integration import make_fused_cache

model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-35B-A3B-4bit")
cache = make_fused_cache(model, bits=3)
response = mlx_lm.generate(model, tokenizer, prompt="...", prompt_cache=cache)
```

### How it works

`make_fused_cache()` handles everything automatically:
- **Hybrid models** (Qwen3.5): detects `is_linear` layers, uses `ArraysCache` for linear attention and `TurboQuantKVCache` for standard attention
- **Lazy quantization**: stores FP16 until `min_fused_context` (default 512 tokens), then bulk-quantizes. Zero overhead for short conversations.
- **SDPA dispatch**: prefill (L_q > 1) uses standard FP16 attention. Decode (L_q == 1) uses fused Metal kernels after threshold.

### Performance (Qwen3.5-35B-A3B-4bit, M4 Pro)

| Context | Speed vs Standard | Memory |
|---------|------------------|--------|
| <512 tokens | 1.0x (identical) | FP16 |
| 600+ tokens | 0.97x | 4.6x compressed |
| Decode at 2K | 2.0x vs naive dequant | 4.6x compressed |

### Adaptive Optimizations (Phase 2, 2026-03-31)

Two runtime optimizations that activate automatically on long contexts:

**Entropy-guided sparse V** — per-head adaptive pruning in the SV kernel:
- Computes Shannon entropy of attention weights per head after softmax
- Concentrated heads (low entropy) → aggressive threshold, skip ~99% of value lookups
- Spread heads (high entropy) → threshold disabled, full computation
- 3x SV kernel speedup on concentrated heads, zero quality loss on spread heads
- Activates at >1024 tokens (97% of production requests)

**Rigidity gate** — skip redundant KV quantization during decode:
- Compares consecutive tokens' rotated unit vectors via cosine similarity
- When rigidity > 0.90 (tokens would produce ~identical codebook indices), reuses previous packed indices with updated norm only
- 78% skip rate on flowing text, 0% on topic changes (correctly gated)
- `cache.rigidity_stats` for observability

### Compact-Index Sparse SV (Phase 3, 2026-03-31)

Two-phase Metal kernel that eliminates the O(L_kv) iteration bottleneck:

1. **Index build kernel:** Scans `wn_combined` per head, collects active positions via Metal atomics into a compact index. Positions exceeding the per-head threshold OR marked as zone priors (system prompt, recent tokens) are included.
2. **Sparse SV kernel:** Iterates only active positions from the compact index. O(active_count) instead of O(L_kv).

| Context | Phase 2 (branch/position) | Phase 3 (compact index) | Speedup |
|---------|--------------------------|------------------------|---------|
| 2K | 0.98ms | 0.45ms | 2.2x |
| 8K | 0.82ms | 0.33ms | 2.5x |
| 16K | 1.36ms | 0.40ms | 3.4x |
| 32K | 2.48ms | 0.44ms | 5.6x |

- Entropy threshold amortized every 50 decode steps (zero quality loss)
- Index build cost ~0.15ms constant regardless of context length
- Never slower than dense path in any configuration tested
- Zone prior infrastructure available (`system_prompt_len`, `recent_zone_len` params)

## Integration with mlx-turboquant

```python
from polarquant_metal.mlx_turboquant_adapter import FusedTurboQuantKVCache
from polarquant_metal.integration import patch_sdpa

# Drop-in replacement for mlx-turboquant's TurboQuantKVCache
patch_sdpa()
cache = [FusedTurboQuantKVCache(bits=3, head_dim=128) for _ in range(num_layers)]
```

## Tests

```bash
python tests/test_kernels.py
```

## Benchmarks

```bash
python benchmarks/bench_fused_vs_naive.py
```

## Architecture

```
polarquant_metal/
├── __init__.py              # Public API
├── kernels.py               # Metal kernel source + Python wrappers (per-head sparse_thresh[])
├── cache.py                 # FusedPolarQuantKVCache
├── turboquant_cache.py      # TurboQuantKVCache (entropy gate + rigidity gate)
├── polar_quant.py           # PolarQuant quantizer (rotation + codebooks)
├── packing.py               # Bit-packing utilities
├── codebooks.py             # Lloyd-Max codebooks (hardcoded, no file dependency)
├── integration.py           # mlx-lm SDPA monkey-patch
└── mlx_turboquant_adapter.py # Drop-in for rachittshah/mlx-turboquant

benchmarks/
├── bench_fused_vs_naive.py        # Core benchmark (decode + prefill, bit widths)
├── stress_test_long_context.py    # 16K-32K decode stress test
├── exp1_entropy_sparse_v.py       # Experiment 1: entropy-guided threshold
├── exp2_rigidity_gate.py          # Experiment 2: anti-churn rigidity
├── exp3_stroboscopic_drift.py     # Experiment 3: drift detection (negative result)
├── exp4_spectral_bitwidth.py      # Experiment 4: per-head bit-width (negative result)
├── EXPERIMENTS.md                 # Consolidated experiment findings
├── STRESS_RESULTS.md              # 16K-32K results
└── EXP[1-4]_RESULTS.md            # Individual experiment results
```

## How the Metal Kernels Work

### Q@K^T Kernel (per-element)

Each thread computes one element `out[b, h, q, k]`:

```metal
float acc = 0.0f;
for (uint d = 0; d < D; d++) {
    float q_val = queries[q_offset + d];
    uint idx = unpack_index<BITS>(&packed_keys[k_offset], d);  // bit-unpack
    float k_val = centroids[idx];                               // codebook lookup
    acc += q_val * k_val;                                       // MAC
}
acc *= norms[k_idx] * scale;  // apply key norm + attention scale
```

### Performance Characteristics

**Memory savings** (same as PolarQuant):
- 3-bit: ~4.6x compression vs FP16
- 4-bit: ~3.8x compression vs FP16

**Speed** (the improvement this kernel provides):
- Eliminates the full dequantize pass (D × L_kv × n_kv_heads values)
- Codebook is tiny (8 entries for 3-bit) — fits in register/L1
- Pre-rotation of queries is O(L_q × D²) — negligible for decode (L_q=1)

## Supported Configurations

- Bit widths: 2, 3, 4
- Head dimensions: any (tested with 64, 128)
- GQA: fully supported (n_heads != n_kv_heads)
- Dtypes: float32, float16, bfloat16

## MLX Server Integration (LIVE)

PolarQuant is deployed on the OpenClaw MLX server (Qwen3.5-35B-A3B-4bit on port 8080).

### Setup

1. Install in server venv: `~/.mlx-server-env/bin/pip install -e ~/workspace/polarquant-metal/`
2. Patch `~/.mlx-server-env/.../app/models/mlx_vlm.py` — add PolarQuant cache creation in `MLX_VLM.__call__` (env-var gated)
3. Add `POLARQUANT_KV=1` to `~/Library/LaunchAgents/ai.mlx.server.plist` EnvironmentVariables
4. Restart: `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.mlx.server.plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.mlx.server.plist`

### Key details

- **Model**: `mlx-community/Qwen3.5-35B-A3B-4bit` loaded via `mlx_vlm` (not `mlx_lm`)
- **Cache target**: `model.language_model` (the text decoder within the VLM wrapper)
- **SDPA patching**: Must scan both `mlx_lm.models.*` AND `mlx_vlm.models.*` — VLM models import SDPA from `mlx_lm.models.base` but store copied references in VLM module namespace
- **Rollback**: Set `POLARQUANT_KV=0` in plist, restart. Backups at `~/.openclaw/backups/*.pre-polarquant`

### Client impact

None. Koa's text-router sends standard OpenAI-compatible requests. PolarQuant is entirely server-side.

## Repository & Research

**Source**: Private Gitea at `http://192.168.4.202:3030/aaron/polarquant-metal` (LAN) or `http://100.69.127.98:3030/aaron/polarquant-metal` (Tailscale). Branch `master`.

**Research references**: Background papers, MLX issue threads, and prior art documents are stored in DEVONthink under the `Research_Capture` database, tagged `polarquant`. The `docs/` directory in this repo contains synthesized notes derived from that material.

## Limitations

1. **Prefill is slow** — the per-element kernel can't parallelize across L_q > 1. Prefill falls back to standard FP16 SDPA automatically. Fused kernels only benefit decode (L_q=1).

2. **No QJL residual correction** — handles Stage 1 (PolarQuant) only. Adding QJL would require an additional correction term in the Q@K^T kernel.

3. **Model-dependent quality** — Llama-3.2-3B and Qwen3.5-35B produce correct output. Phi-4-Mini degrades (PolarQuant itself produces low cosine similarity on this architecture — same issue in upstream PR #1059).

4. **SV kernel is the bottleneck** — 47% of time at 2K tokens, 88% at 32K. Tiled SV and simd_broadcast_first were both tested and found slower than the simple per-element kernel (Metal L1 cache handles broadcast efficiently). Pre-combined weight*norm optimization gives 25% improvement. Entropy-guided Sparse V (Phase 2) mitigates this for concentrated heads.

## Negative Results (things we tested that didn't work)

Documented here for completeness. See `benchmarks/EXPERIMENTS.md` for full data.

1. **Stroboscopic FP16 drift detection** — Hypothesis: quantization error accumulates over long contexts. Result: **No drift.** Cosine similarity stays >0.998 across 16K tokens. Re-quantizing from FP16 ground truth produces byte-identical output. Law of large numbers: softmax averaging dilutes per-token errors as context grows. Consistent with TurboQuant's [0.997 NIAH score at 104K context](https://arxiv.org/abs/2504.19874) and [flovflo/turboquant-mlx](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv) reporting 100% exact match across 8.5K-64K tokens. Conclusion: no recalibration mechanism needed.

2. **Spectral bit-width selection** — Hypothesis: heads with periodic attention patterns tolerate 2-bit quantization. Result: **No pattern survived 2-bit.** PolarQuant's random orthogonal rotation decorrelates signals before quantization, making error pattern-independent (same mechanism as [QuaRot](https://arxiv.org/abs/2404.00456), NeurIPS 2024). The best 2-bit cosine similarity was 0.946 (periodic-64), below the 0.95 safety threshold. Uniform 3-bit is the correct default.

3. **Adaptive codebook learning (stigmergy)** — Hypothesis: online codebook adaptation via usage-frequency reinforcement. Result: **Not novel.** This is standard online k-means / EMA codebook updates (VQ-VAE, 1967). Additionally, PolarQuant's rotation makes distributions approximately Gaussian, for which Lloyd-Max is already optimal. Would not improve quality.

4. **Fixed sparse_v_threshold** — A uniform threshold of 0.01 achieves 3x speedup on concentrated heads but **destroys** spread heads (cosine sim = 0.000). This motivated the entropy-guided approach (Experiment 1), which correctly disables pruning for high-entropy heads.

## Prior Art & Novel Contributions

Novelty assessment grounded in [Perplexity deep research](https://huggingface.co/spaces/3ud41mon14/polarquant-metal) (2026-03-31). Claims ordered by strength.

### Primary Contributions (no prior art found)

1. **Entropy-guided per-head adaptive Sparse V** -- Shannon entropy of post-softmax attention weights gates per-head pruning threshold at runtime. Concentrated heads pruned aggressively, spread heads protected. No prior work applies entropy-gated thresholds to sparse attention on any platform. Related: [HIES](https://arxiv.org/abs/2410.10165) (NeurIPS 2025 workshop) uses entropy for offline head pruning; [arXiv 2501.03489](https://arxiv.org/abs/2501.03489) uses entropy for training-time regularization -- neither for runtime decode-time skip gating.
2. **Rigidity-gated quantization skip** -- cosine similarity of consecutive rotated unit vectors detects redundant KV entries; skips quantize+pack when codebook indices would be identical. 78% skip rate on smooth text, 0% on topic changes. No direct prior art. Related: [CosineGate](https://arxiv.org/abs/2411.09967) (NeurIPS 2025) uses cosine incompatibility to skip ResNet blocks; Token Filtering (Dec 2025) uses KV cosine for layer-level skip -- neither operates on per-token quantization.
3. **Asymmetric K/V bit-widths in MLX** -- different quantization bits for keys vs values. First MLX-native implementation. Concept from [KIVI](https://arxiv.org/abs/2402.02750) (ICML 2024) and [KVSplit](https://github.com/dipampaul17/KVSplit) (llama.cpp, May 2025). [mlx-lm issue #191](https://github.com/ml-explore/mlx-lm/issues/191) discussed but never merged.

### Incremental / Concurrent Contributions

4. **Fused bidirectional Metal kernels** (QK + SV) -- both Q@K^T and scores@V computed directly from packed codebook indices. QK side independently implemented by oMLX v0.2.21 and mlx-lm PR #1067. SV side is the natural extension; oMLX's fused 2-pass kernel may already cover it.
5. **Sparse V on Apple Silicon** -- threshold-based skipping of near-zero attention positions in Metal SV kernel. [SpargeAttn](https://github.com/thu-ml/SpargeAttn) (ICML 2025) on CUDA. [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) independently proposed attention-gated value dequantization on Metal (March 24, 2026) -- simultaneous independent work.
6. **Combined pipeline** -- fused attention + entropy-guided Sparse V + rigidity gate + asymmetric K/V + lazy threshold on M4 Pro. Novel as an integrated system; individual components have varying novelty.

**Result:** 75.3 tok/s vs 71.4 tok/s standard (5% faster than FP16 with 8x KV cache compression). Phase 2 adds per-head adaptive gains on top.

### Prior Art (not novel to this project)

| Technique | Prior Work |
|-----------|-----------|
| Fused QK codebook kernels on Metal | oMLX v0.2.21 and mlx-lm PR #1067 independently implemented |
| Lazy quantization (FP16 prefill, quantize at decode) | oMLX and PR #1067 converged on same pattern |
| Sparse V concept | SpargeAttn (ICML 2025, Tsinghua) on CUDA |
| Asymmetric K/V concept | KIVI (ICML 2024) on CUDA; extended by PackKV (Dec 2025) |

### Novel Contributions (first on Metal/MLX)

1. **Fused SV kernel** -- scores@V directly from packed codebook indices on Metal. No public Metal implementation exists.
2. **Sparse V on Apple Silicon** -- threshold-based skipping of near-zero attention positions in Metal kernel. SpargeAttn exists on CUDA but has no Metal port.
3. **Asymmetric K/V in MLX ecosystem** -- different bitwidths for K vs V caches. Nothing in the MLX ecosystem uses this.
4. **Entropy-guided per-head adaptive Sparse V** -- per-head Shannon entropy gates pruning threshold at runtime. Concentrated heads pruned aggressively, spread heads protected. No prior work applies entropy-gated sparse attention to Metal/MLX codebook kernels.
5. **Rigidity-gated quantization skip** -- cosine similarity of rotated unit vectors detects redundant KV entries and skips quantize+pack. Novel application of anti-churn metrics to KV cache compression.
6. **Compact-index sparse SV with atomic index build** -- two-phase kernel: Metal atomics build position index, sparse kernel iterates only active positions. 5.6x at 32K. No prior Metal implementation of atomic-indexed sparse attention.
7. **Combined pipeline** -- fused attention + entropy-adaptive Sparse V + compact-index dispatch + rigidity gate + asymmetric K/V + amortized entropy + zone priors on M4 Pro.

**Result:** 75.3→81 tok/s on Qwen3.5-35B (vs 71.4 standard). Phase 3 SV kernel 5.6x faster at 32K. 9 cross-disciplinary experiments documented (4 wins, 4 negatives, 1 partial).

### Key References

- [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) -- Zandieh et al., ICLR 2026
- [SpargeAttn](https://github.com/thu-ml/SpargeAttn) -- ICML 2025, sparse warp online softmax (CUDA)
- [KIVI](https://arxiv.org/abs/2402.02750) -- ICML 2024, per-channel K / per-token V quantization (CUDA)
- [PackKV](https://arxiv.org/abs/2412.03631) -- Dec 2025, extends KIVI
- [AsymKV](https://aclanthology.org/2025.coling-main.576/) -- COLING 2025, 1-bit V with higher-bit K
- [KVSplit](https://github.com/dipampaul17/KVSplit) -- May 2025, K8V4 for llama.cpp on Apple Silicon
- [QuaRot](https://arxiv.org/abs/2404.00456) -- NeurIPS 2024, rotation-based outlier elimination
- [oMLX v0.2.21](https://github.com/jundot/omlx) -- March 2026, fused 2-pass Flash Attention (Metal)
- [mlx-lm PR #1067](https://github.com/ml-explore/mlx-lm/pull/1067) -- March 2026, fused Metal quantize/dequantize
- [mlx-lm Issue #191](https://github.com/ml-explore/mlx-lm/issues/191) -- Asymmetric K/V discussion
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) -- March 2026, sparse-V on Metal (concurrent)
- [CosineGate](https://arxiv.org/abs/2411.09967) -- NeurIPS 2025, cosine-gated residual block skipping
- [HIES](https://arxiv.org/abs/2410.10165) -- NeurIPS 2025 workshop, entropy-based head importance
- [Entropy-Guided Attention (arXiv:2501.03489)](https://arxiv.org/abs/2501.03489) -- Jan 2025, headwise entropy regularization

## License

MIT

## References

- [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) — Zandieh et al., ICLR 2026
- [flovflo/turboquant-mlx-qwen35-kv](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv) — MLX TurboQuant test data (100% recall at 64K)
- [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) — Paper-faithful MLX implementation
- [ml-explore/mlx-lm Issue #1060](https://github.com/ml-explore/mlx-lm/issues/1060) — Upstream tracking
- [MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) — MLX kernel API docs
- [QuaRot (arXiv:2404.00456)](https://arxiv.org/abs/2404.00456) — NeurIPS 2024, rotation-based outlier elimination
- [Interactive benchmarks & novelty assessment](https://huggingface.co/spaces/3ud41mon14/polarquant-metal) — HuggingFace Space
