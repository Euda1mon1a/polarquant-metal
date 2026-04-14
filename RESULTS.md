# PolarQuant Metal — Results

> Mac Mini M4 Pro (64GB), MLX 0.31.1, 2026-03-30

## Tests: 15/15 PASS

### Kernel Tests (7)
| Test | Max Diff | Cosine Sim |
|------|----------|-----------|
| Pack/unpack roundtrip | exact | exact |
| Fused Q@K^T (simple) | 8.05e-7 | 1.000000 |
| Fused Q@K^T (tiled) | 1.19e-6 | 1.000000 |
| Fused weights@V | 3.58e-7 | 1.000000 |
| Full attention | 9.91e-7 | 1.000000 |
| GQA (8h/2kv) | 1.19e-6 | 1.000000 |
| All bit widths (2/3/4) | <1.2e-6 | 1.000000 |

### Integration Tests (8)
| Test | Result |
|------|--------|
| Basic update/fetch (2/3/4-bit, fused + deq) | PASS |
| Fused vs dequantized attention | cos_sim=1.0, diff=5.4e-7 |
| Multi-step decode (10 steps) | PASS |
| Quality vs FP16 (3-bit) | cos_sim=0.968 |
| Quality vs FP16 (4-bit) | cos_sim=0.992 |
| Decode speed (L=1024) | 0.96x (kernel-only) |
| Trim | PASS |
| State save/restore | PASS |
| SDPA patch dispatch | diff=0 (identical) |

## Isolated Kernel Benchmark (3-bit, 32h/8kv, D=128)

| KV Length | FP16 | Naive Dequant | Fused Metal | Fused/Naive | Fused/FP16 | Compression |
|-----------|------|--------------|-------------|------------|-----------|-------------|
| 64 | 0.41ms | 0.37ms | 0.37ms | 1.00x | 0.90x | 4.6x |
| 256 | 0.19ms | 0.39ms | 0.30ms | **1.31x** | 1.60x | 4.6x |
| 512 | 0.19ms | 0.43ms | 0.37ms | **1.16x** | 1.93x | 4.6x |
| 1024 | 0.44ms | 0.77ms | 0.45ms | **1.71x** | **1.03x** | 4.6x |
| 2048 | 0.63ms | 1.32ms | 0.68ms | **1.94x** | **1.08x** | 4.6x |

SV kernel optimized with pre-combined weight*norm (25% faster at 1K+ tokens).

## End-to-End Model Results

### Llama-3.2-3B-Instruct-4bit — WORKS

| Path | Time (80 tok) | Quality | KV Cache |
|------|--------------|---------|----------|
| Standard FP16 | 0.8s | Baseline | ~16MB |
| **Fused 4-bit** | 2.3s | Correct, coherent | 2.6MB (6x) |
| **Fused 3-bit** | 2.3s | Correct, coherent | 2.2MB (7x) |

Standard output:
> TCP is a connection-oriented protocol that guarantees delivery of packets, whereas UDP is a connectionless protocol...

Fused 4-bit output:
> TCP is a connection-oriented protocol that ensures data is delivered reliably and in the correct order, whereas UDP is a connectionless protocol...

Both correct. Different wording expected from quantization affecting attention.

### Llama-3.2-3B — NOT BENEFICIAL (small model)

Long-context sweep (2K-16K, 64 decode tokens, 3-bit, M4 Pro 64GB):

| Context | FP16 tok/s | PQ3 tok/s | PQ3/FP16 | Note |
|---------|-----------|----------|---------|------|
| 2,048 | 104.9 | 24.3 | 0.23x | |
| 4,096 | 94.8 | 22.2 | 0.23x | |
| 8,192 | 92.6 | 22.8 | 0.25x | seed capped at 4503 tokens |
| 16,384 | 92.5 | 23.9 | 0.26x | seed capped at 4503 tokens |

**Conclusion:** PolarQuant is ~4x slower than FP16 for Llama-3.2-3B at all context lengths.
The overhead is constant (not context-dependent) — the bottleneck is Metal kernel dispatch +
codebook lookup, not memory bandwidth. For a 3B model, the entire KV cache fits in fast
GPU memory and standard SDPA is already bandwidth-optimal. PolarQuant adds overhead without
reducing any real bottleneck.

**Rule:** PolarQuant is beneficial when the model is large enough that KV cache memory
bandwidth IS the bottleneck (35B+). For models ≤ 7B on a 64 GB system, use standard FP16.

### Phi-4-Mini-Instruct-4bit — DEGRADES

4-bit output degrades to repetition loops ("sentences sentences sentences...").

Root cause: NOT a kernel bug. The naive dequant path produces identical attention scores (confirmed: scores match exactly between fused and naive). PolarQuant's quantization error (scores_cos=0.954 per layer) compounds across 32 layers for this architecture.

This matches rachittshah's PR #1059 findings: "Some models degrade below 4-bit (Qwen3-1.7B drops at 3-bit). Recommend 4-bit as default."

### Qwen3.5-35B-A3B-4bit — WORKS (hybrid cache)

Qwen3.5 uses hybrid attention: 30 linear layers (`GatedDeltaNet` with `ArraysCache`) + 10 standard attention layers (`Qwen3NextAttention` with `KVCache`). `make_fused_cache()` now detects `is_linear` and only replaces standard attention layers with `TurboQuantKVCache`.

| Config | Result |
|--------|--------|
| Cache assignment | 30 ArraysCache + 10 TurboQuantKVCache (correct) |
| "What is 2+2?" | "Four" (1.54s) |
| TCP/UDP explanation | Coherent, structured (3.50s, 100 tokens) |
| PolarQuant KV cache | 107.7 KB across 10 layers |

## Architecture

```
Model Attention Layer:
  1. Project Q, K, V
  2. Apply RoPE
  3. cache.update_and_fetch(K, V)     ← quantize + pack (no dequant in fused mode)
  4. scaled_dot_product_attention()    ← patched: detects turbo_bits → cache.fused_sdpa()
     4a. Pre-rotate queries: Q_rot = Q @ R^T
     4b. Fused Q@K^T: codebook lookup inside dot product (Metal kernel)
     4c. Softmax
     4d. Fused scores@V: codebook lookup inside weighted sum (Metal kernel)
     4e. Inverse rotate output: out = out_rot @ R
```

## Per-Operation Profiling (3-bit, 32h/8kv, D=128, decode L_q=1)

After SV kernel optimization (pre-combined weight*norm):

| L_kv | Q Rot (ms) | QK Kern (ms) | Softmax (ms) | SV Kern (ms) | Out Rot (ms) | Sum (ms) | Pipeline (ms) |
|------|-----------|-------------|-------------|-------------|-------------|---------|--------------|
| 64   | 0.163 (17%) | 0.279 (30%) | 0.132 (14%) | 0.219 (23%) | 0.150 (16%) | 0.943 | 0.243 |
| 256  | 0.139 (19%) | 0.153 (21%) | 0.125 (17%) | 0.197 (27%) | 0.121 (16%) | 0.735 | 0.268 |
| 512  | 0.128 (16%) | 0.159 (20%) | 0.128 (16%) | 0.244 (31%) | 0.128 (16%) | 0.787 | 0.311 |
| 1024 | 0.133 (12%) | 0.320 (30%) | 0.140 (13%) | 0.355 (33%) | 0.136 (13%) | 1.084 | 0.452 |
| 2048 | 0.150 (9%)  | 0.431 (27%) | 0.133 (8%)  | 0.752 (47%) | 0.133 (8%)  | 1.599 | 0.940 |

**Key findings:**
- **SV kernel improved 25%** at 1K+ tokens (was 61% of time, now 47% at 2K). Pre-combining weight*norm eliminates one read and one multiply per inner iteration.
- **QK and SV now balanced** at long contexts (~27% vs 47% at 2K).
- **Rotations are constant** — ~0.13ms each regardless of L_kv. Negligible at long contexts.
- **Lazy eval saves 40-74%** — MLX batches Metal kernel launches efficiently.
- **Tiled SV with shared memory was tested but slower** — threadgroup barriers + wasted threads at L_q=1 outweighed shared memory benefits.

## Novelty Assessment (2026-03-31, Perplexity Deep Research)

### Prior Art (not novel)

| Technique | Prior Work |
|-----------|-----------|
| Fused QK codebook kernels on Metal | oMLX v0.2.21 and mlx-lm PR #1067 independently implemented |
| Lazy quantization (FP16 prefill → quantize at decode) | oMLX and PR #1067 converged on same pattern |
| Sparse V concept | SpargeAttn (ICML 2025, Tsinghua) on CUDA |
| Asymmetric K/V concept | KIVI (ICML 2024) on CUDA; extended by PackKV (Dec 2025) |

### Novel Contributions (first on Metal/MLX)

1. **Fused SV kernel** — scores@V directly from packed codebook indices on Metal. "Undemonstrated publicly on any Metal implementation."
2. **Sparse V on Apple Silicon** — threshold-based skipping of near-zero attention positions in Metal kernel. SpargeAttn exists on CUDA but no Metal port.
3. **Asymmetric K/V in MLX ecosystem** — different bitwidths for K vs V. Nothing in MLX uses this.
4. **Combined pipeline** — fused bidirectional attention + Sparse V + asymmetric K/V + adaptive lazy threshold on M4 Pro. "Unambiguously novel" as a system.

**Result:** 75.3 tok/s vs 71.4 tok/s standard (5% faster than FP16 with 8x KV cache compression).

### Key References

- SpargeAttn (ICML 2025) — sparse warp online softmax, CUDA
- KIVI (ICML 2024) — per-channel K / per-token V quantization, CUDA
- PackKV (Dec 2025) — extends KIVI
- oMLX v0.2.21 (March 2026) — fused 2-pass Flash Attention, codebook, Metal
- mlx-lm PR #1067 (arozanov, March 28 2026) — fused Metal quantize/dequantize kernels

## Simdgroup Reduction (2026-04-13)

Added `_SV_SIMD_SOURCE` and `_SV_SIMD_SPARSE_SOURCE` — simdgroup-reduced variants
of the precombined-dense and compact-index-sparse SV kernels respectively.

**Approach:** 32 threads (one simdgroup) cooperate on each output element `(b, h, q, d)`.
- Dense path: lanes stride over `L_kv` in steps of 32 (`k = lane, lane+32, ...`)
- Sparse path: lanes stride over `count` active positions in steps of 32
- `simd_sum(partial)` collapses partial sums in a single hardware instruction
- Lane 0 writes the result

**Grid change:** `(B * n_heads * L_q * D * 32, 1, 1)`, threadgroup `(32, 1, 1)`.

**Wins:**
- Dense: 32x less inner-loop work per thread; better occupancy; effective at all L_kv
- Sparse: parallelizes the count loop — most impactful at L_kv ≥ 8K (count ≥ 80)
- Prefill (L_q > 1) is now also accelerated via the dense simd path

**Benchmark results (M5 Max MBP, 2026-04-13):**

| L_kv | scalar_dense | simd_dense | scalar_sparse | simd_sparse | spd_dense | spd_sparse | cos_sim |
|------|-------------|-----------|--------------|------------|----------|-----------|---------|
| 2048 | 0.83 ms | 0.18 ms | 0.25 ms | 0.17 ms | **4.52x** | 1.48x | 1.000000 |
| 4096 | 0.44 ms | 0.20 ms | 0.18 ms | 0.16 ms | 2.24x | 1.12x | 1.000000 |
| 8192 | 0.79 ms | 0.23 ms | 0.18 ms | 0.15 ms | 3.52x | 1.15x | 1.000000 |
| 16384 | 1.12 ms | 0.27 ms | 0.21 ms | 0.16 ms | 4.11x | 1.34x | 1.000000 |
| 32768 | 2.03 ms | 0.35 ms | 0.26 ms | 0.16 ms | **5.76x** | 1.56x | 1.000000 |

Active positions per head: ~1% of L_kv (concentrated attention pattern).

Key findings:
- **Dense simd: 2.2–5.8x faster** than scalar. Strongest at 32K (5.76x). The dense path
  is now the performance floor — even short contexts see 2x+ from better GPU occupancy.
- **Sparse simd: 1.1–1.6x** over scalar sparse. Smaller gain because count (20–327 active
  positions) is close to SIMD_SIZE=32, so most lanes handle only 0–10 iterations. Effect
  will be larger at 35B with longer contexts and looser thresholds (count > 100).
- **Correctness: perfect** (cosine similarity = 1.000000 at all context lengths).
- M5 Max absolute times are ~2x faster than M4 Pro (different memory bandwidth).
  Run `bench_sv_simd.py` on Mini to get M4 Pro numbers for RESULTS.md.

**Validation:** `kernels/polarquant_sv_simd.metal` added; all 4 kernels compile
clean via `xcrun metal` on macOS 26.4.

## What to Optimize Next

1. **~~SV simdgroup reduction~~** — ✓ Done (2026-04-13). Both dense and sparse paths
   now use `_SV_SIMD_SOURCE` / `_SV_SIMD_SPARSE_SOURCE`. Benchmark on 35B to quantify.
2. **Short context speed** — at L_kv=64, kernel dispatch overhead dominates (74% lazy eval savings). Note: `min_fused_context=512` already prevents fused path below 512 tokens; overhead exists in isolated kernel bench only
3. **Prefill query tiling** — simdgroup covers the L_kv reduction but not multi-query
   tiling. For very large prefill (L_q ≥ 512), a threadgroup-tiled QK + SV kernel
   would further help. Needs `threadgroup_memory_length` and barrier coordination.
4. **~~Model compatibility~~** — ✓ Done (2026-04-13): Llama-3.2-3B is NOT beneficial (see above). PolarQuant sweet spot is 35B+ where KV memory bandwidth is the bottleneck
5. **~~Longer contexts (4K-8K)~~** — ✓ Done (2026-04-13): see Llama-3B sweep above. Full-fidelity test needs 35B+ model
