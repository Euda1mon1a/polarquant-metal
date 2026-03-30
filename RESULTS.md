# PolarQuant Metal — Benchmark Results

> Mac Mini M4 Pro (64GB), MLX 0.31.1, 2026-03-29

## Tests: ALL PASS (7/7)

| Test | Max Diff | Cosine Sim |
|------|----------|-----------|
| Pack/unpack roundtrip | exact | exact |
| Fused Q@K^T (simple) | 1.07e-6 | 1.000000 |
| Fused Q@K^T (tiled) | 1.19e-6 | 0.999999 |
| Fused weights@V | 3.58e-7 | 1.000000 |
| Full attention | 1.31e-6 | 1.000000 |
| GQA (8h/2kv) | 8.34e-7 | 1.000000 |
| All bit widths (2/3/4) | <6.6e-7 | 1.000000 |

## Decode Benchmark (L_q=1, B=1, 32h/8kv, D=128, 3-bit)

| KV Length | FP16 | Naive Dequant | Fused Metal | Fused vs Naive | Fused vs FP16 | Compression |
|-----------|------|--------------|-------------|---------------|--------------|-------------|
| 64 | 0.19ms | 0.38ms | 0.47ms | 0.80x | 2.43x | 4.6x |
| 256 | 0.22ms | 0.30ms | 0.27ms | **1.11x** | 1.22x | 4.6x |
| 512 | 0.19ms | 0.40ms | 0.31ms | **1.32x** | 1.60x | 4.6x |
| 1024 | 0.43ms | 0.80ms | 0.44ms | **1.82x** | **1.03x** | 4.6x |
| 2048 | 0.48ms | 1.44ms | 0.97ms | **1.48x** | 2.02x | 4.6x |

**Key finding:** At 1024 tokens, fused kernel matches FP16 speed (1.03x) with 4.6x memory compression.

## Prefill Benchmark (L_q=L_kv, same config)

| Size | FP16 | Naive | Fused | Fused vs Naive |
|------|------|-------|-------|---------------|
| 64x64 | 0.20ms | 0.28ms | 0.68ms | 0.41x |
| 256x256 | 0.70ms | 0.97ms | 4.15ms | 0.23x |

**Prefill is bad.** Kernel doesn't parallelize across L_q. Needs simdgroup optimization.

## Bit Width Comparison (L_kv=512)

| Bits | FP16 | Naive | Fused | Fused/Naive | Compression |
|------|------|-------|-------|------------|-------------|
| 2-bit | 0.18ms | 0.43ms | 0.29ms | **1.50x** | 7.1x |
| 3-bit | 0.20ms | 0.38ms | 0.34ms | **1.11x** | 4.6x |
| 4-bit | 0.19ms | 0.42ms | 0.33ms | **1.27x** | 3.8x |

## What to Optimize Next

1. **Prefill kernel** — simdgroup-level reduction, shared memory for query tiles
2. **Short cache (L<256)** — kernel launch overhead dominates; batch more work per thread
3. **Decode at L>2048** — fused is 2x FP16, need to investigate memory access patterns
4. **simd_sum for inner D-loop** — current kernel is scalar accumulate
