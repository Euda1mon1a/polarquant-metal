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
| 64 | 0.19ms | 0.38ms | 0.47ms | 0.80x | 2.43x | 4.6x |
| 256 | 0.22ms | 0.30ms | 0.27ms | **1.11x** | 1.22x | 4.6x |
| 512 | 0.19ms | 0.40ms | 0.31ms | **1.32x** | 1.60x | 4.6x |
| 1024 | 0.43ms | 0.80ms | 0.44ms | **1.82x** | **1.03x** | 4.6x |
| 2048 | 0.48ms | 1.44ms | 0.97ms | **1.48x** | 2.02x | 4.6x |

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

### Phi-4-Mini-Instruct-4bit — DEGRADES

4-bit output degrades to repetition loops ("sentences sentences sentences...").

Root cause: NOT a kernel bug. The naive dequant path produces identical attention scores (confirmed: scores match exactly between fused and naive). PolarQuant's quantization error (scores_cos=0.954 per layer) compounds across 32 layers for this architecture.

This matches rachittshah's PR #1059 findings: "Some models degrade below 4-bit (Qwen3-1.7B drops at 3-bit). Recommend 4-bit as default."

### Qwen3.5-35B-A3B-4bit — NOT TESTED

Requires special handling: Qwen3.5 uses hybrid linear/standard attention with `ArraysCache(size=2)` for linear layers and `KVCache()` for standard layers. Cannot replace all caches with TurboQuantKVCache.

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

## What to Optimize Next

1. **Short context speed** — kernel dispatch overhead dominates at <256 tokens
2. **Prefill kernel** — simdgroup reduction, shared memory for query tiles (currently 0.2x)
3. **Qwen3.5 support** — per-layer cache type selection for hybrid architectures
4. **Profile bottleneck** — Metal kernel vs rotation matmul vs packing overhead
5. **Model compatibility** — test more architectures to map which work with PolarQuant
