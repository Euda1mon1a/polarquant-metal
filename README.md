# PolarQuant Metal: Fused Metal Kernels for PolarQuant KV Cache

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
git clone <this-repo>
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
├── kernels.py               # Metal kernel source + Python wrappers
├── cache.py                 # FusedPolarQuantKVCache
├── polar_quant.py           # PolarQuant quantizer (rotation + codebooks)
├── packing.py               # Bit-packing utilities
├── codebooks.py             # Lloyd-Max codebooks (hardcoded, no file dependency)
├── integration.py           # mlx-lm SDPA monkey-patch
└── mlx_turboquant_adapter.py # Drop-in for rachittshah/mlx-turboquant
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

## Limitations

1. **Prefill is slow** — the per-element kernel can't parallelize across L_q > 1. Prefill falls back to standard FP16 SDPA automatically. Fused kernels only benefit decode (L_q=1).

2. **No QJL residual correction** — handles Stage 1 (PolarQuant) only. Adding QJL would require an additional correction term in the Q@K^T kernel.

3. **Model-dependent quality** — Llama-3.2-3B and Qwen3.5-35B produce correct output. Phi-4-Mini degrades (PolarQuant itself produces low cosine similarity on this architecture — same issue in upstream PR #1059).

4. **SV kernel is the bottleneck** — 47% of time at 2K tokens. Tiled SV and simd_broadcast_first were both tested and found slower than the simple per-element kernel (Metal L1 cache handles broadcast efficiently). Pre-combined weight*norm optimization gives 25% improvement.

## License

MIT

## References

- [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) — Zandieh et al., ICLR 2026
- [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) — Paper-faithful MLX implementation
- [ml-explore/mlx-lm Issue #1060](https://github.com/ml-explore/mlx-lm/issues/1060) — Upstream tracking
- [MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) — MLX kernel API docs
