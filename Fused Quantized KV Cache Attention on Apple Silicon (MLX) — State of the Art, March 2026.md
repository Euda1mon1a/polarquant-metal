# Fused Quantized KV Cache Attention on Apple Silicon (MLX) — State of the Art, March 2026

## Executive Summary

As of late March 2026, the MLX/Apple Silicon ecosystem has seen a burst of activity around codebook-style KV cache quantization (TurboQuant/PolarQuant), with at least two open pull requests against `mlx-lm` and one shipping third-party runtime. However, several of the techniques you describe — specifically (a) true zero-materialization fused codebook attention kernels for both Q@K^T and P@V, (b) Sparse-V aggregation skipping on Metal, and (c) asymmetric K/V bitwidths with fused Metal attention — remain unimplemented or unpublished in the Apple Silicon ecosystem. The prefill-FP16 → bulk-quantize-at-decode-start pattern has been independently converged upon by multiple implementors, but context-length-threshold adaptive switching has not been explored. A system combining all five elements on M4 Pro would be meaningfully novel.

***

## 1. Fused Codebook Attention Without Dequantization

### The Core Algebraic Identity

The key insight enabling fused quantized attention — used in all current codebook-based approaches — is:

\[ \langle q,\ R^\top \cdot \text{centroids}[\text{idx}] \rangle = \langle R \cdot q,\ \text{centroids}[\text{idx}] \rangle \]

Pre-rotating the query once turns every per-position KV operation into a centroid table lookup plus dot product, with the codebook fitting in L1/L2 cache and KV indices loaded as uint8 (1 byte) instead of FP16 (2 bytes). This is the "fused" path that loads packed bit-indices rather than dequantizing to float first.

### What Exists on Apple Silicon

**oMLX v0.2.21 (March 2026, jundot)** is the first shipping Apple Silicon LLM runtime to implement a fused codebook attention kernel. Per the release notes: *"Decode attention uses a fused 2-pass Flash Attention Metal kernel that reads directly from packed indices — no dequantization, no fp16 intermediate tensors."* The runtime applies TurboQuant's random-rotation + Lloyd-Max codebook approach, storing 3-bit or 4-bit indices, and builds the kernel using MLX's `mx.fast.metal_kernel` API. The key innovation is an incremental decode buffer (O(1) cost per new token) rather than full cache dequantization per step.[^1][^2]

**mlx-lm PR #1067 (arozanov, filed March 28, 2026)** proposes adding TurboQuant to mlx-lm mainline. It includes fused Metal kernels for the quantize and dequantize operations in single GPU dispatches, reports **0.98x FP16 decode speed** at 4.6x compression on Qwen2.5-32B running on an M4 Pro 48GB machine, and is the first public mlx-lm PR to reach near-FP16 speed with a codebook KV scheme. As of March 29, 2026, it is open and not yet merged.

**mlx-lm PR #1059 (rachittshah, March 26, 2026)** adds PolarQuant as a pure mlx.core implementation (no custom Metal). This PR explicitly acknowledges that "decode throughput is ~0.5x FP16 due to the dequantize-on-fetch path. A fused Metal kernel… would close this gap but is out of scope for this PR." It achieves logit cosine similarity of 0.988 at 3-bit on Llama 3.2-3B, with 4.6x compression, but lacks the speed to match FP16.

**turboquant_plus (llama.cpp/Metal, TheTom, March 24, 2026)** implements Metal shaders for the TurboQuant dequantization path in llama.cpp. On an M5 Max with Qwen 3.5 35B MoE it achieves ~4.9x KV compression with speed approximately at q8_0 parity. The Metal shaders are described as "unoptimized" and the project is not merged upstream.[^3][^4]

### What Exists on NVIDIA (CUDA / Triton)

**PyTorch/Triton (dejan.ai, March 2026)** provides the most detailed public analysis of a fused codebook Q@K^T kernel on NVIDIA GPUs. The Triton kernel loads uint8 codebook indices and achieves a 1.15-1.22x speedup over standard dequantize-then-matmul on an RTX 4090. Critically, the implementation only fuses key compression; values remain in FP16 because "the softmax@V multiplication is less critical… at typical sequence lengths".[^5]

**PackKV (arXiv 2512.24449, December 2025)** achieves fused decompression-in-matmul on CUDA A100 and RTX Pro 6000, with 75.7% throughput improvement for K and 171.7% for V compared to cuBLAS matmul. This is the strongest existing result for a fused kernel spanning both K and V, but has no Metal/Apple Silicon port.[^6][^7]

**Flash-attention de-quant CUDA experiments (GitHub issue #2401)** document the difficulty of the problem directly. Iterative CUDA kernel optimization went from 3,823 µs (V1) to 500 µs (V3) for a 4-bit codebook lookup kernel vs. a 46 µs BF16 baseline — still 10x slower before further optimization.[^8]

### What Remains Novel on M4 Pro

The **P@V (values aggregation) fusion** — loading packed value indices and computing the weighted sum without materializing FP16 intermediates — is undemonstrated in any public MLX implementation. oMLX's phrasing ("no fp16 intermediate tensors") suggests this may be achieved, but the kernel code is not publicly auditable. A **single Metal kernel dispatch** that handles both Q@K^T and P@V from packed indices, confirmed on M4 Pro with benchmarks showing faster-than-FP16 decode, has not been published.

***

## 2. Sparse-V Attention (Skipping Near-Zero Softmax Weights)

### State of the Art

**SpargeAttn (Tsinghua thu-ml, ICML 2025)** is the primary published implementation of V-aggregation skipping based on softmax weight sparsity. The system has two stages: a block-level sparse prediction stage that skips selected Q@K^T computations, and a **"sparse warp online softmax" stage that further skips P_ij @ V_j products** by comparing warp-local maximum values against the running global maximum. Claims of 4-7x speedup over FlashAttention are reported on NVIDIA GPUs. **SpargeAttention2** (published February 2026) extends this with a hybrid Top-k+Top-p masking strategy.[^9][^10]

**Self-Indexing KVCache (arXiv 2603.14224, March 2026)** combines sparse FlashAttention CUDA kernels with LUT-GEMV for 6.7x acceleration in sparse attention computation and 2x end-to-end speedup vs FlashAttention v2.[^11]

**OpenAI's sparse_attention (2019)** remains the earliest published block-sparse Q@K^T CUDA kernel, but does not address the P@V aggregation step.[^12]

### Apple Silicon Gap

**No port of SpargeAttn or any sparse V-aggregation kernel exists for Metal or the MLX ecosystem** as of March 2026. An RFC was filed in vLLM-Omni to add a SpargeAttn backend, but this targets diffusion model workloads on NVIDIA GPUs. The Thu-ml SpargeAttn repository is CUDA-only. Sparse V-skip on Apple Silicon would require a new Metal shader implementing the two-stage filter within a Flash Attention-style tiling loop, which is a materially different implementation challenge from CUDA given Metal's SIMD-group memory model.[^13][^14][^15]

***

## 3. Asymmetric K/V Quantization with Fused Kernels

### Asymmetric Granularity (Published Work)

**KIVI (ICML 2024)** is the canonical reference for asymmetric K/V quantization. It uses per-channel quantization for keys and per-token quantization for values — exploiting the empirical observation that key distributions have large outliers in specific channels while value distributions are more uniform per token. KIVI achieves 2.6x memory reduction and up to 3.47x throughput improvement on CUDA with custom fused dequant+matmul kernels.[^16][^17]

**PackKV (December 2025)** explicitly reports asymmetric treatment: 153.2% higher memory reduction for K vs 179.6% for V compared to state-of-the-art quantization, reflecting different compression rates for keys and values.[^7][^6]

**KVQuant** supports per-layer sensitivity-weighted non-uniform datatypes that can differ between K and V, with custom CUDA kernels achieving ~1.7x speedup over FP16.[^18]

**LMDeploy** supports online int4/int8 KV quantization with per-head per-token asymmetric quantization on NVIDIA.[^19]

### Asymmetric Bitwidths in Current MLX Work

All current MLX KV quantization work (PRs #1059, #1067, oMLX, mlx-turboquant) applies the **same bit width to both K and V**. The algorithmic argument for asymmetry is well-established (keys need per-channel treatment; values tolerate per-token), but no MLX implementation has exercised different bit widths per tensor or even different quantization granularity. Combining asymmetric bitwidth assignment (e.g., 4-bit K with 2-bit V, or per-channel K with per-token V) with separate fused Metal attention kernels for each path is fully novel in the Apple Silicon context.[^20][^1]

***

## 4. KV Cache Compression in mlx-lm Beyond `mx.quantize`

### Built-in Capabilities

MLX's `mx.quantize` and `mx.quantized_matmul` provide affine (scale+zero-point) weight quantization with a fused dequant+matmul Metal kernel (`affine_qmm_t`). This is designed for model weight compression, not KV cache. The stock mlx-lm rotating KV cache manages long-context generation but applies no compression.[^21][^22]

### Community Extensions (March 2026)

| Project | Approach | Speed vs FP16 | Compression | Status |
|---------|----------|---------------|-------------|--------|
| mlx-lm PR #1059 (rachittshah) | PolarQuant, pure mlx.core | ~0.5x (no fused kernel) | 4.6x at 3-bit | Open PR, not merged |
| mlx-lm PR #1067 (arozanov) | TurboQuant, fused Metal kernels | **~0.98x** | 4.6x at 3-bit | Open PR, not merged |
| oMLX v0.2.21 (jundot) | TurboQuant, 2-pass fused FlashAttn Metal | ~0.98x (M4 Pro 48GB) | 4.6x at 3-bit | Shipped in oMLX[^2] |
| mlx-turboquant (rachittshah) | PolarQuant standalone package | ~0.5x | 4.6x at 3-bit | pip-installable[^20] |
| turboquant_plus (TheTom) | llama.cpp + Metal shaders | ≈q8_0 parity | ~4.9x | External, not merged[^4] |
| ZMLX | Fused MoE decode Metal kernels | +2–12% vs baseline | Model weights only | Shipping[^23] |
| vMLX / JANG | Paged KV, q4/q8, importance-aware | q4/q8 tiers | 4–8x (weights) | Shipping[^24][^25] |

The key gap is that **no TurboQuant/PolarQuant fused Metal attention kernel is yet merged into mlx-lm mainline**. The two competing PRs (#1059 and #1067) have different trade-offs: #1059 prioritizes simplicity (200 lines, no C, no Metal shader authoring required) and #1067 prioritizes speed (fused Metal kernel, near-FP16 throughput). Neither addresses sparse V-skip or asymmetric K/V quantization.

***

## 5. Lazy / Deferred KV Quantization

### The Prefill-FP16 Pattern

The "store FP16 during prefill, bulk-quantize at decode start" pattern has been independently converged upon by every MLX-targeted TurboQuant implementation, and is described identically in oMLX v0.2.21 and mlx-lm PR #1067. The motivation is correct: prefill is compute-bound and memory pressure from a growing KV cache is not yet the bottleneck; the decode phase, which is memory-bandwidth-bound on Apple Silicon's unified memory architecture, benefits from smaller cache reads.[^2][^1]

**KIVI** (CUDA) pioneered a related but different mechanism: maintaining a sliding **FP16 buffer of the most recent 32 tokens** and quantizing only the older "residual" portion of the KV cache. This addresses per-channel quantization's cold-start problem — you cannot know the per-channel scale until you have enough tokens. The mlx-lm PR #1067 also notes this but solves it by using data-oblivious codebooks (no per-tensor scale needed).[^26]

### What Is Missing: Context-Length Threshold Switching

The pattern described in the original question — staying in FP16 until a configurable context length threshold (e.g., 2K tokens) and then dynamically switching to a compressed attention kernel — has **not been published or implemented anywhere**. The closest existing behavior is:

- oMLX: quantizes at first *decode* token, regardless of prefill length[^1]
- KIVI: maintains FP16 for last 32 tokens, quantizes older context[^26]
- mlx-lm PR #1059: explicit `to_turboquant()` conversion callable by user

None of these implements adaptive behavior based on measured context length or memory pressure. A system that monitors accumulated KV cache size, and switches attention kernel paths (FP16 FlashAttention → compressed codebook attention) at a user-configurable threshold would be genuinely novel, particularly if the switch also triggers per-head quality-adaptive bit-width selection.

***

## Novelty Assessment for a Combined M4 Pro Pipeline

The following table summarizes which components are novel vs. prior art:

| Component | Prior Art Status | Novelty on M4 Pro/MLX |
|-----------|-----------------|----------------------|
| Fused Q@K^T from packed codebook indices | CUDA/Triton: published[^5]; Metal: shipped in oMLX[^2], PR in mlx-lm | **Partially prior art** — exists but not merged in stock mlx-lm |
| Fused P@V from packed value indices | CUDA: PackKV[^6]; Metal: undemonstrated publicly | **Novel on Apple Silicon** |
| Sparse-V (skip near-zero P@V blocks) | CUDA: SpargeAttn[^10], Self-Indexing KVCache[^11]; Metal: none | **Novel on Metal** |
| Asymmetric K/V bitwidths + fused kernel | CUDA: KIVI[^16], PackKV[^6]; Metal: none | **Novel on Apple Silicon** |
| Context-length-threshold lazy quantization | Closest: KIVI FP16 buffer (size-triggered)[^26]; oMLX (decode-step-triggered)[^2] | **Novel as described** |
| All five combined in one pipeline | Not found anywhere | **Novel combination** |

### Achieving Faster-than-FP16 Decode on M4 Pro

The claim of "faster-than-FP16 decode speed with 8x compression" is aggressive but mechanistically tractable. Apple Silicon decode is entirely memory-bandwidth-bound. For a 32B model at 16K context on M4 Pro 48GB, the FP16 KV cache is ~4 GB; at 3-bit TurboQuant it is ~900 MB. Each decode step reads proportionally less data, so the kernel is loading 4.6x fewer bytes from unified memory. The 0.98x FP16 speed observed in PR #1067 is likely bottlenecked by the quantize/dequantize overhead and kernel launch latency, not by the attention compute itself.[^27]

To break the FP16 ceiling you need: (a) fused kernel that reads compressed indices without a separate dequant dispatch, (b) incremental decode buffer that avoids re-processing the full cache per step, and (c) Metal shader tuned for Apple Silicon's SIMD-group size (32 threads) and unified memory cache hierarchy. The oMLX 2-pass fused kernel appears to achieve ~0.98x at 3-bit; at lower bitwidths (2-bit, providing ~8x compression) the bandwidth advantage grows, potentially pushing the fused kernel past FP16 speed — analogous to what DFloat11 achieves on bandwidth-constrained consumer NVIDIA GPUs (exceeding BF16 on RTX4060).[^28][^29]

***

## Key Open Questions and Gaps

- **Value-side fusion completeness**: All published MLX/Metal implementations either skip V compression or fuse only K. Demonstrating full two-sided fusion with benchmarks is the clearest technical gap.
- **Sparse V on Metal**: SpargeAttn's sparse warp online softmax requires warp-level intrinsics (`simd_prefix_inclusive_sum`-style operations). Metal Shading Language supports SIMD-group operations and this is theoretically portable, but no one has published this port.
- **Asymmetric quantization granularity**: Per-channel K quantization requires tracking running channel statistics during decode. Combining this with a codebook quantizer (rather than affine scale/zero-point) is algorithmically underexplored.
- **Benchmark methodology on Apple Silicon**: Existing speed claims (0.98x FP16) are on specific model sizes and context lengths. Benchmarks at shorter contexts (< 4K) where codebook overhead may dominate bandwidth savings have not been published.

---

## References

1. [V0.2.21 released - big update!!](https://www.reddit.com/r/oMLX/comments/1s452ef/v0221_released_big_update/) - V0.2.21 released - big update!!

2. [jundot/omlx v0.2.21 on GitHub](https://newreleases.io/project/github/jundot/omlx/release/v0.2.21) - New release jundot/omlx version v0.2.21 on GitHub.

3. [I implemented Google's TurboQuant paper (ICLR 2026) in llama.cpp ...](https://x.com/no_stp_on_snek/status/2036792058854121601)

4. [TurboQuant: Finally, Fast and Widely Available Low-Bit KV Cache ...](https://kaitchup.substack.com/p/turboquant-finally-fast-and-widely) - Separately, the turboquant_plus project reports a llama.cpp/Metal integration on Apple Silicon, with...

5. [TurboQuant: From Paper to Triton Kernel in One Session - Dejan.ai](https://dejan.ai/blog/turboquant/) - Implementing Google’s KV cache compression algorithm on Gemma 3 4B and everything that went wrong al...

6. [PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy ...](https://arxiv.org/html/2512.24449v1) - Computation-aware decompression integration: We propose a co-design strategy that embeds decompressi...

7. [PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy ...](https://arxiv.org/abs/2512.24449) - Abstract page for arXiv paper 2512.24449: PackKV: Reducing KV Cache Memory Footprint through LLM-Awa...

8. [[De-quant findings] #2401 - Dao-AILab/flash-attention - GitHub](https://github.com/Dao-AILab/flash-attention/issues/2401) - I was thinking of a single fused kernel of quantized load_KV branch as a standalone CUDA C kernel co...

9. [SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top ...](https://arxiv.org/abs/2602.13515) - SpargeAttention2 includes (i) a hybrid masking rule that combines Top-k and Top-p for more robust ma...

10. [SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference](https://arxiv.org/html/2502.18137v2)

11. [Self-Indexing KVCache: Predicting Sparse Attention from ... - arXiv](https://arxiv.org/html/2603.14224v1) - To further alleviate the KV cache bottleneck, solutions such as sparsification, quantization, and lo...

12. [openai/sparse_attention: Examples of using sparse attention, as in ...](https://github.com/openai/sparse_attention) - The kernels allow specification of block sparsity in the QK^T matrix. This means you define a patter...

13. [Activity · thu-ml/SpargeAttn - GitHub](https://github.com/thu-ml/SpargeAttn/activity) - [ICML2025] SpargeAttention: A training-free sparse attention that accelerates any model inference. -...

14. [[RFC]: Add SpargeAttn Sparse Attention Backend #765 - GitHub](https://github.com/vllm-project/vllm-omni/issues/765) - This RFC proposes adding SpargeAttn as a new diffusion attention backend in vLLM-Omni to better supp...

15. [thu-ml/SpargeAttn: [ICML2025] SpargeAttention: A training ... - GitHub](https://github.com/thu-ml/SpargeAttn) - The official implementation of SpargeAttn, a universal training-free sparse attention accelerating l...

16. [KIVI: Tuning-Free 2-Bit KV Cache Quantization - Emergent Mind](https://www.emergentmind.com/papers/2402.02750) - This paper introduces KIVI, a tuning-free asymmetric 2bit quantization method for KV cache that achi...

17. [GitHub - jy-yuan/KIVI: [ICML 2024] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://github.com/jy-yuan/KIVI) - [ICML 2024] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache - jy-yuan/KIVI

18. [KVQuant: Towards 10 Million Context Length LLM Inference with KV ...](https://arxiv.org/abs/2401.18079) - Our work, KVQuant, facilitates low precision KV cache quantization by incorporating several novel me...

19. [Key-Value(KV) Cache Quantization - Read the Docs](https://lmdeploy.readthedocs.io/en/v0.5.0/quantization/kv_quant.html) - LMDeploy has supported online key-value (kv) cache quantization with int4 and int8 numerical precisi...

20. [TurboQuant KV cache compression for MLX (Apple Silicon) - GitHub](https://github.com/rachittshah/mlx-turboquant) - TurboQuant KV cache compression for MLX on Apple Silicon. Implements PolarQuant (Google, ICLR 2026) ...

21. [Metal - Hugging Face](https://huggingface.co/docs/transformers/quantization/metal) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

22. [MLX LM: Python toolkit for running and fine-tuning LLMs on Apple ...](https://jimmysong.io/ai/mlx-lm/) - The library includes prompt caching and a rotating KV cache to improve long-context generation and s...

23. [ZMLX — Metal kernels and model patching for MLX on ... - Libraries.io](https://libraries.io/pypi/zmlx) - ZMLX: Metal-kernel toolkit and optimization lab for MLX on Apple Silicon. Fused MoE decode (+2-12% o...

24. [Welcome to r/mlxllm! 🚀 Pushing Apple Silicon to the limit with vMLX & JANG (397B on a 128GB Mac)](https://www.reddit.com/r/MLXLLM/comments/1s0euar/welcome_to_rmlxllm_pushing_apple_silicon_to_the/) - Welcome to r/mlxllm! 🚀 Pushing Apple Silicon to the limit with vMLX & JANG (397B on a 128GB Mac)

25. [JANG — 397B on a 128 GB Mac | The GGUF for MLX](https://jangq.ai) - 397 billion parameters on a 128 GB Mac — 86.5% MMLU at 112 GB. 92% MMLU with JANG_2L. First Nemotron...

26. [KV Cache is huge and bottlenecks LLM inference. We quantize them to 2bit in a finetuning-free + plug-and-play fashion.](https://www.reddit.com/r/LocalLLaMA/comments/1ap3bkt/kv_cache_is_huge_and_bottlenecks_llm_inference_we/) - KV Cache is huge and bottlenecks LLM inference. We quantize them to 2bit in a finetuning-free + plug...

27. [perf: benchmark generation with quantized models (4-bit inference ...](https://github.com/teilomillet/textpolicy/issues/43) - Since Apple Silicon inference is memory-bandwidth-bound, quantized generation should be ~2-4x faster...

28. [Fast CUDA DFloat11 decoding kernel : r/LocalLLaMA - Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1myz6f1/fast_cuda_dfloat11_decoding_kernel/) - A few months ago, I came across the amazing work on DFloat11, which achieves lossless output while s...

29. [MLX上のTurboQuant：カスタムMetalカーネルによるKVキャッシュ ...](https://ai-navigate-news.com/ja/articles/89f498a2-656e-42c7-9456-06f78324bf40) - - この記事では、コードリポジトリ（turboquant-mlx）と、mlx-lmへのPRの両方が共有されており、この取り組みがMLXエコシステムに積極的に統合されていることが示唆される。

