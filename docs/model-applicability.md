# PolarQuant Metal — Model Applicability

## The Unifying Rule

PolarQuant's three mechanisms (fused QK/SV Metal kernels, sparse SV, entropy-guided per-head thresholding) all target the **autoregressive transformer decoder KV cache**. Applicability is determined by one question:

> Does this model generate tokens autoregressively using a transformer decoder?

- **Yes → full benefit.** The KV cache grows with context; compression, sparse SV, and entropy-guided thresholding all engage.
- **No (diffusion, flow-matching, encoder-only) → incompatible.** No KV cache, no path.

---

## Current Stack (OpenClaw MLX Serving)

| Model | Port | Architecture | Verdict |
|---|---|---|---|
| **Qwen3.5-35B-A3B** | 8080 | Autoregressive decoder (MoE) | ✅ Full benefit — primary target |
| **Phi-4-Mini** | 8081 | Autoregressive decoder | ✅ Full benefit |
| **MedGemma 1.5 4B** | ~~8086~~ | Gemma autoregressive decoder | ✅ Full benefit (currently DISABLED — @med routes to Qwen on 8080) |
| **PaddleOCR-VL-1.5** | 8083 | VLM — encoder + autoregressive decoder | ✅ Decoder side; visual token prefix amplifies KV cache pressure |
| **Whisper large-v3-turbo** | 8090 | Encoder-decoder | ⚠️ Decoder is autoregressive but generates <200 tokens; negligible gain |
| **Kokoro** | 8090 | StyleTTS2 / flow-matching | ❌ No KV cache — incompatible |
| **FLUX.1-schnell** | 8084 | Diffusion | ❌ No KV cache — incompatible |
| **all-MiniLM-L6-v2** | 8082 | Encoder-only (embeddings) | ❌ No autoregressive cache |

### Notes

**PaddleOCR-VL**: Visual token prefixes (256–1024 tokens per image) inflate the KV cache significantly before the first generated token. This is the same amplification effect seen in LLaVA/Qwen2.5-VL — compression benefit is proportionally higher than equivalent pure-text contexts.

**Whisper**: The decoder runs but transcription sequences are short. The break-even point for sparse SV is roughly 500+ tokens; Whisper rarely exceeds 200. Not a high-value porting target unless doing long-form multi-hour transcription.

**Kokoro**: StyleTTS2 architecture generates mel spectrograms via flow-matching (non-autoregressive). There is no token-by-token generation loop and no KV cache. No applicable path.

---

## Broader Applicability by Modality

| Modality | Verdict | Notes |
|---|---|---|
| LLM (text) | ✅ Full benefit | Core target |
| TTS — autoregressive (XTTS-v2, OuteTTS, Parler) | ✅ Full benefit | Generate thousands of audio tokens; high-value |
| TTS — flow/diffusion (Kokoro, F5-TTS, Voicebox) | ❌ Incompatible | No KV cache |
| STT/ASR (Whisper, Canary) | ⚠️ Decoder only, short sequences | Low priority |
| STS voice pipelines (Ultravox, LLaMA-Omni, Qwen-Audio) | ✅ LLM backbone benefits fully | The bottleneck is always the LLM decoder |
| Streaming STS (Moshi) | ⚠️ Needs latency-safe entropy variant | Entropy computation must fit within ~160ms streaming budget |
| Vision-Language (Qwen2.5-VL, LLaVA-Next) | ✅ High-value | Visual tokens amplify benefit; best expansion target |
| Image generation (FLUX, SD3) | ❌ Incompatible | Diffusion |

---

## Porting Priority

1. **Phi-4-Mini (8081)** — small, fast feedback loop, same autoregressive structure as Qwen
2. **MedGemma (8086)** — Gemma architecture, straightforward port
3. **PaddleOCR-VL (8083)** — decoder side only; visual prefix makes compression more impactful here than on pure-text models of equivalent size
4. **Qwen2.5-VL** — if added to the stack, highest-impact VLM target
5. **XTTS-v2 / OuteTTS** — if TTS moves to autoregressive models, drop-in
6. **Moshi** — future work; requires a latency-budget-aware entropy computation path

---

## Integration Status (Current Stack)

| Model | Port | PQ-Metal | Priority | Notes |
|---|---|---|---|---|
| **Qwen3.5-35B** | 8080 | ✅ Active | Done | Primary target, validated |
| **Phi-4-Mini** | 8081 | ⬜ Ship next | High | GQA, small footprint — KV cache interface identical to Llama family, near-free integration |
| **MedGemma 1.5 4B** | ~~8086~~ | ⬜ When re-enabled | Medium | Currently DISABLED (2026-03-29), @med routes to Qwen on 8080. Re-enable as standalone when memory budget allows, then integrate PQ. |
| **PaddleOCR-VL-1.5** | 8083 | ⬜ Ship next | High | Visual prefix likely 256–512 tokens — KV compression pays on every inference call, not just long conversations |
| **Whisper** | 8090 | ⏭ Skip | Low | <200 decoder tokens, gain is noise-level |
| **Kokoro** | 8090 | ❌ Skip | None | Flow matching, no path |
| **FLUX.1-schnell** | 8084 | ❌ Skip | None | Diffusion, no path |
| **all-MiniLM-L6-v2** | 8082 | ❌ Skip | None | Encoder-only, no path |

### Integration Order Rationale

**1. Phi-4-Mini first.** Lowest-friction port — mlx-lm already handles it with the same `QuantizedKVCache` path the stack hooks into. Gives a second validated target with near-zero extra work and a useful comparison point: how does a small dense model's attention entropy profile differ from the 35B MoE?

**2. PaddleOCR-VL second.** Most interesting case architecturally. OCR tasks generate visual tokens from document images — often dense, high-token-count prefixes (tables, full-page scans). Sparse SV likely performs differently on visual attention patterns (cross-modal attention tends to be more uniformly spread than pure text). Novel data point on entropy-guided threshold behavior, plus a genuinely useful real-world performance improvement for document processing.

**3. MedGemma when re-enabled.** Currently disabled (2026-03-29) — @med queries route to Qwen on 8080 instead. Most strategically important for clinical use, but needs to be re-enabled as a standalone endpoint first. PolarQuant integration would justify re-enabling it by reducing its memory footprint below the threshold that caused it to be shed.

### Memory Impact of Full Integration

All 8 services compete for unified memory on the same M4 Pro. With PolarQuant Metal on the three decoder targets (Phi-4-Mini, MedGemma, PaddleOCR-VL), KV cache footprint for those models drops ~4.6×. That freed memory either:

- Allows longer contexts on all three simultaneously without weight eviction
- Creates headroom to add a 4th decoder model (e.g., a reasoning-specialized slot)

Whisper and all-MiniLM are already memory-cheap by nature. FLUX and Kokoro are the heavyweight non-LLM residents — their memory pressure is a separate problem this stack can't solve.

---

## Toward a Unified Autoregressive Stack

The current stack has three architecturally incompatible residents (FLUX, Kokoro, all-MiniLM) that can't benefit from PolarQuant Metal. The following swaps would unify the stack onto autoregressive decoders, letting the custom kernels accelerate every port.

### TTS: Kokoro → OuteTTS (or XTTS-v2)

OuteTTS is a standard causal LLM trained to output audio tokens. Audio tokens accumulate fast — thousands of tokens for a few paragraphs. With 4.6× KV compression and sparse SV, long-form narration (e.g., a full research paper) becomes feasible without OOM. XTTS-v2 is the same story: GPT-style autoregressive decoder, drop-in integration.

**Trade-off:** Kokoro's flow-matching is fast for short utterances. Autoregressive TTS is slower at low token counts but scales better at long context — the crossover point is roughly 30–60 seconds of audio.

### STT/Voice: Whisper → Ultravox or Qwen2-Audio

Instead of a dedicated transcription model, these are multimodal LLMs that ingest audio tokens directly into the LLM context window. Audio token prefixes are large; for extended clinical dictation or long voice sessions, KV compression applies natively. The model also responds directly rather than requiring a separate LLM call, collapsing two pipeline stages into one.

**Trade-off:** Whisper is purpose-built for transcription accuracy. Ultravox/Qwen2-Audio are more flexible but may lag on raw WER for short medical terms. Worth benchmarking on your actual dictation vocabulary before committing.

### Image Generation: FLUX → VAR-based autoregressive model

Visual AutoRegressive (VAR) models (e.g., LlamaGen, Lumina-mGPT variants) predict image tokens next-token-style using a standard transformer decoder. Generating a high-resolution image autoregressively runs thousands of decode steps — exactly the workload the compact-index kernel was built to accelerate.

**Trade-off:** Autoregressive image quality currently lags diffusion at equivalent parameter counts. FLUX.1-schnell at 4-bit is hard to beat for photorealistic output. This swap makes sense if image generation becomes a high-frequency workload; otherwise FLUX stays the pragmatic choice.

### Embeddings: all-MiniLM → NV-Embed-v2 (or similar decoder-based embedder)

State-of-the-art embedding models have shifted to decoder-only LLMs (append `[EOS]`, use its hidden state as the embedding). While embeddings are a prefill task where sparse SV doesn't engage, **KV prefix caching** is the win: a cached library of clinical guidelines in 3-bit PolarQuant memory lets subsequent queries reuse the attention states without recomputation.

**Trade-off:** NV-Embed-v2 is much larger than all-MiniLM-L6-v2 (7B vs. 22M parameters). Only worth the memory cost if RAG quality or prefix-cache reuse is a bottleneck.
