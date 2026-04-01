# PolarQuant Metal — Kernel Optimization Roadmap

This document records the next layer of Metal kernel improvements identified from
external research, grounded in the actual dispatch code in `polarquant_metal/kernels.py`.
Items are ordered by effort-to-impact ratio.

---

## 1. `use_optimal_threadgroups=True` — ~13% for free

**Status:** Not implemented. Zero occurrences in `kernels.py`.
**Effort:** 10 minutes — add one keyword argument to each of the 6 dispatch sites.
**Source:** MLX PR #1833.

Every `mx.fast.metal_kernel()` dispatch in PolarQuant uses the default `dispatch_threads`
path. MLX's `use_optimal_threadgroups=True` flag switches to `dispatch_threadgroups`
(Metal's preferred dispatch mode), which achieves ~13% higher throughput on M3 Max for
moderately complex kernels. The win comes from better occupancy — Metal can schedule
threadgroups more efficiently when it controls packing rather than mapping threads
individually.

The current dispatch pattern throughout `kernels.py`:
```python
kernel(...,
    grid=(total_elements, 1, 1),
    threadgroup=(min(256, total_elements), 1, 1),
)
```

All 6 dispatch sites need `use_optimal_threadgroups=True` added. The tiled QK matmul
(lines ~586-587) uses a 2D grid — this one warrants a quick correctness check after
the change since `dispatch_threadgroups` changes the thread indexing semantics slightly.
The 1D dispatch sites are straightforward.

This is the single highest return-on-investment change available right now.

---

## 2. Threadgroup Memory for Per-Token Norms — 7–14% prefill improvement

**Status:** Not implemented.
**Effort:** 1–2 hours per kernel (MSL kernel body edit).
**Source:** MLX issue #3251 (`group_size=32` throughput regression root cause analysis).

PolarQuant's fused kernels (`polarquant_qk_matmul`, `polarquant_sv_matmul`) read the
per-token norm vectors (`k_norms`, `v_norms`) from device memory on every thread. These
are analogous to the `scale/bias` tables in standard quantized GEMM — small per-token
floats that many threads need simultaneously.

The MLX issue documented 7–14% throughput loss and up to 2x slower prefill on M2 Ultra
and M4 Max when norm values weren't cached in threadgroup shared memory. The root cause:
device memory reads per-thread cause repeated global memory traffic that saturates the
memory subsystem, even on Apple Silicon's high-bandwidth unified memory.

The fix is to load the norm slice for the current tile into a `threadgroup float[]` array
at the start of the kernel, then reference the threadgroup copy for all per-token scale
operations. For PolarQuant this applies to both the QK and SV kernels.

Worth doing after `use_optimal_threadgroups` since profiling (see §4) should confirm
whether norms are actually the bottleneck at the context lengths we care about.

---

## 3. Automated Kernel Evolution via OpenEvolve — potential 10–15% from search

**Status:** Not attempted.
**Effort:** 1–3 hours setup + overnight compute.
**Source:** OpenEvolve HuggingFace blog; r/LocalLLaMA thread on Qwen3 GQA optimization.

OpenEvolve (Google DeepMind AlphaEvolve fork) uses an LLM to mutate Metal kernel source
in a loop, evaluating each mutation for correctness + throughput. It found 12.5% attention
throughput improvements for Qwen3's 40:8 GQA head ratio by searching tile sizes and memory
access patterns that no human had tried.

PolarQuant's context: Qwen2.5-72B uses 64 Q heads / 8 KV heads (8:1 GQA ratio) — similar
to the Qwen3 topology that was optimized. The `polarquant_qk_matmul` kernel's tile
configuration and the shared-memory layout for codebook centroids are the natural search
targets.

Setup is low-friction:
```bash
pip install openevolve
cd openevolve/examples/mlx_metal_kernel_opt
./run_evolve_experiment.sh --run-name pq_qk_matmul --iterations 25
```

The main integration work is wrapping PolarQuant's kernel in the OpenEvolve evaluation
harness (correctness check against reference + throughput measurement). Worth doing after
the manual optimizations in §1–2 are baselined, so the automated search starts from a
better initial point.

---

## 4. Metal GPU Frame Capture for Real Profiling

**Status:** Not done.
**Effort:** 30 minutes setup; then use findings to prioritize §1–3.
**Source:** `mx.metal.capture()` MLX API.

All PolarQuant performance analysis so far has been behavioral (tok/s at the Python
level). We don't know from measurements where cycles actually go inside
`polarquant_qk_matmul` at 8K+ context. The estimates in §2 (norms as bottleneck) are
informed inference, not profiler data.

```python
with mx.metal.capture("pq_qk_trace.gputrace"):
    out = polarquant_qk_matmul(q_rotated, k_packed, k_norms, centroids, scale, bits)
    mx.force_eval(out)  # trigger execution inside capture window
```

The resulting `.gputrace` opens in Xcode Instruments and shows per-shader execution time,
memory bandwidth utilization, and threadgroup occupancy. This would confirm or deny:
- Whether norms are actually the memory bottleneck (§2)
- Whether the tiled QK kernel has occupancy issues at the current threadgroup sizes
- Where the Phase 3 sparse SV kernel spends its time at 8K–16K context

Do this before spending time on threadgroup memory edits — if the profiler shows
something else is the actual bottleneck, it reorders the priority list entirely.

---

## 5. The `group_size` Quantization Note (External Reference)

Not a PolarQuant action item, but worth recording: the MLX community identified 7–14%
throughput loss and 2x prefill regression on `group_size=32` quantized models compared
to `group_size=128`, caused by the MLX QMV kernel's unoptimized scale/bias reads. If
PolarQuant is used alongside standard `mlx_lm` quantized models (e.g., the 7B draft in
speculative decoding), prefer `group_size=128` for those models. The Qwen2.5-7B-Instruct-4bit
used in Phase A/B speculative decoding — worth checking which `group_size` it was
quantized with before assuming the draft model isn't contributing overhead.

---

## Priority Order Summary

| # | Change | Effort | Expected gain | Dependency |
|---|--------|--------|---------------|------------|
| 1 | `use_optimal_threadgroups=True` | 10 min | ~13% | None |
| 4 | Metal GPU Frame Capture profile | 30 min | Informs everything | Do before §2 |
| 2 | Threadgroup memory for norms | 1–2 hr | 7–14% | Profile confirms bottleneck |
| 3 | OpenEvolve automated search | 3 hr + overnight | 10–15% additional | After §1+2 baselined |

The current codebase has never been profiled at the Metal level. `use_optimal_threadgroups`
should be added first (zero risk, zero logic change), then a profiling pass before any
further kernel body edits.
