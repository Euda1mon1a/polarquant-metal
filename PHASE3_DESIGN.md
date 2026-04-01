# Phase 3: Compact-Index Sparse SV Kernel

> **Status:** Design — not implemented
> **Prerequisite:** Phase 2a (per-head threshold) shipped, Exp 9 (workload prediction) validated premise
> **Target:** SV kernel bottleneck (88% of decode time at 32K)

## Problem

The SV kernel iterates over ALL L_kv positions per output element:

```metal
for (uint k = 0; k < L_kv; k++) {           // 16,384 iterations at 16K context
    float wn_val = wn_combined[wn_base + k];
    if (wn_val > sv_thresh || wn_val < -sv_thresh) {  // branch per position
        uint idx = unpack_index(&v_indices[...], d_idx);
        acc += wn_val * v_centroids[idx];              // actual work
    }
}
```

At 99% skip rate (concentrated heads), 16,220 of 16,384 iterations execute
only the branch — wasted loop overhead. The codebook lookup + multiply runs
on ~164 positions but the kernel pays for 16,384 iterations.

## Solution: Two-Phase Compact-Index Dispatch

### Phase 1: Build active index (new Metal kernel)

One threadgroup per head scans `wn_combined` and builds a compact list of
positions that exceed the per-head threshold.

```metal
// Kernel: polarquant_sv_index_build
// Grid: (ceil(L_kv / 256), n_heads, B)
// Output: active_indices (B, n_heads, max_active) uint32
//         active_count   (B, n_heads) uint32

threadgroup atomic_uint tg_count;
threadgroup uint tg_indices[256];  // local buffer

uint k = threadgroup_position_in_grid.x * 256 + thread_position_in_threadgroup.x;

if (k < L_kv) {
    float wn_val = wn_combined[wn_base + k];
    if (abs(wn_val) > sparse_thresh[h_idx]) {
        uint slot = atomic_fetch_add_explicit(&tg_count, 1, memory_order_relaxed);
        tg_indices[slot] = k;
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);

// Write threadgroup results to global active_indices
// (leader thread copies tg_indices to global buffer at atomic offset)
```

**Cost:** O(L_kv / 256) threadgroups, one pass. At 16K: 64 threadgroups.
Each thread does one comparison + conditional atomic — negligible vs the
current 16K-iteration inner loop.

### Phase 2: Sparse SV matmul (modified Metal kernel)

Iterate only over active positions using the compact index:

```metal
// Kernel: polarquant_sv_sparse
// Grid: (B * n_heads * L_q * D, 1, 1)  — same as current

uint count = active_count[b_idx * n_heads + h_idx];

float acc = 0.0f;
for (uint i = 0; i < count; i++) {
    uint k = active_indices[base + i];  // compact index lookup
    float wn_val = wn_combined[wn_base + k];
    uint idx = unpack_index<BITS>(&v_indices[vi_base + k * D_packed], d_idx);
    acc += wn_val * v_centroids[idx];
}

out[elem] = T(acc);
```

**Cost:** O(active_count) per output element. At 99% skip: 164 iterations
instead of 16,384 — **100x fewer loop iterations.**

### Dispatch Logic (Python side)

```python
def fused_sdpa(self, queries, scale=None, mask=None):
    ...
    weights = mx.softmax(scores, axis=-1, precise=True)

    # Phase 2a: per-head entropy thresholds (amortized every 50 steps)
    thresholds = self._get_cached_thresholds(weights)

    # Phase 3: compact-index sparse SV
    if any(t > 0 for t in thresholds):
        # Build active index (Phase 1 kernel)
        active_indices, active_count = polarquant_sv_build_index(
            wn_combined, thresholds, L_kv, n_heads)

        # Decide per-head dispatch from active_count
        # If a head has active_count > 50% of L_kv, use dense path (no index)
        # If a head has active_count == 0, skip entirely
        # Otherwise: sparse path with compact index

        out_rotated = polarquant_sv_sparse(
            active_indices, active_count,
            wn_combined, v_packed, v_centroids,
            head_dim, bits)
    else:
        # All heads threshold=0: dense path (no indexing overhead)
        out_rotated = polarquant_sv_matmul(...)
```

## Expected Performance

| Context | Current (branch per position) | Phase 3 (compact index) | Speedup |
|---------|------|---------|---------|
| 4K, 99% skip | 4K iterations | ~40 iterations + index build | ~10-50x on SV inner loop |
| 16K, 99% skip | 16K iterations | ~160 iterations + index build | ~50-100x on SV inner loop |
| 32K, 99% skip | 32K iterations | ~320 iterations + index build | ~50-100x on SV inner loop |
| 16K, 50% skip | 16K iterations | ~8K iterations + index build | ~1.5-2x |
| 16K, 0% skip | 16K iterations | 16K iterations (dense fallback) | 1.0x (no regression) |

The index build cost is O(L_kv / 256) threadgroups — ~64 at 16K — amortized
across all D output elements per head. The dense fallback ensures no regression
when thresholds are zero or skip rate is low.

## Memory Overhead

- `active_indices`: (B, n_heads, max_active) uint32. At max_active = L_kv:
  1 × 8 × 16384 × 4 = 512 KB. Allocated once, reused.
- `active_count`: (B, n_heads) uint32. 32 bytes. Negligible.

Total: ~512 KB overhead for 16K context. <1% of model memory.

## Risks

1. **Atomic contention in index build.** Multiple threads in a threadgroup
   atomically increment the counter. At 256 threads/threadgroup, contention
   is low (Metal atomics are fast in threadgroup memory). Global atomics for
   cross-threadgroup merge need care.

2. **Non-coalesced reads in sparse phase.** Active positions are non-contiguous
   in `v_indices` — random access pattern. Partially mitigated by L1 cache
   (codebook is tiny, 8 entries). The wn_combined reads are also scattered
   but each value is read once.

3. **Dense fallback correctness.** When active_count > 50% of L_kv, the
   index overhead exceeds the branch savings. Must fall back to current
   dense kernel. The 50% threshold needs tuning via benchmarking.

4. **max_active allocation.** In worst case (all positions active), needs
   L_kv entries per head. Pre-allocate at L_kv and reuse across steps.

## Relationship to Prior Experiments

- **Exp 1 (entropy):** Provides the per-head thresholds that determine skip rate
- **Exp 6 (amortization):** Reduces entropy compute cost, feeds thresholds to Phase 3
- **Exp 7 (zones):** Zone boundaries could inform max_active estimation per zone
- **Exp 9 (Erlang):** Validated that workload IS predictable (90% dispatch accuracy).
  Phase 3 replaces prediction with exact counting on-GPU.

## Experiments to Re-evaluate After Phase 3

Phase 3 changes the performance profile fundamentally — the SV inner loop
goes from O(L_kv) to O(active_count). This invalidates assumptions in
several prior experiments:

| Exp | Original Finding | Re-evaluate? | Why |
|-----|-----------------|-------------|-----|
| 5 (Hub tokens) | Negative — wn boosting degrades quality | **Yes** | With compact index, hub tokens could be added to the active index directly (no wn boosting needed). Protected positions = always in active_indices regardless of threshold. Clean implementation path that Exp 5's approach couldn't achieve. |
| 7 (Blast radius zones) | Positive — 4-bit sys_prompt | **Yes** | With compact index, system prompt positions could get a separate active index with threshold=0 (always active). Combines zone precision with sparse dispatch — system prompt computed densely, mid-context sparsely, in the same kernel. |
| 9 (Erlang) | Partial — prediction accurate but CPU-expensive | **Superseded** | Phase 3 replaces prediction with exact GPU counting. Erlang model no longer needed. But the 90% dispatch accuracy result validates that the dispatch decision (skip/sparse/dense) is predictable. |
| 4 (Spectral bit-width) | Negative — rotation kills pattern-dependence | **Maybe** | With per-zone dispatch (Phase 3 + Exp 7), different zones could use different codebook bit-widths in the sparse kernel. The rotation decorrelation argument applies per-position, but zone-level precision selection is position-based, not pattern-based. |
| 6 (Amortization) | Positive — fixed interval=50 | **Re-tune** | Phase 3 changes the cost of entropy computation relative to the SV kernel. If SV is 100x faster, the entropy compute becomes a larger fraction of total time. May need interval=100 or higher. |

## Implementation Order

1. `polarquant_sv_build_index` Metal kernel
2. `polarquant_sv_sparse` Metal kernel (modified SV inner loop)
3. Python dispatch logic in `fused_sdpa()` with dense fallback
4. Benchmark at 4K, 16K, 32K with varied skip rates
5. Tune dense/sparse crossover threshold

## Prior Art

- **Flash Attention** (Dao et al., 2022): tiled attention with shared memory,
  but operates on dense matrices. Phase 3 is sparse-specific.
- **SpargeAttn** (ICML 2025): sparse attention on CUDA with warp-level pruning.
  Phase 3 applies the same concept to Metal with PolarQuant codebook integration.
- **CSR-format SpMM:** Standard sparse matrix multiplication. Phase 3 is
  essentially CSR-SpMM fused with codebook dequantization.
