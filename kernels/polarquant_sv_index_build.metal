// kernels/polarquant_sv_index_build.metal
//
// Phase 3 — Compact-index builder for sparse SV kernel.
//
// Scans wn_combined per-head and collects positions where
//   |wn| > sparse_thresh[h]  OR  zone_prior[k] == 1
// into a compact index. Zone priors ensure system-prompt tokens
// (always-relevant) are never incorrectly pruned.
//
// Grid: (L_kv, 1, 1) — one thread per KV position.
// Each thread iterates all B*n_heads and writes to the correct slot atomically.
//
// Output layout (per head): [count, idx0, idx1, ..., idx_{L_kv-1}]
// stride = 1 + L_kv
//
// IMPORTANT: This file is NOT loaded at runtime.
//   Runtime uses mx.fast.metal_kernel() with atomic_outputs=True.
//   This file exists for build-time MSL validation (zig build → xcrun metal).

#include <metal_stdlib>
using namespace metal;

kernel void polarquant_sv_index_build(
    // Inputs (matches input_names order)
    device const half*    wn            [[buffer(0)]],  // (B, n_heads, L_q, L_kv)
    device const float*   sparse_thresh [[buffer(1)]],  // (n_heads,)
    device const uint*    zone_prior    [[buffer(2)]],  // (L_kv,)  1 = always active
    // Output — atomic because multiple threads write per-head
    device atomic_uint*   count_and_indices [[buffer(3)]],
    // Shape arrays (MLX convention: one uint32* per input)
    device const uint*    wn_shape            [[buffer(4)]],  // [B, n_heads, L_q, L_kv]
    device const uint*    sparse_thresh_shape [[buffer(5)]],
    device const uint*    zone_prior_shape    [[buffer(6)]],
    // Thread position
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint k     = thread_position_in_grid.x;
    uint L_kv  = wn_shape[3];
    uint n_heads = wn_shape[1];

    if (k >= L_kv) return;

    // stride: each head occupies (1 + L_kv) slots [count, idx0..idx_{L_kv-1}]
    uint stride = 1 + L_kv;

    for (uint bh = 0; bh < wn_shape[0] * n_heads; bh++) {
        uint b_idx = bh / n_heads;
        uint h_idx = bh % n_heads;

        // wn layout: (B, n_heads, L_q, L_kv) — L_q=1 at decode
        uint wn_offset = ((b_idx * n_heads + h_idx) * wn_shape[2]) * L_kv + k;
        float wn_val  = float(wn[wn_offset]);
        float thresh  = sparse_thresh[h_idx];
        bool is_prior = zone_prior[k] > 0u;

        if (is_prior || wn_val > thresh || wn_val < -thresh) {
            uint head_base = bh * stride;
            // Atomically claim a slot in this head's index
            uint slot = atomic_fetch_add_explicit(
                &count_and_indices[head_base], 1u, memory_order_relaxed);
            // Write position k into the claimed slot
            atomic_store_explicit(
                &count_and_indices[head_base + 1 + slot], k, memory_order_relaxed);
        }
    }
}
