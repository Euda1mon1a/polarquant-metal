// kernels/polarquant_sv_sparse.metal
//
// Phase 3 — Sparse SV matmul using compact active-position index.
//
// Instead of iterating all L_kv positions (dense path), this kernel reads
// the compact index built by polarquant_sv_index_build and iterates only
// the active positions. At 99% skip rate (concentrated heads at 32K context),
// this reduces inner loop iterations from 32,768 to ~256 — 5.6x measured.
//
// Grid: (B * n_heads * D, 1, 1) — one thread per output element.
// Decode-only (L_q = 1).
//
// Concrete 3-bit instantiation for build-time validation.
// At runtime, MLX instantiates for 2/3/4-bit via _build_sv_sparse_kernel(bits).
//
// IMPORTANT: This file is NOT loaded at runtime.
//   Runtime uses mx.fast.metal_kernel(name="polarquant_sv_sparse_3bit", ...).
//   This file exists for build-time MSL validation (zig build → xcrun metal).

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Unpack helper — shared with QK kernel (from _QK_KERNEL_HEADER in kernels.py)
// ---------------------------------------------------------------------------
template <int BITS>
inline uint unpack_index(const device uint32_t* packed, uint idx) {
    constexpr uint vals_per_int = 32 / BITS;
    constexpr uint mask = (1u << BITS) - 1u;
    uint word_idx  = idx / vals_per_int;
    uint bit_offset = (idx % vals_per_int) * BITS;
    return (packed[word_idx] >> bit_offset) & mask;
}

// ---------------------------------------------------------------------------
// Sparse SV kernel — 3-bit concrete instantiation
// ---------------------------------------------------------------------------
kernel void polarquant_sv_sparse_3bit(
    // Inputs (matches input_names in _build_sv_sparse_kernel)
    device const uint*      count_and_indices [[buffer(0)]],  // compact index
    device const half*      wn               [[buffer(1)]],   // (B, n_heads, 1, L_kv) combined
    device const uint32_t*  v_indices        [[buffer(2)]],   // (B, n_kv_heads, L_kv, D_packed) packed
    device const half*      v_centroids      [[buffer(3)]],   // (n_levels,) codebook
    device const uint*      actual_dim       [[buffer(4)]],   // scalar: head_dim D
    device const uint*      max_L_kv         [[buffer(5)]],   // scalar: L_kv for stride computation
    // Output
    device half*            out              [[buffer(6)]],   // (B, n_heads, 1, D)
    // Shape arrays
    device const uint*      count_and_indices_shape [[buffer(7)]],
    device const uint*      wn_shape               [[buffer(8)]],
    device const uint*      v_indices_shape         [[buffer(9)]],
    device const uint*      v_centroids_shape       [[buffer(10)]],
    device const uint*      actual_dim_shape        [[buffer(11)]],
    device const uint*      max_L_kv_shape          [[buffer(12)]],
    // Thread position
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    constexpr int BITS = 3;

    uint elem = thread_position_in_grid.x;

    uint actual_D  = actual_dim[0];
    uint L_kv      = max_L_kv[0];
    uint n_heads   = wn_shape[1];
    uint n_kv_heads = v_indices_shape[1];

    uint d_idx = elem % actual_D;
    uint h_idx = (elem / actual_D) % n_heads;
    uint b_idx = elem / (actual_D * n_heads);

    // GQA: map query head to kv head
    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (actual_D + vals_per_int - 1) / vals_per_int;

    // Stride: each head occupies (1 + L_kv) slots in count_and_indices
    uint stride    = 1 + L_kv;
    uint head_base = (b_idx * n_heads + h_idx) * stride;
    uint count     = count_and_indices[head_base];  // number of active positions

    uint wn_base = (b_idx * n_heads + h_idx) * L_kv;
    uint vi_base = (b_idx * n_kv_heads + kv_h_idx) * L_kv * D_packed;

    // Sparse inner loop: iterate only active positions
    float acc = 0.0f;
    for (uint i = 0; i < count; i++) {
        uint k      = count_and_indices[head_base + 1 + i];
        float wn_val = float(wn[wn_base + k]);
        uint  idx    = unpack_index<BITS>(&v_indices[vi_base + k * D_packed], d_idx);
        acc += wn_val * float(v_centroids[idx]);
    }

    out[elem] = half(acc);
}
