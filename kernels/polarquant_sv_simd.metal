// kernels/polarquant_sv_simd.metal
//
// Simdgroup-reduced SV kernels — concrete 3-bit instantiations for Zig build-time
// MSL validation.
//
// Two kernels:
//   polarquant_sv_simd_3bit       — dense path, handles L_q ≥ 1 (decode + prefill)
//   polarquant_sv_simd_sparse_3bit — compact-index sparse path, decode (L_q=1)
//
// Both use 32 lanes per output element to parallelize the inner loop over
// K/V positions.  simd_sum(partial) collapses the partial sums.
//
// IMPORTANT: These files are NOT loaded at runtime.
//   Runtime uses mx.fast.metal_kernel() with JIT compilation.
//   This file exists for build-time MSL validation (zig build → xcrun metal).

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Unpack helper — shared with all other kernels
// ---------------------------------------------------------------------------
template <int BITS>
inline uint unpack_index(const device uint32_t* packed, uint idx) {
    constexpr uint vals_per_int = 32 / BITS;
    constexpr uint mask = (1u << BITS) - 1u;
    uint word_idx   = idx / vals_per_int;
    uint bit_offset = (idx % vals_per_int) * BITS;
    return (packed[word_idx] >> bit_offset) & mask;
}

// ---------------------------------------------------------------------------
// polarquant_sv_simd_3bit — dense simdgroup SV, L_q ≥ 1
//
// Grid: (B * n_heads * L_q * D * 32, 1, 1), Threadgroup: (32, 1, 1)
// Each simdgroup computes one (b, h, q, d) output element.
// Lanes stride over L_kv; simd_sum reduces.
// ---------------------------------------------------------------------------
kernel void polarquant_sv_simd_3bit(
    device const half*      wn_combined              [[buffer(0)]],  // (B, n_heads, L_q, L_kv)
    device const uint32_t*  v_indices                [[buffer(1)]],  // (B, n_kv_heads, L_kv, D_packed)
    device const half*      v_centroids              [[buffer(2)]],  // (n_levels,)
    device const uint*      actual_dim               [[buffer(3)]],  // scalar
    device const float*     sparse_thresh            [[buffer(4)]],  // (n_heads,)
    device half*            out                      [[buffer(5)]],  // (B, n_heads, L_q, D)
    device const uint*      wn_combined_shape        [[buffer(6)]],
    device const uint*      v_indices_shape          [[buffer(7)]],
    device const uint*      v_centroids_shape        [[buffer(8)]],
    device const uint*      actual_dim_shape         [[buffer(9)]],
    device const uint*      sparse_thresh_shape      [[buffer(10)]],
    uint3 thread_position_in_grid         [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup  [[thread_position_in_threadgroup]]
) {
    constexpr int BITS = 3;
    constexpr uint SIMD_SIZE = 32;

    uint gid  = thread_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;  // 0..31
    uint elem = gid / SIMD_SIZE;

    uint actual_D   = actual_dim[0];
    uint L_q        = wn_combined_shape[2];
    uint L_kv       = wn_combined_shape[3];
    uint n_heads    = wn_combined_shape[1];
    uint n_kv_heads = v_indices_shape[1];

    uint d_idx = elem % actual_D;
    uint q_idx = (elem / actual_D) % L_q;
    uint h_idx = (elem / actual_D / L_q) % n_heads;
    uint b_idx = elem / (actual_D * L_q * n_heads);

    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (actual_D + vals_per_int - 1) / vals_per_int;

    uint wn_base = ((b_idx * n_heads + h_idx) * L_q + q_idx) * L_kv;
    uint vi_base = ((b_idx * n_kv_heads + kv_h_idx) * L_kv) * D_packed;

    float sv_thresh = sparse_thresh[h_idx];

    // Each lane processes a stride-32 subset of L_kv
    float partial = 0.0f;
    for (uint k = lane; k < L_kv; k += SIMD_SIZE) {
        float wn_val = float(wn_combined[wn_base + k]);
        if (wn_val > sv_thresh || wn_val < -sv_thresh) {
            uint idx = unpack_index<BITS>(&v_indices[vi_base + k * D_packed], d_idx);
            partial += wn_val * float(v_centroids[idx]);
        }
    }

    // Reduce across simdgroup — single instruction on Apple Silicon
    float acc = simd_sum(partial);

    // Only lane 0 writes (avoids 32x redundant stores)
    if (lane == 0) {
        out[elem] = half(acc);
    }
}

// ---------------------------------------------------------------------------
// polarquant_sv_simd_sparse_3bit — compact-index sparse SV, decode (L_q=1)
//
// Grid: (B * n_heads * D * 32, 1, 1), Threadgroup: (32, 1, 1)
// Each simdgroup computes one (b, h, d) output element.
// Lanes stride over count active positions; simd_sum reduces.
// ---------------------------------------------------------------------------
kernel void polarquant_sv_simd_sparse_3bit(
    device const uint*      count_and_indices        [[buffer(0)]],  // compact index
    device const half*      wn                       [[buffer(1)]],  // (B, n_heads, 1, L_kv)
    device const uint32_t*  v_indices                [[buffer(2)]],  // (B, n_kv_heads, L_kv, D_packed)
    device const half*      v_centroids              [[buffer(3)]],  // (n_levels,)
    device const uint*      actual_dim               [[buffer(4)]],  // scalar
    device const uint*      max_L_kv                 [[buffer(5)]],  // scalar
    device half*            out                      [[buffer(6)]],  // (B, n_heads, 1, D)
    device const uint*      count_and_indices_shape  [[buffer(7)]],
    device const uint*      wn_shape                 [[buffer(8)]],
    device const uint*      v_indices_shape          [[buffer(9)]],
    device const uint*      v_centroids_shape        [[buffer(10)]],
    device const uint*      actual_dim_shape         [[buffer(11)]],
    device const uint*      max_L_kv_shape           [[buffer(12)]],
    uint3 thread_position_in_grid         [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup  [[thread_position_in_threadgroup]]
) {
    constexpr int BITS = 3;
    constexpr uint SIMD_SIZE = 32;

    uint gid  = thread_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint elem = gid / SIMD_SIZE;

    uint actual_D   = actual_dim[0];
    uint L_kv       = max_L_kv[0];
    uint n_heads    = wn_shape[1];
    uint n_kv_heads = v_indices_shape[1];

    uint d_idx = elem % actual_D;
    uint h_idx = (elem / actual_D) % n_heads;
    uint b_idx = elem / (actual_D * n_heads);

    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (actual_D + vals_per_int - 1) / vals_per_int;

    uint stride    = 1 + L_kv;
    uint head_base = (b_idx * n_heads + h_idx) * stride;
    uint count     = count_and_indices[head_base];

    uint wn_base = (b_idx * n_heads + h_idx) * L_kv;
    uint vi_base = (b_idx * n_kv_heads + kv_h_idx) * L_kv * D_packed;

    // Each lane handles a stride-32 subset of active positions
    float partial = 0.0f;
    for (uint i = lane; i < count; i += SIMD_SIZE) {
        uint k       = count_and_indices[head_base + 1 + i];
        float wn_val = float(wn[wn_base + k]);
        uint  idx    = unpack_index<BITS>(&v_indices[vi_base + k * D_packed], d_idx);
        partial += wn_val * float(v_centroids[idx]);
    }

    // Reduce across simdgroup — single instruction on Apple Silicon
    float acc = simd_sum(partial);

    if (lane == 0) {
        out[elem] = half(acc);
    }
}
