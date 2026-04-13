// kernels/polarquant_qk.metal
//
// QK attention kernel — standalone Metal source for build-time validation.
// This mirrors polarquant_metal/kernels.py:_QK_KERNEL_HEADER + _QK_KERNEL_SOURCE.
//
// IMPORTANT: This file is NOT loaded at runtime.
//   - Runtime uses mx.fast.metal_kernel() (JIT compilation from the same source).
//   - This file exists so `zig build` can catch MSL syntax errors at build time
//     rather than at Python runtime.
//
// Concrete 4-bit instantiation for compilation check.
// Numerical logic is identical to the Python-embedded version.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Helper: unpack a single BITS-bit index from a packed uint32 array.
// Matches _QK_KERNEL_HEADER in kernels.py exactly.
// ---------------------------------------------------------------------------
template <int BITS>
inline uint unpack_index(const device uint32_t* packed, uint idx) {
    constexpr uint vals_per_int = 32 / BITS;
    constexpr uint mask = (1u << BITS) - 1u;
    uint word_idx = idx / vals_per_int;
    uint bit_offset = (idx % vals_per_int) * BITS;
    return (packed[word_idx] >> bit_offset) & mask;
}

// ---------------------------------------------------------------------------
// QK kernel: fused Q @ K^T with PolarQuant dequantization.
//
// Buffer layout follows MLX fast.metal_kernel convention:
//   inputs in order, then outputs, then per-input shapes (uint32, 4 elements).
//   Threads: 1D grid, one thread per output element.
//
// Concrete 4-bit version. At runtime MLX instantiates this for 2/3/4-bit.
// ---------------------------------------------------------------------------
kernel void polarquant_qk_4bit(
    // Input buffers (matches input_names order in _build_qk_kernel)
    device const half*      queries    [[buffer(0)]],
    device const uint32_t*  indices    [[buffer(1)]],  // bit-packed key centroids
    device const float*     norms      [[buffer(2)]],
    device const half*      centroids  [[buffer(3)]],
    device const float*     scale      [[buffer(4)]],
    // Output
    device half*            out        [[buffer(5)]],
    // Shape arrays (MLX appends one uint32* per input, 4 elements: B/H/Lq/Lkv etc.)
    device const uint32_t*  queries_shape    [[buffer(6)]],
    device const uint32_t*  indices_shape    [[buffer(7)]],
    device const uint32_t*  norms_shape      [[buffer(8)]],
    device const uint32_t*  centroids_shape  [[buffer(9)]],
    device const uint32_t*  scale_shape      [[buffer(10)]],
    // Thread indexing
    uint3  thread_position_in_grid [[thread_position_in_grid]]
) {
    constexpr int BITS = 4;

    uint elem = thread_position_in_grid.x;

    // Decode output indices from flat element index.
    // Output shape: (B, n_heads, L_q, L_kv)
    uint L_kv    = indices_shape[2];
    uint L_q     = queries_shape[2];
    uint n_heads = queries_shape[1];
    uint n_kv_heads = indices_shape[1];
    uint D       = queries_shape[3];

    uint k_idx = elem % L_kv;
    uint q_idx = (elem / L_kv) % L_q;
    uint h_idx = (elem / L_kv / L_q) % n_heads;
    uint b_idx = elem / (L_kv * L_q * n_heads);

    // GQA: map query head to kv head
    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (D + vals_per_int - 1) / vals_per_int;

    // queries: (B, n_heads, L_q, D)
    uint q_offset = ((b_idx * n_heads + h_idx) * L_q + q_idx) * D;
    // indices: (B, n_kv_heads, L_kv, D_packed)
    uint k_offset = ((b_idx * n_kv_heads + kv_h_idx) * L_kv + k_idx) * D_packed;
    // norms: (B, n_kv_heads, L_kv, 1)
    uint n_offset = (b_idx * n_kv_heads + kv_h_idx) * L_kv + k_idx;

    // Accumulate dot product: sum_d query[d] * centroid[index[d]]
    float acc = 0.0f;
    for (uint d = 0; d < D; d++) {
        float q_val = float(queries[q_offset + d]);
        uint  idx   = unpack_index<BITS>(&indices[k_offset], d);
        float k_val = float(centroids[idx]);
        acc += q_val * k_val;
    }

    // Scale by key vector norm and attention scale
    acc *= float(norms[n_offset]) * scale[0];

    out[elem] = half(acc);
}
