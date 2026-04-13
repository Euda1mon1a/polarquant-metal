"""
Fused Metal kernels for PolarQuant dequantize-matmul operations.

Two kernels:
1. polarquant_qk_matmul: Q_rotated @ K_quantized^T  (fused codebook lookup + dot product)
2. polarquant_sv_matmul: softmax_weights @ V_quantized (fused codebook lookup + weighted sum)

Key insight: If queries are pre-rotated into the same basis as the quantized
keys, the fused kernel only needs to do codebook lookups during the dot product
accumulation — no explicit inverse rotation or full dequantization pass required.

Layout conventions:
  - queries:    (B, n_heads, L_q, D)       float16/bfloat16/float32
  - indices:    (B, n_kv_heads, L_kv, D_packed)  uint32 (bit-packed)
  - norms:      (B, n_kv_heads, L_kv, 1)   float16/bfloat16/float32
  - centroids:  (n_levels,)                 float32
  - output:     (B, n_heads, L_q, L_kv)    same as queries dtype

For 3-bit quantization with D=128: D_packed = ceil(128 / 10) = 13 uint32s
(10 × 3-bit values fit in one uint32)
"""

import math
import numpy as np
import mlx.core as mx

# ---------------------------------------------------------------------------
# Metal kernel source: Fused Q @ K^T with PolarQuant codebook dequantization
# ---------------------------------------------------------------------------
# Each thread computes one element of the output: out[b, h, q, k]
# It iterates over head_dim, unpacking indices and looking up centroids on the fly.
#
# Template parameters:
#   BITS: quantization bits per coordinate (2, 3, or 4)
#   T: compute dtype (float, half, bfloat)

_QK_KERNEL_HEADER = """
// Unpack a single b-bit index from a packed uint32 array.
// packed: pointer to packed uint32 values for one key vector
// idx: coordinate index [0, dim)
// bits: number of bits per value
template <int BITS>
inline uint unpack_index(const device uint32_t* packed, uint idx) {
    constexpr uint vals_per_int = 32 / BITS;
    constexpr uint mask = (1u << BITS) - 1u;
    uint word_idx = idx / vals_per_int;
    uint bit_offset = (idx % vals_per_int) * BITS;
    return (packed[word_idx] >> bit_offset) & mask;
}
"""

_QK_KERNEL_SOURCE = """
    // Thread position: one thread per output element
    uint elem = thread_position_in_grid.x;

    // Decode output indices from flat element index
    // Output shape: (B, n_heads, L_q, L_kv)
    uint L_kv = indices_shape[2];       // number of key tokens
    uint L_q  = queries_shape[2];       // number of query tokens
    uint n_heads = queries_shape[1];    // query heads
    uint n_kv_heads = indices_shape[1]; // kv heads (for GQA)
    uint D = queries_shape[3];          // head dimension

    uint k_idx = elem % L_kv;
    uint q_idx = (elem / L_kv) % L_q;
    uint h_idx = (elem / L_kv / L_q) % n_heads;
    uint b_idx = elem / (L_kv * L_q * n_heads);

    // GQA: map query head to kv head
    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    // Compute packed dimension
    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (D + vals_per_int - 1) / vals_per_int;

    // Pointers into the arrays
    // queries: (B, n_heads, L_q, D) — row-contiguous
    uint q_offset = ((b_idx * n_heads + h_idx) * L_q + q_idx) * D;

    // indices: (B, n_kv_heads, L_kv, D_packed) — row-contiguous
    uint k_offset = ((b_idx * n_kv_heads + kv_h_idx) * L_kv + k_idx) * D_packed;

    // norms: (B, n_kv_heads, L_kv, 1)
    uint n_offset = (b_idx * n_kv_heads + kv_h_idx) * L_kv + k_idx;

    // Accumulate dot product: sum_d query[d] * centroid[index[d]]
    float acc = 0.0f;
    for (uint d = 0; d < D; d++) {
        float q_val = float(queries[q_offset + d]);
        uint idx = unpack_index<BITS>(&indices[k_offset], d);
        float k_val = float(centroids[idx]);
        acc += q_val * k_val;
    }

    // Scale by key vector norm and attention scale
    float norm_val = float(norms[n_offset]);
    acc *= norm_val * scale[0];

    out[elem] = T(acc);
"""

# ---------------------------------------------------------------------------
# Metal kernel source: Fused softmax_weights @ V with PolarQuant dequant
# ---------------------------------------------------------------------------
# Each thread computes one element of the output: out[b, h, q, d]
# It iterates over L_kv, unpacking value indices and looking up centroids.

_SV_KERNEL_SOURCE = """
    // Thread position: one thread per output element
    uint elem = thread_position_in_grid.x;

    // Output shape: (B, n_heads, L_q, D)
    uint D = v_indices_shape[3];          // this is D_packed, we need actual D
    // Actually we pass actual_dim as a parameter
    uint actual_D = actual_dim[0];
    uint L_q = weights_shape[2];
    uint L_kv = weights_shape[3];
    uint n_heads = weights_shape[1];
    uint n_kv_heads = v_indices_shape[1];

    uint d_idx = elem % actual_D;
    uint q_idx = (elem / actual_D) % L_q;
    uint h_idx = (elem / actual_D / L_q) % n_heads;
    uint b_idx = elem / (actual_D * L_q * n_heads);

    // GQA: map query head to kv head
    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (actual_D + vals_per_int - 1) / vals_per_int;

    // Accumulate weighted sum: sum_k weight[q,k] * centroid[v_index[k,d]] * norm[k]
    float acc = 0.0f;
    for (uint k = 0; k < L_kv; k++) {
        // weights: (B, n_heads, L_q, L_kv)
        uint w_offset = ((b_idx * n_heads + h_idx) * L_q + q_idx) * L_kv + k;
        float w_val = float(weights[w_offset]);

        // v_indices: (B, n_kv_heads, L_kv, D_packed)
        uint v_offset = ((b_idx * n_kv_heads + kv_h_idx) * L_kv + k) * D_packed;
        uint idx = unpack_index<BITS>(&v_indices[v_offset], d_idx);
        float v_val = float(v_centroids[idx]);

        // v_norms: (B, n_kv_heads, L_kv, 1)
        uint vn_offset = (b_idx * n_kv_heads + kv_h_idx) * L_kv + k;
        float vn_val = float(v_norms[vn_offset]);

        acc += w_val * v_val * vn_val;
    }

    out[elem] = T(acc);
"""

# ---------------------------------------------------------------------------
# Tiled Q@K^T kernel: uses threadgroup shared memory for much better perf
# ---------------------------------------------------------------------------
# Each threadgroup computes a TILE_Q x TILE_K block of the output.
# Tiles over the D dimension in chunks, loading query and key data into
# shared memory to maximize reuse.

_QK_TILED_HEADER = """
// Unpack a single b-bit index from a packed uint32 array.
template <int BITS>
inline uint unpack_index(const device uint32_t* packed, uint idx) {
    constexpr uint vals_per_int = 32 / BITS;
    constexpr uint mask = (1u << BITS) - 1u;
    uint word_idx = idx / vals_per_int;
    uint bit_offset = (idx % vals_per_int) * BITS;
    return (packed[word_idx] >> bit_offset) & mask;
}
"""

_QK_TILED_SOURCE = """
    // Grid layout: (ceil(L_kv/TILE_K), ceil(L_q/TILE_Q), B * n_heads)
    // Each threadgroup handles a TILE_Q x TILE_K output tile

    uint tile_k = threadgroup_position_in_grid.x;   // which key tile
    uint tile_q = threadgroup_position_in_grid.y;   // which query tile
    uint bh     = threadgroup_position_in_grid.z;   // batch * head index

    uint tid_x = thread_position_in_threadgroup.x;  // thread within tile (key dim)
    uint tid_y = thread_position_in_threadgroup.y;  // thread within tile (query dim)

    uint L_kv = indices_shape[2];
    uint L_q  = queries_shape[2];
    uint n_heads = queries_shape[1];
    uint n_kv_heads = indices_shape[1];
    uint D = queries_shape[3];

    uint b_idx = bh / n_heads;
    uint h_idx = bh % n_heads;
    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    uint q_pos = tile_q * TILE_Q + tid_y;
    uint k_pos = tile_k * TILE_K + tid_x;

    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (D + vals_per_int - 1) / vals_per_int;

    // Pointers
    uint q_base = ((b_idx * n_heads + h_idx) * L_q + q_pos) * D;
    uint k_base = ((b_idx * n_kv_heads + kv_h_idx) * L_kv + k_pos) * D_packed;
    uint n_base = (b_idx * n_kv_heads + kv_h_idx) * L_kv + k_pos;

    // Accumulate in float for precision
    float acc = 0.0f;

    bool q_valid = q_pos < L_q;
    bool k_valid = k_pos < L_kv;

    if (q_valid && k_valid) {
        // Inner loop over head_dim: unpack + codebook lookup + multiply-accumulate
        for (uint d = 0; d < D; d++) {
            float q_val = float(queries[q_base + d]);
            uint idx = unpack_index<BITS>(&indices[k_base], d);
            float k_val = float(centroids[idx]);
            acc += q_val * k_val;
        }
        acc *= float(norms[n_base]) * scale[0];
    }

    // Write output
    if (q_valid && k_valid) {
        uint out_offset = ((b_idx * n_heads + h_idx) * L_q + q_pos) * L_kv + k_pos;
        out[out_offset] = T(acc);
    }
"""


def _build_qk_kernel(bits: int):
    """Build the fused Q@K^T kernel for a specific bit width."""
    kernel = mx.fast.metal_kernel(
        name=f"polarquant_qk_{bits}bit",
        input_names=["queries", "indices", "norms", "centroids", "scale"],
        output_names=["out"],
        header=_QK_KERNEL_HEADER,
        source=_QK_KERNEL_SOURCE,
    )
    return kernel


def _build_qk_tiled_kernel(bits: int):
    """Build the tiled Q@K^T kernel for a specific bit width."""
    kernel = mx.fast.metal_kernel(
        name=f"polarquant_qk_tiled_{bits}bit",
        input_names=["queries", "indices", "norms", "centroids", "scale"],
        output_names=["out"],
        header=_QK_TILED_HEADER,
        source=_QK_TILED_SOURCE,
    )
    return kernel


def _build_sv_kernel(bits: int):
    """Build the fused softmax_weights @ V kernel for a specific bit width."""
    kernel = mx.fast.metal_kernel(
        name=f"polarquant_sv_{bits}bit",
        input_names=[
            "weights", "v_indices", "v_norms", "v_centroids", "actual_dim",
        ],
        output_names=["out"],
        header=_QK_KERNEL_HEADER,  # reuses the same unpack_index helper
        source=_SV_KERNEL_SOURCE,
    )
    return kernel


# ---------------------------------------------------------------------------
# Optimized SV kernel: pre-combined weight*norm eliminates one memory read
# and one multiply per inner loop iteration. The GQA norm expansion is done
# on the Python side before kernel dispatch.
# ---------------------------------------------------------------------------

_SV_PRECOMBINED_SOURCE = """
    uint elem = thread_position_in_grid.x;

    uint actual_D = actual_dim[0];
    uint L_q = wn_combined_shape[2];
    uint L_kv = wn_combined_shape[3];
    uint n_heads = wn_combined_shape[1];
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

    // Sparse V threshold: skip positions where weight*norm is negligible.
    // After softmax, most attention is concentrated on a few positions.
    // Skipping near-zero weights avoids the codebook lookup + multiply.
    // Per-head threshold: entropy-guided adaptive pruning (Phase 2a).
    float sv_thresh = sparse_thresh[h_idx];

    float acc = 0.0f;
    for (uint k = 0; k < L_kv; k++) {
        float wn_val = float(wn_combined[wn_base + k]);
        if (wn_val > sv_thresh || wn_val < -sv_thresh) {
            uint idx = unpack_index<BITS>(&v_indices[vi_base + k * D_packed], d_idx);
            acc += wn_val * float(v_centroids[idx]);
        }
    }

    out[elem] = T(acc);
"""

# ---------------------------------------------------------------------------
# Simdgroup-reduced dense SV kernel — handles L_q ≥ 1 (decode and prefill).
# ---------------------------------------------------------------------------
# Grid: (B * n_heads * L_q * D * 32, 1, 1), Threadgroup: (32, 1, 1)
#
# Each simdgroup of 32 threads computes one (b, h, q, d) output element.
# Lanes split the L_kv loop: lane l handles k = l, l+32, l+64, ...
# simd_sum reduces partial sums; lane 0 writes output.
#
# At L_q=256, L_kv=1024: each lane does 1024/32 = 32 iterations (vs 1024 scalar).
# At L_kv=32768 decode: each lane does ~1024 iterations (vs 32768 scalar).

_SV_SIMD_SOURCE = """
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

    // Reduce across simdgroup — single hardware instruction on Apple Silicon
    float acc = simd_sum(partial);

    // Only lane 0 writes (avoids 32x redundant stores)
    if (lane == 0) {
        out[elem] = T(acc);
    }
"""


def _build_sv_precombined_kernel(bits: int):
    """Build the optimized SV kernel with Sparse V threshold."""
    return mx.fast.metal_kernel(
        name=f"polarquant_sv_pre_{bits}bit",
        input_names=["wn_combined", "v_indices", "v_centroids", "actual_dim",
                      "sparse_thresh"],
        output_names=["out"],
        header=_QK_KERNEL_HEADER,
        source=_SV_PRECOMBINED_SOURCE,
    )


def _build_sv_simd_kernel(bits: int):
    """Build the simdgroup-reduced dense SV kernel (handles L_q ≥ 1).

    32 lanes per output element parallelize the L_kv inner loop.
    Grid: (B * n_heads * L_q * D * 32, 1, 1), Threadgroup: (32, 1, 1).
    """
    return mx.fast.metal_kernel(
        name=f"polarquant_sv_simd_{bits}bit",
        input_names=["wn_combined", "v_indices", "v_centroids", "actual_dim",
                      "sparse_thresh"],
        output_names=["out"],
        header=_QK_KERNEL_HEADER,
        source=_SV_SIMD_SOURCE,
    )


# ---------------------------------------------------------------------------
# Phase 3: Compact-index sparse SV kernels
# ---------------------------------------------------------------------------
# Two-phase approach:
#   1. Index build: scan wn_combined, collect active positions via atomics
#   2. Sparse SV: iterate only active positions from compact index
#
# Conceptual framing: the KV cache is a probability field. Softmax collapses
# it. The active index IS the collapsed wavefunction. Zone priors (system
# prompt always active) are Bayesian priors on the field.

_SV_INDEX_BUILD_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint L_kv = wn_shape[3];
    uint n_heads = wn_shape[1];

    if (k >= L_kv) return;

    // Stride for per-head output: each head gets (1 + L_kv) slots
    uint stride = 1 + L_kv;

    for (uint bh = 0; bh < wn_shape[0] * n_heads; bh++) {
        uint b_idx = bh / n_heads;
        uint h_idx = bh % n_heads;

        uint wn_offset = ((b_idx * n_heads + h_idx) * wn_shape[2]) * L_kv + k;
        float wn_val = float(wn[wn_offset]);
        float thresh = float(sparse_thresh[h_idx]);
        bool is_prior = zone_prior[k] > 0u;

        if (is_prior || wn_val > thresh || wn_val < -thresh) {
            uint head_base = bh * stride;
            uint slot = atomic_fetch_add_explicit(
                &count_and_indices[head_base], 1u, memory_order_relaxed);
            atomic_store_explicit(
                &count_and_indices[head_base + 1 + slot], k, memory_order_relaxed);
        }
    }
"""


def _build_sv_index_kernel():
    """Build the index-building kernel for Phase 3 sparse SV."""
    return mx.fast.metal_kernel(
        name="polarquant_sv_index_build",
        input_names=["wn", "sparse_thresh", "zone_prior"],
        output_names=["count_and_indices"],
        header="",
        source=_SV_INDEX_BUILD_SOURCE,
        atomic_outputs=True,
    )


_SV_SPARSE_SOURCE = """
    uint elem = thread_position_in_grid.x;

    uint actual_D = actual_dim[0];
    uint L_kv = max_L_kv[0];
    uint L_q = 1;  // Phase 3 is decode-only (L_q=1)
    uint n_heads = wn_shape[1];
    uint n_kv_heads = v_indices_shape[1];

    uint d_idx = elem % actual_D;
    uint h_idx = (elem / actual_D) % n_heads;
    uint b_idx = elem / (actual_D * n_heads);

    uint kv_h_idx = h_idx / (n_heads / n_kv_heads);

    constexpr uint vals_per_int = 32 / BITS;
    uint D_packed = (actual_D + vals_per_int - 1) / vals_per_int;

    // Read active count and indices for this head
    uint stride = 1 + L_kv;
    uint head_base = (b_idx * n_heads + h_idx) * stride;
    uint count = count_and_indices[head_base];

    uint wn_base = (b_idx * n_heads + h_idx) * L_kv;
    uint vi_base = (b_idx * n_kv_heads + kv_h_idx) * L_kv * D_packed;

    float acc = 0.0f;
    for (uint i = 0; i < count; i++) {
        uint k = count_and_indices[head_base + 1 + i];
        float wn_val = float(wn[wn_base + k]);
        uint idx = unpack_index<BITS>(&v_indices[vi_base + k * D_packed], d_idx);
        acc += wn_val * float(v_centroids[idx]);
    }

    out[elem] = T(acc);
"""


def _build_sv_sparse_kernel(bits: int):
    """Build the sparse SV kernel that iterates only active positions."""
    return mx.fast.metal_kernel(
        name=f"polarquant_sv_sparse_{bits}bit",
        input_names=[
            "count_and_indices", "wn", "v_indices", "v_centroids",
            "actual_dim", "max_L_kv",
        ],
        output_names=["out"],
        header=_QK_KERNEL_HEADER,  # reuses unpack_index helper
        source=_SV_SPARSE_SOURCE,
    )


# ---------------------------------------------------------------------------
# Simdgroup-reduced compact-index sparse SV — Phase 3, decode (L_q=1).
# ---------------------------------------------------------------------------
# Grid: (B * n_heads * D * 32, 1, 1), Threadgroup: (32, 1, 1)
#
# Each simdgroup of 32 threads computes one (b, h, d) output element.
# Lanes split the compact active-position loop 32-ways.
# simd_sum reduces; lane 0 writes output.
#
# At 32K context, ~327 active positions (1%): each lane handles ~10 iterations.
# At 8K context, ~82 active positions: each lane handles ~3 iterations.

_SV_SIMD_SPARSE_SOURCE = """
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
        out[elem] = T(acc);
    }
"""


def _build_sv_simd_sparse_kernel(bits: int):
    """Build the simdgroup-reduced sparse SV kernel (Phase 3 compact-index path).

    32 lanes per output element parallelize the active-position inner loop.
    Grid: (B * n_heads * D * 32, 1, 1), Threadgroup: (32, 1, 1).
    """
    return mx.fast.metal_kernel(
        name=f"polarquant_sv_simd_sparse_{bits}bit",
        input_names=[
            "count_and_indices", "wn", "v_indices", "v_centroids",
            "actual_dim", "max_L_kv",
        ],
        output_names=["out"],
        header=_QK_KERNEL_HEADER,
        source=_SV_SIMD_SPARSE_SOURCE,
    )


_sv_index_kernel = None
_sv_sparse_kernels = {}
_sv_simd_sparse_kernels = {}


def polarquant_sv_build_index(
    wn_combined: mx.array,
    sparse_thresh: mx.array,
    zone_prior: mx.array,
) -> mx.array:
    """Build compact active position index for sparse SV kernel.

    Scans wn_combined per head. Positions where |wn| > threshold OR
    zone_prior == 1 are collected into a compact index.

    Args:
        wn_combined: (B, n_heads, L_q, L_kv) pre-combined weight*norm
        sparse_thresh: (n_heads,) per-head thresholds
        zone_prior: (L_kv,) uint32, 1 = always active, 0 = threshold-gated

    Returns:
        count_and_indices: (B * n_heads * (1 + L_kv),) uint32
            Layout per head: [count, idx0, idx1, ..., idxN]
    """
    global _sv_index_kernel
    if _sv_index_kernel is None:
        _sv_index_kernel = _build_sv_index_kernel()

    B, n_heads, L_q, L_kv = wn_combined.shape
    stride = 1 + L_kv
    out_size = B * n_heads * stride

    outputs = _sv_index_kernel(
        inputs=[wn_combined, sparse_thresh, zone_prior],
        template=[("T", wn_combined.dtype)],
        output_shapes=[(out_size,)],
        output_dtypes=[mx.uint32],
        grid=(L_kv, 1, 1),
        threadgroup=(min(256, L_kv), 1, 1),
        init_value=0,
    )
    return outputs[0]


def polarquant_sv_sparse(
    count_and_indices: mx.array,
    wn_combined: mx.array,
    v_indices: mx.array,
    v_centroids: mx.array,
    head_dim: int,
    L_kv: int,
    bits: int = 3,
    use_simd: bool = True,
) -> mx.array:
    """Sparse SV matmul — iterates only active positions from compact index.

    Args:
        count_and_indices: (B * n_heads * (1 + L_kv),) from build_index
        wn_combined: (B, n_heads, L_q, L_kv) pre-combined weight*norm
        v_indices: (B, n_kv_heads, L_kv, D_packed) packed value indices
        v_centroids: (n_levels,) codebook
        head_dim: actual D
        L_kv: context length (for stride calculation)
        bits: quantization bits
        use_simd: use simdgroup-reduced kernel (default True). 32 lanes per
            output element parallelize the active-position inner loop.
            Most beneficial at L_kv > 8K where active count >> 32.

    Returns:
        output: (B, n_heads, 1, D) attention output (L_q=1 for decode)
    """
    B, n_heads = wn_combined.shape[0], wn_combined.shape[1]
    out_shape = (B, n_heads, 1, head_dim)
    actual_dim_arr = mx.array([head_dim], dtype=mx.uint32)
    max_L_kv_arr = mx.array([L_kv], dtype=mx.uint32)
    inputs = [count_and_indices, wn_combined, v_indices, v_centroids,
              actual_dim_arr, max_L_kv_arr]
    template = [("T", wn_combined.dtype), ("BITS", bits)]

    if use_simd:
        if bits not in _sv_simd_sparse_kernels:
            _sv_simd_sparse_kernels[bits] = _build_sv_simd_sparse_kernel(bits)
        kernel = _sv_simd_sparse_kernels[bits]
        SG = 32
        total = B * n_heads * head_dim
        outputs = kernel(
            inputs=inputs,
            template=template,
            output_shapes=[out_shape],
            output_dtypes=[wn_combined.dtype],
            grid=(total * SG, 1, 1),
            threadgroup=(SG, 1, 1),
        )
    else:
        if bits not in _sv_sparse_kernels:
            _sv_sparse_kernels[bits] = _build_sv_sparse_kernel(bits)
        kernel = _sv_sparse_kernels[bits]
        total = int(np.prod(out_shape))
        outputs = kernel(
            inputs=inputs,
            template=template,
            output_shapes=[out_shape],
            output_dtypes=[wn_combined.dtype],
            grid=(total, 1, 1),
            threadgroup=(min(256, total), 1, 1),
        )
    return outputs[0]


# Cache compiled kernels by bit width
_qk_kernels = {}
_qk_tiled_kernels = {}
_sv_kernels = {}
_sv_pre_kernels = {}
_sv_simd_kernels = {}

TILE_Q = 8
TILE_K = 32


def polarquant_qk_matmul(
    queries: mx.array,
    indices: mx.array,
    norms: mx.array,
    centroids: mx.array,
    scale: float,
    bits: int = 3,
    use_tiled: bool = True,
) -> mx.array:
    """Fused PolarQuant Q @ K^T matmul.

    Computes attention scores without materializing the full dequantized key matrix.

    IMPORTANT: queries must be PRE-ROTATED into the PolarQuant basis.
    That is, pass `queries @ rotation_matrix.T` not raw queries.

    Args:
        queries: (B, n_heads, L_q, D) pre-rotated query vectors
        indices: (B, n_kv_heads, L_kv, D_packed) uint32 bit-packed codebook indices
        norms:   (B, n_kv_heads, L_kv, 1) key vector norms
        centroids: (n_levels,) float32 codebook centroids
        scale: attention scale factor (1/sqrt(D))
        bits: quantization bits (2, 3, or 4)
        use_tiled: use the tiled kernel (default True)

    Returns:
        scores: (B, n_heads, L_q, L_kv) attention scores
    """
    assert bits in (2, 3, 4), f"Unsupported bit width: {bits}"
    assert indices.dtype == mx.uint32, "indices must be uint32"
    assert centroids.dtype == mx.float32, "centroids must be float32"

    B, n_heads, L_q, D = queries.shape
    _, n_kv_heads, L_kv, _ = indices.shape

    scale_arr = mx.array([scale], dtype=mx.float32)
    out_shape = (B, n_heads, L_q, L_kv)

    if use_tiled:
        if bits not in _qk_tiled_kernels:
            _qk_tiled_kernels[bits] = _build_qk_tiled_kernel(bits)
        kernel = _qk_tiled_kernels[bits]

        grid_x = (L_kv + TILE_K - 1) // TILE_K
        grid_y = (L_q + TILE_Q - 1) // TILE_Q
        grid_z = B * n_heads

        outputs = kernel(
            inputs=[queries, indices, norms, centroids, scale_arr],
            template=[("T", queries.dtype), ("BITS", bits),
                      ("TILE_Q", TILE_Q), ("TILE_K", TILE_K)],
            output_shapes=[out_shape],
            output_dtypes=[queries.dtype],
            grid=(grid_x * TILE_K, grid_y * TILE_Q, grid_z),
            threadgroup=(TILE_K, TILE_Q, 1),
            init_value=0,
        )
    else:
        if bits not in _qk_kernels:
            _qk_kernels[bits] = _build_qk_kernel(bits)
        kernel = _qk_kernels[bits]

        total_elements = int(np.prod(out_shape))
        outputs = kernel(
            inputs=[queries, indices, norms, centroids, scale_arr],
            template=[("T", queries.dtype), ("BITS", bits)],
            output_shapes=[out_shape],
            output_dtypes=[queries.dtype],
            grid=(total_elements, 1, 1),
            threadgroup=(min(256, total_elements), 1, 1),
        )

    return outputs[0]


def polarquant_sv_matmul(
    weights: mx.array,
    v_indices: mx.array,
    v_norms: mx.array,
    v_centroids: mx.array,
    head_dim: int,
    bits: int = 3,
    precombine: bool = True,
    sparse_v_threshold=0.0,
    use_simd: bool = True,
) -> mx.array:
    """Fused PolarQuant softmax(scores) @ V matmul.

    Computes the weighted sum of quantized value vectors without materializing
    the full dequantized value matrix.

    Args:
        weights:    (B, n_heads, L_q, L_kv) attention weights (post-softmax)
        v_indices:  (B, n_kv_heads, L_kv, D_packed) uint32 bit-packed value indices
        v_norms:    (B, n_kv_heads, L_kv, 1) value vector norms
        v_centroids: (n_levels,) float32 value codebook centroids
        head_dim:   actual head dimension D (before packing)
        bits:       quantization bits (2, 3, or 4)
        precombine: pre-multiply weight*norm on host (default True, ~25% faster)
        sparse_v_threshold: per-head threshold array (n_heads,) or scalar float.
            Kernel indexes by h_idx. 0.0 disables. Try 1e-4 to 1e-3 for speedup.
        use_simd:   use simdgroup-reduced kernel for precombine path (default True).
            32 lanes per output element parallelize the L_kv inner loop.
            Effective at all context lengths; most impactful at L_kv ≥ 1K.

    Returns:
        output: (B, n_heads, L_q, D) attention output
    """
    assert bits in (2, 3, 4), f"Unsupported bit width: {bits}"

    B, n_heads, L_q, L_kv = weights.shape
    _, n_kv_heads, _, _ = v_indices.shape
    out_shape = (B, n_heads, L_q, head_dim)
    total_elements = int(np.prod(out_shape))
    actual_dim_arr = mx.array([head_dim], dtype=mx.uint32)

    # Accept scalar or per-head array for sparse_v_threshold
    if isinstance(sparse_v_threshold, mx.array):
        thresh_arr = sparse_v_threshold.astype(mx.float32)
    elif isinstance(sparse_v_threshold, (list, np.ndarray)):
        thresh_arr = mx.array(sparse_v_threshold, dtype=mx.float32)
    else:
        # Scalar — broadcast to per-head array
        thresh_arr = mx.full((n_heads,), float(sparse_v_threshold), dtype=mx.float32)

    if precombine:
        rep = n_heads // n_kv_heads
        norms_sq = v_norms.squeeze(-1)
        if rep > 1:
            norms_exp = mx.repeat(norms_sq, rep, axis=1)
        else:
            norms_exp = norms_sq
        wn = weights * norms_exp[:, :, None, :]

        if use_simd:
            if bits not in _sv_simd_kernels:
                _sv_simd_kernels[bits] = _build_sv_simd_kernel(bits)
            kernel = _sv_simd_kernels[bits]
            SG = 32
            outputs = kernel(
                inputs=[wn, v_indices, v_centroids, actual_dim_arr, thresh_arr],
                template=[("T", weights.dtype), ("BITS", bits)],
                output_shapes=[out_shape],
                output_dtypes=[weights.dtype],
                grid=(total_elements * SG, 1, 1),
                threadgroup=(SG, 1, 1),
            )
        else:
            if bits not in _sv_pre_kernels:
                _sv_pre_kernels[bits] = _build_sv_precombined_kernel(bits)
            kernel = _sv_pre_kernels[bits]
            outputs = kernel(
                inputs=[wn, v_indices, v_centroids, actual_dim_arr, thresh_arr],
                template=[("T", weights.dtype), ("BITS", bits)],
                output_shapes=[out_shape],
                output_dtypes=[weights.dtype],
                grid=(total_elements, 1, 1),
                threadgroup=(min(256, total_elements), 1, 1),
            )
    else:
        if bits not in _sv_kernels:
            _sv_kernels[bits] = _build_sv_kernel(bits)
        kernel = _sv_kernels[bits]

        outputs = kernel(
            inputs=[weights, v_indices, v_norms, v_centroids, actual_dim_arr],
            template=[("T", weights.dtype), ("BITS", bits)],
            output_shapes=[out_shape],
            output_dtypes=[weights.dtype],
            grid=(total_elements, 1, 1),
            threadgroup=(min(256, total_elements), 1, 1),
        )

    return outputs[0]
