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


# Cache compiled kernels by bit width
_qk_kernels = {}
_qk_tiled_kernels = {}
_sv_kernels = {}
_sv_pre_kernels = {}

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
