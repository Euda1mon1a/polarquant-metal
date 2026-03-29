"""
FusedPolarQuantKVCache: Drop-in KV cache using fused Metal kernels.

This cache stores keys and values in PolarQuant-compressed form (packed uint32
indices + norms) and uses custom Metal kernels to compute attention scores
and output directly from the compressed representation — no dequantize-on-fetch.

The key optimization: queries are pre-rotated into the PolarQuant basis so the
fused kernel can do codebook lookup + dot product in a single pass per element.
"""

import math

import mlx.core as mx
import numpy as np

from .polar_quant import PolarQuant
from .packing import pack_indices, unpack_indices
from .codebooks import load_codebook_f32
from .kernels import polarquant_qk_matmul, polarquant_sv_matmul


class FusedPolarQuantKVCache:
    """PolarQuant KV cache with fused Metal dequant-matmul kernels.

    Stores compressed KV cache and computes attention scores + output
    without materializing the full dequantized arrays.

    Args:
        bits: Bits per coordinate (2, 3, or 4).
        head_dim: Dimension of each attention head.
        key_seed: Random seed for key rotation matrix.
        value_seed: Random seed for value rotation matrix.
    """

    step = 256  # allocation granularity

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        key_seed: int = 42,
        value_seed: int = 43,
    ):
        self.bits = bits
        self.head_dim = head_dim
        self.offset = 0

        self.key_pq = PolarQuant(bits=bits, dim=head_dim, seed=key_seed)
        self.value_pq = PolarQuant(bits=bits, dim=head_dim, seed=value_seed)

        # Precomputed for the Metal kernel
        self.key_centroids_f32 = load_codebook_f32(bits, head_dim)
        self.value_centroids_f32 = load_codebook_f32(bits, head_dim)

        # Rotation matrix for pre-rotating queries (key basis)
        self.key_rotation_t = self.key_pq.rotation_t

        # Packed storage
        self._key_packed = None      # (B, n_kv_heads, capacity, D_packed) uint32
        self._key_norms = None       # (B, n_kv_heads, capacity, 1)
        self._value_packed = None    # (B, n_kv_heads, capacity, D_packed) uint32
        self._value_norms = None     # (B, n_kv_heads, capacity, 1)
        self._capacity = 0

        vals_per_int = 32 // bits
        self._d_packed = (head_dim + vals_per_int - 1) // vals_per_int

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Compress and store new KV entries. Returns (key_state, value_state).

        Unlike the naive approach that returns dequantized arrays, this returns
        a tuple of (packed_indices, norms, rotation_matrix) that the fused
        attention function will use directly.

        Args:
            keys: (B, n_kv_heads, S, D) new key vectors
            values: (B, n_kv_heads, S, D) new value vectors

        Returns:
            key_state: (packed_indices, norms, rotation_t, centroids, scale)
            value_state: (packed_indices, norms, centroids, head_dim)
        """
        B, n_kv_heads, S, D = keys.shape
        prev = self.offset

        # Quantize
        k_indices, k_norms = self.key_pq.quantize(keys)
        v_indices, v_norms = self.value_pq.quantize(values)

        # Pack indices into uint32
        k_packed = pack_indices(k_indices, self.bits)
        v_packed = pack_indices(v_indices, self.bits)

        # Allocate or expand storage
        needed = prev + S
        if self._key_packed is None or needed > self._capacity:
            self._expand(B, n_kv_heads, S, keys.dtype)

        # Store
        self._key_packed[..., prev:prev + S, :] = k_packed
        self._key_norms[..., prev:prev + S, :] = k_norms
        self._value_packed[..., prev:prev + S, :] = v_packed
        self._value_norms[..., prev:prev + S, :] = v_norms

        self.offset += S

        # Return sliced views for attention computation
        return self.key_state, self.value_state

    @property
    def key_state(self):
        """Returns (packed_indices, norms) for current cache content."""
        return (
            self._key_packed[..., :self.offset, :],
            self._key_norms[..., :self.offset, :],
        )

    @property
    def value_state(self):
        """Returns (packed_indices, norms) for current cache content."""
        return (
            self._value_packed[..., :self.offset, :],
            self._value_norms[..., :self.offset, :],
        )

    def fused_attention(
        self,
        queries: mx.array,
        mask=None,
        n_heads: int = None,
    ) -> mx.array:
        """Compute full attention using fused Metal kernels.

        This is the main entry point — call this instead of doing
        separate Q@K^T, softmax, scores@V.

        Args:
            queries: (B, n_heads, L_q, D) query vectors (NOT pre-rotated)
            mask: optional attention mask
            n_heads: number of query heads (for GQA, inferred from queries if None)

        Returns:
            output: (B, n_heads, L_q, D) attention output
        """
        B, n_q_heads, L_q, D = queries.shape
        scale = 1.0 / math.sqrt(D)

        k_packed, k_norms = self.key_state
        v_packed, v_norms = self.value_state

        # Pre-rotate queries into key PolarQuant basis
        q_rotated = queries @ self.key_rotation_t

        # Fused Q @ K^T (scores computation)
        scores = polarquant_qk_matmul(
            queries=q_rotated,
            indices=k_packed,
            norms=k_norms,
            centroids=self.key_centroids_f32,
            scale=scale,
            bits=self.bits,
        )

        # Apply mask
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                L_kv = self.offset
                q_indices = mx.arange(L_kv - L_q, L_kv)
                k_indices = mx.arange(L_kv)
                mask = q_indices[:, None] >= k_indices[None]
            if hasattr(mask, "dtype"):
                if mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                else:
                    scores += mask

        # Softmax
        weights = mx.softmax(scores, axis=-1, precise=True)

        # Fused weights @ V
        # For values, we need to pre-rotate the output by inverse value rotation.
        # But actually: the value dequant produces R_v^T @ centroids[idx] * norm.
        # The attention output is sum_k w_k * R_v @ centroids[v_idx_k] * v_norm_k.
        # If we do the fused matmul in the rotated basis, we get:
        #   out_rotated[d] = sum_k w_k * centroids[v_idx_k[d]] * v_norm_k
        # Then we inverse-rotate: out = out_rotated @ R_v
        out_rotated = polarquant_sv_matmul(
            weights=weights,
            v_indices=v_packed,
            v_norms=v_norms,
            v_centroids=self.value_centroids_f32,
            head_dim=self.head_dim,
            bits=self.bits,
        )

        # Inverse rotation for values
        output = out_rotated @ self.value_pq.rotation
        return output.astype(queries.dtype)

    def _expand(self, B, n_kv_heads, new_tokens, dtype):
        """Allocate or expand compressed storage."""
        alloc = ((self.step + new_tokens - 1) // self.step) * self.step

        shape_packed = (B, n_kv_heads, alloc, self._d_packed)
        shape_norms = (B, n_kv_heads, alloc, 1)

        if self._key_packed is not None and self.offset > 0:
            old_kp = self._key_packed[..., :self.offset, :]
            old_kn = self._key_norms[..., :self.offset, :]
            old_vp = self._value_packed[..., :self.offset, :]
            old_vn = self._value_norms[..., :self.offset, :]

            new_kp = mx.zeros(shape_packed, dtype=mx.uint32)
            new_kn = mx.zeros(shape_norms, dtype=dtype)
            new_vp = mx.zeros(shape_packed, dtype=mx.uint32)
            new_vn = mx.zeros(shape_norms, dtype=dtype)

            self._key_packed = mx.concatenate([old_kp, new_kp], axis=2)
            self._key_norms = mx.concatenate([old_kn, new_kn], axis=2)
            self._value_packed = mx.concatenate([old_vp, new_vp], axis=2)
            self._value_norms = mx.concatenate([old_vn, new_vn], axis=2)
        else:
            self._key_packed = mx.zeros(shape_packed, dtype=mx.uint32)
            self._key_norms = mx.zeros(shape_norms, dtype=dtype)
            self._value_packed = mx.zeros(shape_packed, dtype=mx.uint32)
            self._value_norms = mx.zeros(shape_norms, dtype=dtype)

        self._capacity = self._key_packed.shape[2]

    # --- mlx-lm cache interface ---

    def size(self):
        return self.offset

    def empty(self):
        return self._key_packed is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        if self._key_packed is None:
            return []
        return [
            self._key_packed[..., :self.offset, :],
            self._key_norms[..., :self.offset, :],
            self._value_packed[..., :self.offset, :],
            self._value_norms[..., :self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if v is not None and v:
            (self._key_packed, self._key_norms,
             self._value_packed, self._value_norms) = v
            self.offset = self._key_packed.shape[2]
            self._capacity = self._key_packed.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.bits, self.head_dim)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self.bits = int(v[1])
        self.head_dim = int(v[2])

    @property
    def nbytes(self):
        if self._key_packed is None:
            return 0
        total = 0
        for arr in [self._key_packed, self._key_norms,
                     self._value_packed, self._value_norms]:
            total += arr[..., :self.offset, :].nbytes
        return total
