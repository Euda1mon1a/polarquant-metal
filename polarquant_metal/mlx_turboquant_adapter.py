"""
Adapter to use the fused Metal kernel with rachittshah/mlx-turboquant.

This module provides a drop-in replacement for mlx-turboquant's
TurboQuantKVCache that uses fused Metal kernels instead of
dequantize-on-fetch.

Usage:
    from polarquant_metal.mlx_turboquant_adapter import FusedTurboQuantKVCache

    # Drop-in replacement:
    cache = [FusedTurboQuantKVCache(bits=3, head_dim=128) for _ in range(num_layers)]
"""

import math
import sys

import mlx.core as mx
import numpy as np

from .polar_quant import PolarQuant
from .packing import pack_indices
from .codebooks import load_codebook_f32
from .kernels import polarquant_qk_matmul, polarquant_sv_matmul


class FusedTurboQuantKVCache:
    """Drop-in replacement for mlx-turboquant's TurboQuantKVCache.

    Uses fused Metal kernels for Q@K^T and weights@V instead of
    dequantize-on-fetch. Same interface as the original.

    The key difference from FusedPolarQuantKVCache: this one implements
    update_and_fetch() to return dequantized arrays for compatibility
    with mlx-lm's standard SDPA path, BUT provides an alternative
    fused_attention() method that bypasses dequantization entirely.

    For maximum speed, use the integration.patch_sdpa() approach which
    routes attention through the fused path automatically.
    """

    step = 256

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        key_seed: int = 42,
        value_seed: int = 43,
    ):
        self.turbo_bits = bits
        self.head_dim = head_dim
        self.offset = 0

        self.key_pq = PolarQuant(bits=bits, dim=head_dim, seed=key_seed)
        self.value_pq = PolarQuant(bits=bits, dim=head_dim, seed=value_seed)

        self.key_centroids_f32 = load_codebook_f32(bits, head_dim)
        self.value_centroids_f32 = load_codebook_f32(bits, head_dim)
        self.key_rotation_t = self.key_pq.rotation_t

        vals_per_int = 32 // bits
        self._d_packed = (head_dim + vals_per_int - 1) // vals_per_int

        # Packed compressed storage
        self._key_packed = None
        self._key_norms = None
        self._value_packed = None
        self._value_norms = None
        self._capacity = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Compress, store, and return PACKED state for fused attention.

        Returns packed state tuples instead of dequantized arrays.
        The integration layer's patched SDPA knows how to handle these.

        Args:
            keys: (B, n_kv_heads, S, D) new key vectors
            values: (B, n_kv_heads, S, D) new value vectors

        Returns:
            key_state: (packed_indices, norms) tuple
            value_state: (packed_indices, norms) tuple
        """
        B, n_kv_heads, S, D = keys.shape
        prev = self.offset

        k_indices, k_norms = self.key_pq.quantize(keys)
        v_indices, v_norms = self.value_pq.quantize(values)

        k_packed = pack_indices(k_indices, self.turbo_bits)
        v_packed = pack_indices(v_indices, self.turbo_bits)

        needed = prev + S
        if self._key_packed is None or needed > self._capacity:
            self._expand(B, n_kv_heads, S, keys.dtype)

        self._key_packed[..., prev:prev + S, :] = k_packed
        self._key_norms[..., prev:prev + S, :] = k_norms
        self._value_packed[..., prev:prev + S, :] = v_packed
        self._value_norms[..., prev:prev + S, :] = v_norms

        self.offset += S

        return self.key_state, self.value_state

    @property
    def key_state(self):
        return (
            self._key_packed[..., :self.offset, :],
            self._key_norms[..., :self.offset, :],
        )

    @property
    def value_state(self):
        return (
            self._value_packed[..., :self.offset, :],
            self._value_norms[..., :self.offset, :],
        )

    def _expand(self, B, n_kv_heads, new_tokens, dtype):
        alloc = ((self.step + new_tokens - 1) // self.step) * self.step
        shape_p = (B, n_kv_heads, alloc, self._d_packed)
        shape_n = (B, n_kv_heads, alloc, 1)

        if self._key_packed is not None and self.offset > 0:
            old_kp = self._key_packed[..., :self.offset, :]
            old_kn = self._key_norms[..., :self.offset, :]
            old_vp = self._value_packed[..., :self.offset, :]
            old_vn = self._value_norms[..., :self.offset, :]

            self._key_packed = mx.concatenate(
                [old_kp, mx.zeros(shape_p, dtype=mx.uint32)], axis=2
            )
            self._key_norms = mx.concatenate(
                [old_kn, mx.zeros(shape_n, dtype=dtype)], axis=2
            )
            self._value_packed = mx.concatenate(
                [old_vp, mx.zeros(shape_p, dtype=mx.uint32)], axis=2
            )
            self._value_norms = mx.concatenate(
                [old_vn, mx.zeros(shape_n, dtype=dtype)], axis=2
            )
        else:
            self._key_packed = mx.zeros(shape_p, dtype=mx.uint32)
            self._key_norms = mx.zeros(shape_n, dtype=dtype)
            self._value_packed = mx.zeros(shape_p, dtype=mx.uint32)
            self._value_norms = mx.zeros(shape_n, dtype=dtype)

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

    def make_mask(self, N, return_array=False, window_size=None):
        from mlx_lm.models.base import create_causal_mask
        offset = self.offset
        if window_size is not None:
            return create_causal_mask(N, offset, window_size=window_size)
        elif N == 1:
            return None
        elif return_array:
            return create_causal_mask(N, offset, window_size=window_size)
        else:
            return "causal"

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
        return tuple(map(str, (self.offset, self.turbo_bits, self.head_dim)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self.turbo_bits = int(v[1])
        self.head_dim = int(v[2])

    @property
    def nbytes(self):
        if self._key_packed is None:
            return 0
        return sum(
            arr[..., :self.offset, :].nbytes
            for arr in [self._key_packed, self._key_norms,
                        self._value_packed, self._value_norms]
        )
