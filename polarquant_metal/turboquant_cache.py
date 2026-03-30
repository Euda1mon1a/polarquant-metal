"""
TurboQuantKVCache with fused Metal kernel support.

Drop-in compatible with rachittshah's PR #1059 TurboQuantKVCache interface,
but adds a `fused_sdpa()` method that uses custom Metal kernels to compute
attention directly from packed quantized data — no dequantize-on-fetch.

Usage:
    from polarquant_metal.turboquant_cache import TurboQuantKVCache

    # As a drop-in for mlx-lm (dequantize path, compatible):
    cache = TurboQuantKVCache(bits=3)
    keys, values = cache.update_and_fetch(k, v)
    output = scaled_dot_product_attention(q, keys, values, scale, mask)

    # With fused kernels (fast path):
    cache = TurboQuantKVCache(bits=3, fused=True)
    cache.update_and_fetch(k, v)  # stores compressed, returns dequantized for compat
    output = cache.fused_sdpa(queries, scale, mask)  # fused Metal kernels
"""

import math

import mlx.core as mx

from .polar_quant import PolarQuant
from .packing import pack_indices
from .codebooks import load_codebook_f32
from .kernels import polarquant_qk_matmul, polarquant_sv_matmul


def _create_attention_mask(h, offset, dtype=mx.float32):
    """Create causal attention mask compatible with mlx-lm."""
    rinds = mx.arange(offset - h.shape[2], offset)
    linds = mx.arange(offset, offset + h.shape[2])
    mask = linds[:, None] >= rinds[None]
    return mask


class TurboQuantKVCache:
    """KV cache compressed with PolarQuant, optionally using fused Metal kernels.

    Compatible with mlx-lm's cache interface (update_and_fetch returns
    dequantized arrays). When `fused=True`, also supports `fused_sdpa()`
    which computes attention directly from packed data.

    Args:
        bits: Bits per coordinate (2, 3, or 4). Default: 3.
        fused: Enable fused Metal kernel support. Default: True.
    """

    step = 256

    def __init__(self, bits: int = 3, fused: bool = True):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        self.turbo_bits = bits
        self.offset = 0
        self._fused = fused

        self._head_dim = None
        self._key_pq = None
        self._value_pq = None

        # Packed storage
        self._k_packed = None
        self._k_norms = None
        self._v_packed = None
        self._v_norms = None

        # Fused kernel precomputed data
        self._key_centroids_f32 = None
        self._value_centroids_f32 = None

    def _init(self, head_dim):
        """Lazy initialization once we know head_dim."""
        self._head_dim = head_dim
        self._key_pq = PolarQuant(bits=self.turbo_bits, dim=head_dim, seed=42)
        self._value_pq = PolarQuant(bits=self.turbo_bits, dim=head_dim, seed=43)

        if self._fused:
            self._key_centroids_f32 = load_codebook_f32(self.turbo_bits, head_dim)
            self._value_centroids_f32 = load_codebook_f32(self.turbo_bits, head_dim)

    def update_and_fetch(self, keys, values):
        """Compress and store new KV entries.

        Returns dequantized (keys, values) for compatibility with mlx-lm's
        standard attention path. The compressed data is stored internally
        for use by fused_sdpa().

        Args:
            keys: (B, n_kv_heads, S, D) new key vectors
            values: (B, n_kv_heads, S, D) new value vectors

        Returns:
            (all_keys_deq, all_values_deq): dequantized full cache contents
        """
        B, n_kv_heads, S, D = keys.shape
        prev = self.offset

        if self._key_pq is None:
            self._init(D)

        # Quantize
        k_idx, k_norms = self._key_pq.quantize(keys)
        v_idx, v_norms = self._value_pq.quantize(values)

        # Pack
        k_packed = pack_indices(k_idx, self.turbo_bits)
        v_packed = pack_indices(v_idx, self.turbo_bits)

        # Expand storage if needed
        needed = prev + S
        if self._k_packed is None or needed > self._k_packed.shape[2]:
            self._expand(B, n_kv_heads, S, keys.dtype, k_packed.shape[-1])

        # Store
        self._k_packed[..., prev:prev + S, :] = k_packed
        self._k_norms[..., prev:prev + S, :] = k_norms
        self._v_packed[..., prev:prev + S, :] = v_packed
        self._v_norms[..., prev:prev + S, :] = v_norms
        self.offset += S

        # Return dequantized for compatibility with standard attention path.
        # When using fused_sdpa(), callers can ignore these return values.
        all_k = self._key_pq.dequantize(
            self._unpack_keys(), self._k_norms[..., :self.offset, :]
        )
        all_v = self._value_pq.dequantize(
            self._unpack_values(), self._v_norms[..., :self.offset, :]
        )
        return all_k, all_v

    def fused_sdpa(self, queries, scale=None, mask=None):
        """Compute full attention using fused Metal kernels.

        This is the fast path — computes Q@K^T and scores@V directly from
        packed quantized data without materializing dequantized arrays.

        Args:
            queries: (B, n_heads, L_q, D) query vectors (NOT pre-rotated)
            scale: attention scale factor (default: 1/sqrt(D))
            mask: attention mask (None, bool array, or float array)

        Returns:
            output: (B, n_heads, L_q, D) attention output
        """
        if not self._fused:
            raise RuntimeError("fused_sdpa requires fused=True")
        if self._k_packed is None or self.offset == 0:
            raise RuntimeError("Cache is empty — call update_and_fetch first")

        B, n_heads, L_q, D = queries.shape
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        k_packed = self._k_packed[..., :self.offset, :]
        k_norms = self._k_norms[..., :self.offset, :]
        v_packed = self._v_packed[..., :self.offset, :]
        v_norms = self._v_norms[..., :self.offset, :]

        # Pre-rotate queries into key PolarQuant basis
        q_rotated = queries @ self._key_pq.rotation_t

        # Fused Q @ K^T
        scores = polarquant_qk_matmul(
            queries=q_rotated,
            indices=k_packed,
            norms=k_norms,
            centroids=self._key_centroids_f32,
            scale=scale,
            bits=self.turbo_bits,
        )

        # Apply mask
        if mask is not None:
            if hasattr(mask, "dtype"):
                if mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                else:
                    scores = scores + mask

        # Softmax
        weights = mx.softmax(scores, axis=-1, precise=True)

        # Fused weights @ V (output in rotated value basis)
        out_rotated = polarquant_sv_matmul(
            weights=weights,
            v_indices=v_packed,
            v_norms=v_norms,
            v_centroids=self._value_centroids_f32,
            head_dim=self._head_dim,
            bits=self.turbo_bits,
        )

        # Inverse rotation from value basis
        output = out_rotated @ self._value_pq.rotation
        return output.astype(queries.dtype)

    def _unpack_keys(self):
        """Unpack key indices for dequantization."""
        from .packing import unpack_indices
        return unpack_indices(
            self._k_packed[..., :self.offset, :],
            self.turbo_bits, self._head_dim,
        )

    def _unpack_values(self):
        """Unpack value indices for dequantization."""
        from .packing import unpack_indices
        return unpack_indices(
            self._v_packed[..., :self.offset, :],
            self.turbo_bits, self._head_dim,
        )

    def _expand(self, B, n_kv_heads, new_tokens, dtype, packed_dim):
        """Allocate or expand compressed storage."""
        alloc = ((self.step + new_tokens - 1) // self.step) * self.step
        shape_p = (B, n_kv_heads, alloc, packed_dim)
        shape_n = (B, n_kv_heads, alloc, 1)

        if self._k_packed is not None and self.offset > 0:
            old = (
                self._k_packed[..., :self.offset, :],
                self._k_norms[..., :self.offset, :],
                self._v_packed[..., :self.offset, :],
                self._v_norms[..., :self.offset, :],
            )
            new = (
                mx.zeros(shape_p, dtype=mx.uint32),
                mx.zeros(shape_n, dtype=dtype),
                mx.zeros(shape_p, dtype=mx.uint32),
                mx.zeros(shape_n, dtype=dtype),
            )
            self._k_packed, self._k_norms, self._v_packed, self._v_norms = (
                mx.concatenate([o, n], axis=2) for o, n in zip(old, new)
            )
        else:
            self._k_packed = mx.zeros(shape_p, dtype=mx.uint32)
            self._k_norms = mx.zeros(shape_n, dtype=dtype)
            self._v_packed = mx.zeros(shape_p, dtype=mx.uint32)
            self._v_norms = mx.zeros(shape_n, dtype=dtype)

    # --- mlx-lm cache interface ---

    @property
    def keys(self):
        if self._k_packed is None or self.offset == 0:
            return None
        return self._key_pq.dequantize(
            self._unpack_keys(), self._k_norms[..., :self.offset, :]
        )

    @property
    def values(self):
        if self._v_packed is None or self.offset == 0:
            return None
        return self._value_pq.dequantize(
            self._unpack_values(), self._v_norms[..., :self.offset, :]
        )

    def size(self):
        return self.offset

    def empty(self):
        return self._k_packed is None or self.offset == 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        if self._k_packed is None:
            return []
        return [
            self._k_packed[..., :self.offset, :],
            self._k_norms[..., :self.offset, :],
            self._v_packed[..., :self.offset, :],
            self._v_norms[..., :self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._k_packed, self._k_norms, self._v_packed, self._v_norms = v
            self.offset = self._k_packed.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.turbo_bits, self._head_dim or 0)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.turbo_bits = int(v[0]), int(v[1])
        head_dim = int(v[2])
        if head_dim > 0:
            self._init(head_dim)

    def make_mask(self, *args, **kwargs):
        return _create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def nbytes(self):
        if self._k_packed is None:
            return 0
        return sum(
            a[..., :self.offset, :].nbytes
            for a in (self._k_packed, self._k_norms, self._v_packed, self._v_norms)
        )
