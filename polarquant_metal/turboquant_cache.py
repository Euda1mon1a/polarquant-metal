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


def _create_causal_mask(N, offset=0, window_size=None):
    """Create causal attention mask compatible with mlx-lm."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
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

    def __init__(self, bits: int = 3, fused: bool = True, min_fused_context: int = 512):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        self.turbo_bits = bits
        self.offset = 0
        self._fused = fused
        self.min_fused_context = min_fused_context

        self._head_dim = None
        self._key_pq = None
        self._value_pq = None
        self._quantized = not fused  # fused=False always quantizes; fused=True defers

        # FP16 storage (used below min_fused_context)
        self._fp16_keys = None
        self._fp16_values = None

        # Packed storage (used at/above min_fused_context)
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
        """Store new KV entries. Uses FP16 until context reaches threshold,
        then bulk-quantizes and switches to PolarQuant compressed storage.

        Below min_fused_context: zero overhead (stores/returns FP16 like KVCache).
        At/above threshold: compressed storage + fused Metal kernel decode.

        Args:
            keys: (B, n_kv_heads, S, D) new key vectors
            values: (B, n_kv_heads, S, D) new value vectors

        Returns:
            (all_keys, all_values): FP16 or packed arrays depending on mode
        """
        B, n_kv_heads, S, D = keys.shape

        if self._key_pq is None:
            self._init(D)

        if not self._quantized and self._fused:
            # Lazy mode (fused=True only): store FP16 until threshold
            if self._fp16_keys is None:
                self._fp16_keys = keys
                self._fp16_values = values
            else:
                self._fp16_keys = mx.concatenate(
                    [self._fp16_keys, keys], axis=2)
                self._fp16_values = mx.concatenate(
                    [self._fp16_values, values], axis=2)
            self.offset += S

            if self.offset >= self.min_fused_context:
                self._bulk_quantize()

            return self._fp16_keys, self._fp16_values

        # Quantized mode: compress and store
        prev = self.offset
        k_idx, k_norms = self._key_pq.quantize(keys)
        v_idx, v_norms = self._value_pq.quantize(values)
        k_packed = pack_indices(k_idx, self.turbo_bits)
        v_packed = pack_indices(v_idx, self.turbo_bits)

        needed = prev + S
        if self._k_packed is None or needed > self._k_packed.shape[2]:
            self._expand(B, n_kv_heads, S, keys.dtype, k_packed.shape[-1])

        self._k_packed[..., prev:prev + S, :] = k_packed
        self._k_norms[..., prev:prev + S, :] = k_norms
        self._v_packed[..., prev:prev + S, :] = v_packed
        self._v_norms[..., prev:prev + S, :] = v_norms
        self.offset += S

        if self._fused:
            # Return packed stubs — SDPA patch uses fused_sdpa
            return (
                self._k_packed[..., :self.offset, :],
                self._v_packed[..., :self.offset, :],
            )

        # Non-fused: return dequantized arrays for standard SDPA
        all_k = self._key_pq.dequantize(
            self._unpack_keys(), self._k_norms[..., :self.offset, :]
        )
        all_v = self._value_pq.dequantize(
            self._unpack_values(), self._v_norms[..., :self.offset, :]
        )
        return all_k, all_v

    def _bulk_quantize(self):
        """Convert accumulated FP16 cache to PolarQuant compressed format."""
        B, n_kv_heads, L, D = self._fp16_keys.shape

        k_idx, k_norms = self._key_pq.quantize(self._fp16_keys)
        v_idx, v_norms = self._value_pq.quantize(self._fp16_values)
        k_packed = pack_indices(k_idx, self.turbo_bits)
        v_packed = pack_indices(v_idx, self.turbo_bits)

        # Allocate packed storage
        self._expand(B, n_kv_heads, L, self._fp16_keys.dtype,
                     k_packed.shape[-1])
        self._k_packed[..., :L, :] = k_packed
        self._k_norms[..., :L, :] = k_norms
        self._v_packed[..., :L, :] = v_packed
        self._v_norms[..., :L, :] = v_norms

        # Free FP16 storage
        self._fp16_keys = None
        self._fp16_values = None
        self._quantized = True

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
        if self.offset == 0:
            raise RuntimeError("Cache is empty — call update_and_fetch first")
        if not self._quantized:
            self._bulk_quantize()

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
        if self.offset == 0:
            return None
        if not self._quantized:
            return self._fp16_keys
        return self._key_pq.dequantize(
            self._unpack_keys(), self._k_norms[..., :self.offset, :]
        )

    @property
    def values(self):
        if self.offset == 0:
            return None
        if not self._quantized:
            return self._fp16_values
        return self._value_pq.dequantize(
            self._unpack_values(), self._v_norms[..., :self.offset, :]
        )

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        if self.offset == 0:
            return []
        if not self._quantized:
            return [self._fp16_keys, self._fp16_values]
        return [
            self._k_packed[..., :self.offset, :],
            self._k_norms[..., :self.offset, :],
            self._v_packed[..., :self.offset, :],
            self._v_norms[..., :self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if v is not None and v:
            if len(v) == 2:
                # FP16 mode
                self._fp16_keys, self._fp16_values = v
                self.offset = self._fp16_keys.shape[2]
                self._quantized = False
            else:
                self._k_packed, self._k_norms, self._v_packed, self._v_norms = v
                self.offset = self._k_packed.shape[2]
                self._quantized = True

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.turbo_bits, self._head_dim or 0)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.turbo_bits = int(v[0]), int(v[1])
        head_dim = int(v[2])
        if head_dim > 0:
            self._init(head_dim)

    def make_mask(self, N, return_array=False, window_size=None):
        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            return _create_causal_mask(N, offset=self.offset, window_size=window_size)
        return "causal"

    @property
    def nbytes(self):
        if self._k_packed is None:
            return 0
        return sum(
            a[..., :self.offset, :].nbytes
            for a in (self._k_packed, self._k_norms, self._v_packed, self._v_norms)
        )
