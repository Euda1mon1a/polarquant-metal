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
from .kernels import (
    polarquant_qk_matmul, polarquant_sv_matmul,
    polarquant_sv_build_index, polarquant_sv_sparse,
)


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

    def __init__(self, bits: int = 3, bits_v: int = None, fused: bool = True,
                 min_fused_context: int = 512, sparse_v_threshold: float = 1e-3,
                 system_prompt_len: int = 0, recent_zone_len: int = 0):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        self.turbo_bits = bits  # K bits (also used as default for V)
        self._bits_v = bits_v if bits_v is not None else bits
        self.offset = 0
        self._fused = fused
        self.min_fused_context = min_fused_context
        self.sparse_v_threshold = sparse_v_threshold

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

        # Rigidity gate state (Phase 2b): previous token's packed data for reuse
        self._prev_k_packed = None
        self._prev_v_packed = None
        self._prev_k_unit_rotated = None  # for cheap cosine comparison
        self._prev_v_unit_rotated = None
        self.rigidity_threshold = 0.90  # cosine sim threshold for index reuse
        self._rigidity_skips = 0
        self._rigidity_total = 0

        # Entropy amortization (Exp 6): recompute every N steps, cache between
        self.entropy_recompute_interval = 50
        self._entropy_step_counter = 0
        self._cached_thresholds = None

        # Zone priors (Phase 3): positions with elevated prior always in active index
        self.system_prompt_len = system_prompt_len
        self.recent_zone_len = recent_zone_len

    def _init(self, head_dim):
        """Lazy initialization once we know head_dim."""
        self._head_dim = head_dim
        self._key_pq = PolarQuant(bits=self.turbo_bits, dim=head_dim, seed=42)
        self._value_pq = PolarQuant(bits=self._bits_v, dim=head_dim, seed=43)

        if self._fused:
            self._key_centroids_f32 = load_codebook_f32(self.turbo_bits, head_dim)
            self._value_centroids_f32 = load_codebook_f32(self._bits_v, head_dim)

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
                # After bulk quantize, fp16 storage is freed — return packed
                # stubs so patched SDPA routes to fused_sdpa()
                return (
                    self._k_packed[..., :self.offset, :],
                    self._v_packed[..., :self.offset, :],
                )

            return self._fp16_keys, self._fp16_values

        # Quantized mode: compress and store
        # Phase 2b: Rigidity gate — skip quantize+pack if new token's rotated
        # unit vector is nearly identical to previous token's (cosine sim > threshold).
        # Only applies to single-token decode (S=1) after we have a previous token.
        prev = self.offset

        if (S == 1 and self._prev_k_packed is not None
                and self.rigidity_threshold > 0):
            # Compute rotated unit vectors (cheap: normalize + matmul)
            k_norms_new = mx.linalg.norm(keys, axis=-1, keepdims=True)
            v_norms_new = mx.linalg.norm(values, axis=-1, keepdims=True)
            k_unit = keys / mx.maximum(k_norms_new, 1e-8)
            v_unit = values / mx.maximum(v_norms_new, 1e-8)
            k_rotated = k_unit @ self._key_pq.rotation_t
            v_rotated = v_unit @ self._value_pq.rotation_t

            # Cosine similarity with previous token (dot product of unit vectors)
            k_sim = (k_rotated * self._prev_k_unit_rotated).sum(axis=-1).mean()
            v_sim = (v_rotated * self._prev_v_unit_rotated).sum(axis=-1).mean()
            mx.eval(k_sim, v_sim)

            self._rigidity_total += 1

            if float(k_sim.item()) > self.rigidity_threshold and float(v_sim.item()) > self.rigidity_threshold:
                # Reuse previous packed indices, only update norms
                k_packed = self._prev_k_packed
                v_packed = self._prev_v_packed
                k_norms = k_norms_new
                v_norms = v_norms_new
                self._prev_k_unit_rotated = k_rotated
                self._prev_v_unit_rotated = v_rotated
                self._rigidity_skips += 1
            else:
                # Full quantization path
                k_idx, k_norms = self._key_pq.quantize(keys)
                v_idx, v_norms = self._value_pq.quantize(values)
                k_packed = pack_indices(k_idx, self.turbo_bits)
                v_packed = pack_indices(v_idx, self._bits_v)
                # Update rigidity state
                self._prev_k_packed = k_packed
                self._prev_v_packed = v_packed
                self._prev_k_unit_rotated = k_rotated
                self._prev_v_unit_rotated = v_rotated
        else:
            k_idx, k_norms = self._key_pq.quantize(keys)
            v_idx, v_norms = self._value_pq.quantize(values)
            k_packed = pack_indices(k_idx, self.turbo_bits)
            v_packed = pack_indices(v_idx, self._bits_v)
            # Initialize rigidity state for next token
            if S == 1 and self.rigidity_threshold > 0:
                k_norms_init = mx.linalg.norm(keys, axis=-1, keepdims=True)
                v_norms_init = mx.linalg.norm(values, axis=-1, keepdims=True)
                k_unit = keys / mx.maximum(k_norms_init, 1e-8)
                v_unit = values / mx.maximum(v_norms_init, 1e-8)
                self._prev_k_packed = k_packed
                self._prev_v_packed = v_packed
                self._prev_k_unit_rotated = k_unit @ self._key_pq.rotation_t
                self._prev_v_unit_rotated = v_unit @ self._value_pq.rotation_t

        # Pre-allocate or grow in step-aligned blocks; write via scatter update.
        # Avoids O(L_kv) concatenate every decode step (was quadratic at long context).
        if (self._k_packed is None
                or self.offset + S > self._k_packed.shape[2]):
            self._expand(B, n_kv_heads, S, k_norms.dtype,
                         k_packed.shape[-1], v_packed.shape[-1])
        self._k_packed = self._k_packed.at[..., self.offset:self.offset + S, :].add(k_packed)
        self._k_norms  = self._k_norms.at[...,  self.offset:self.offset + S, :].add(k_norms)
        self._v_packed = self._v_packed.at[..., self.offset:self.offset + S, :].add(v_packed)
        self._v_norms  = self._v_norms.at[...,  self.offset:self.offset + S, :].add(v_norms)
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
        v_packed = pack_indices(v_idx, self._bits_v)

        # Allocate packed storage (K and V may have different packed dims)
        dtype = self._fp16_keys.dtype
        shape_n = (B, n_kv_heads, L, 1)
        self._k_packed = k_packed
        self._k_norms = k_norms
        self._v_packed = v_packed
        self._v_norms = v_norms

        # Free FP16 storage
        self._fp16_keys = None
        self._fp16_values = None
        self._quantized = True

    def _compute_adaptive_threshold(self, weights):
        """Compute per-head entropy-guided sparse V thresholds.

        Low entropy (concentrated attention) -> use configured threshold.
        High entropy (spread attention) -> disable pruning.
        Returns per-head threshold array for Phase 2a Metal kernel.

        Args:
            weights: (B, n_heads, L_q, L_kv) post-softmax attention weights

        Returns:
            mx.array: (n_heads,) per-head thresholds in [0, sparse_v_threshold]
        """
        n_heads = weights.shape[1]
        eps = 1e-10
        log_w = mx.log(weights + eps)
        # Per-head entropy: (B, n_heads, L_q) -> mean over B and L_q -> (n_heads,)
        head_entropy = -(weights * log_w).sum(axis=-1).mean(axis=(0, 2))  # (n_heads,)
        max_ent = math.log(weights.shape[-1])
        mx.eval(head_entropy)

        # Sigmoid mapping per head: low entropy -> high threshold, high entropy -> ~0
        thresholds = []
        for h in range(n_heads):
            norm_ent = float(head_entropy[h].item()) / max_ent
            t = self.sparse_v_threshold / (1.0 + math.exp(10.0 * (norm_ent - 0.5)))
            thresholds.append(t)

        return mx.array(thresholds, dtype=mx.float32)

    def _get_zone_mask(self, L_kv):
        """Build zone prior mask for Phase 3 sparse SV.

        Positions with prior=1 are always included in the active index,
        regardless of attention threshold. This implements Bayesian priors
        on the attention probability field — system prompt and recent tokens
        have elevated prior probability of importance.

        Args:
            L_kv: current context length

        Returns:
            mx.array: (L_kv,) uint32, 1 = always active, 0 = threshold-gated
        """
        mask = mx.zeros(L_kv, dtype=mx.uint32)
        if self.system_prompt_len > 0 and self.system_prompt_len < L_kv:
            mask = mask.at[:self.system_prompt_len].add(mx.ones(self.system_prompt_len, dtype=mx.uint32))
        if self.recent_zone_len > 0:
            start = max(0, L_kv - self.recent_zone_len)
            rlen = L_kv - start
            mask = mask.at[start:].add(mx.ones(rlen, dtype=mx.uint32))
        return mask

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

        # Entropy-guided adaptive sparse V threshold (Phase 1 + Exp 6 amortization)
        # Compute per-head entropy every N steps, cache between. Exp 6 showed
        # thresholds are regime-robust: zero quality loss at interval=50.
        L_kv = weights.shape[-1]
        if self.sparse_v_threshold > 0 and L_kv > 1024:
            self._entropy_step_counter += 1
            if (self._cached_thresholds is None
                    or self._entropy_step_counter >= self.entropy_recompute_interval):
                self._cached_thresholds = self._compute_adaptive_threshold(weights)
                self._entropy_step_counter = 0
            adaptive_threshold = self._cached_thresholds
        else:
            adaptive_threshold = self.sparse_v_threshold

        # Phase 3: Compact-index sparse SV when thresholds active and context long.
        # Falls back to Phase 2 dense kernel for short contexts or zero thresholds.
        n_heads = weights.shape[1]
        use_sparse = (
            isinstance(adaptive_threshold, mx.array)
            and L_kv > 2048
            and L_q == 1  # Phase 3 is decode-only
            and float(adaptive_threshold.max().item()) > 0
        )

        if use_sparse:
            # Precombine weight*norm for sparse kernel (same as dense precombine)
            rep = n_heads // v_packed.shape[1]
            norms_sq = v_norms.squeeze(-1)
            norms_exp = mx.repeat(norms_sq, rep, axis=1) if rep > 1 else norms_sq
            wn = weights * norms_exp[:, :, None, :]

            # Build zone prior mask and compact active index
            zone_mask = self._get_zone_mask(L_kv)
            count_and_indices = polarquant_sv_build_index(wn, adaptive_threshold, zone_mask)

            # Sparse SV: iterate only active positions
            out_rotated = polarquant_sv_sparse(
                count_and_indices, wn, v_packed,
                self._value_centroids_f32,
                self._head_dim, L_kv, self._bits_v,
            )
        else:
            # Dense fallback (Phase 2 kernel)
            out_rotated = polarquant_sv_matmul(
                weights=weights,
                v_indices=v_packed,
                v_norms=v_norms,
                v_centroids=self._value_centroids_f32,
                head_dim=self._head_dim,
                bits=self._bits_v,
                sparse_v_threshold=adaptive_threshold,
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
            self._bits_v, self._head_dim,
        )

    def _expand(self, B, n_kv_heads, new_tokens, dtype, packed_dim_k, packed_dim_v=None):
        """Allocate or expand compressed storage in step-aligned blocks.

        Grows the buffer by ceil(new_tokens / step) * step slots. Called when the
        current capacity (shape[2]) is insufficient. Between expansions, decode steps
        write via scatter update (at[].add()) which is O(S) not O(L_kv).
        """
        if packed_dim_v is None:
            packed_dim_v = packed_dim_k
        alloc = ((self.step + new_tokens - 1) // self.step) * self.step
        shape_k = (B, n_kv_heads, alloc, packed_dim_k)
        shape_v = (B, n_kv_heads, alloc, packed_dim_v)
        shape_n = (B, n_kv_heads, alloc, 1)

        if self._k_packed is not None and self.offset > 0:
            self._k_packed = mx.concatenate([
                self._k_packed[..., :self.offset, :],
                mx.zeros(shape_k, dtype=mx.uint32),
            ], axis=2)
            self._k_norms = mx.concatenate([
                self._k_norms[..., :self.offset, :],
                mx.zeros(shape_n, dtype=dtype),
            ], axis=2)
            self._v_packed = mx.concatenate([
                self._v_packed[..., :self.offset, :],
                mx.zeros(shape_v, dtype=mx.uint32),
            ], axis=2)
            self._v_norms = mx.concatenate([
                self._v_norms[..., :self.offset, :],
                mx.zeros(shape_n, dtype=dtype),
            ], axis=2)
        else:
            self._k_packed = mx.zeros(shape_k, dtype=mx.uint32)
            self._k_norms = mx.zeros(shape_n, dtype=dtype)
            self._v_packed = mx.zeros(shape_v, dtype=mx.uint32)
            self._v_norms = mx.zeros(shape_n, dtype=dtype)

    @property
    def rigidity_stats(self):
        """Return rigidity gate statistics."""
        if self._rigidity_total == 0:
            return {"skips": 0, "total": 0, "skip_rate": 0.0}
        return {
            "skips": self._rigidity_skips,
            "total": self._rigidity_total,
            "skip_rate": self._rigidity_skips / self._rigidity_total,
        }

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
