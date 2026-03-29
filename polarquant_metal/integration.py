"""
Integration with mlx-lm: monkey-patches scaled_dot_product_attention
to route through the fused Metal kernels when a FusedPolarQuantKVCache
is detected.
"""

import math
import sys

import mlx.core as mx

from .cache import FusedPolarQuantKVCache
from .kernels import polarquant_qk_matmul, polarquant_sv_matmul


_original_sdpa = None
_patched_modules = []


def _fused_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """Patched SDPA that uses fused Metal kernels for PolarQuant caches.

    When cache is a FusedPolarQuantKVCache, this function:
    1. Pre-rotates queries into the PolarQuant key basis
    2. Calls the fused Q@K^T Metal kernel (codebook lookup + dot product)
    3. Applies mask and softmax
    4. Calls the fused weights@V Metal kernel
    5. Inverse-rotates the output from value basis

    For non-PolarQuant caches, falls through to the original SDPA.

    NOTE: In the patched path, `keys` and `values` are actually
    (packed_state, norm_state) tuples from cache.update_and_fetch(),
    not raw tensors. The cache's update_and_fetch returns the packed
    representation directly.
    """
    if not isinstance(cache, FusedPolarQuantKVCache):
        return _original_sdpa(queries, keys, values, cache, scale=scale,
                              mask=mask, sinks=sinks)

    if sinks is not None:
        raise ValueError("Attention sinks not supported with FusedPolarQuantKVCache.")

    B, n_q_heads, L_q, D = queries.shape

    # keys = (packed_indices, norms) from cache.key_state
    # values = (packed_indices, norms) from cache.value_state
    k_packed, k_norms = keys
    v_packed, v_norms = values

    n_kv_heads = k_packed.shape[1]

    # Pre-rotate queries into PolarQuant key basis
    q_rotated = queries @ cache.key_rotation_t

    # Fused Q @ K^T
    scores = polarquant_qk_matmul(
        queries=q_rotated,
        indices=k_packed,
        norms=k_norms,
        centroids=cache.key_centroids_f32,
        scale=scale,
        bits=cache.bits,
    )

    # Apply mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            L_kv = k_packed.shape[2]
            q_off = L_kv - L_q
            q_indices = mx.arange(q_off, q_off + L_q)
            k_indices = mx.arange(L_kv)
            mask = q_indices[:, None] >= k_indices[None]
        if hasattr(mask, "dtype"):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores += mask

    # Softmax
    weights = mx.softmax(scores, axis=-1, precise=True)

    # Fused weights @ V (in rotated value basis)
    out_rotated = polarquant_sv_matmul(
        weights=weights,
        v_indices=v_packed,
        v_norms=v_norms,
        v_centroids=cache.value_centroids_f32,
        head_dim=cache.head_dim,
        bits=cache.bits,
    )

    # Inverse rotation for values
    output = out_rotated @ cache.value_pq.rotation
    return output.astype(queries.dtype)


def patch_sdpa():
    """Monkey-patch mlx-lm's scaled_dot_product_attention for fused PolarQuant.

    Call this once before running inference with FusedPolarQuantKVCache.
    """
    global _original_sdpa
    import mlx_lm.models.base as base_module

    if _original_sdpa is None:
        _original_sdpa = base_module.scaled_dot_product_attention

    base_module.scaled_dot_product_attention = _fused_sdpa

    # Patch already-imported model modules
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and mod is not None:
            if hasattr(mod, "scaled_dot_product_attention"):
                if mod.scaled_dot_product_attention is not _fused_sdpa:
                    _patched_modules.append(
                        (mod, mod.scaled_dot_product_attention)
                    )
                    mod.scaled_dot_product_attention = _fused_sdpa


def unpatch_sdpa():
    """Restore original SDPA."""
    global _original_sdpa
    if _original_sdpa is not None:
        import mlx_lm.models.base as base_module
        base_module.scaled_dot_product_attention = _original_sdpa
        for mod, orig_fn in _patched_modules:
            mod.scaled_dot_product_attention = orig_fn
        _patched_modules.clear()
        _original_sdpa = None


def make_fused_cache(
    model,
    bits: int = 3,
    head_dim: int = None,
) -> list:
    """Create FusedPolarQuantKVCache instances for each layer and patch SDPA.

    Args:
        model: mlx-lm model
        bits: PolarQuant bits per coordinate (2-4)
        head_dim: Head dimension (auto-detected if None)

    Returns:
        List of FusedPolarQuantKVCache, one per layer
    """
    num_layers = len(model.layers)

    if head_dim is None:
        args = model.args
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None and hasattr(args, "hidden_size") and hasattr(
            args, "num_attention_heads"
        ):
            head_dim = args.hidden_size // args.num_attention_heads
        if head_dim is None:
            raise ValueError("Could not auto-detect head_dim.")

    patch_sdpa()

    return [
        FusedPolarQuantKVCache(bits=bits, head_dim=head_dim)
        for _ in range(num_layers)
    ]
