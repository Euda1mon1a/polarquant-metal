"""
Integration with mlx-lm: patches scaled_dot_product_attention to dispatch
to fused Metal kernels when a TurboQuantKVCache with fused=True is detected.

Usage:
    from polarquant_metal.integration import patch_sdpa, make_fused_cache

    cache = make_fused_cache(model, bits=3)
    patch_sdpa()
    # Now model(input_ids, cache=cache) uses fused Metal kernels automatically
"""

import sys

from .turboquant_cache import TurboQuantKVCache

_original_sdpa = None


def patch_sdpa():
    """Patch mlx-lm's scaled_dot_product_attention to support TurboQuantKVCache.

    Adds a dispatch check: if cache has `turbo_bits` attribute (set by
    TurboQuantKVCache), routes to `cache.fused_sdpa()` which computes
    attention directly from packed quantized data.

    Same pattern as mlx-lm's existing `hasattr(cache, "bits")` check for
    QuantizedKVCache — attribute-based dispatch, no model code changes.
    """
    global _original_sdpa
    import mlx_lm.models.base as base_module

    if _original_sdpa is not None:
        return  # Already patched

    _original_sdpa = base_module.scaled_dot_product_attention

    def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        # TurboQuant fused path conditions:
        # 1. Decode only (L_q == 1) — prefill kernel is too slow for L_q > 1
        # 2. Context >= min_fused_context — overhead doesn't pay off below this
        if hasattr(cache, "turbo_bits") and cache._fused:
            L_q = queries.shape[2]
            if L_q == 1 and cache.offset >= cache.min_fused_context:
                return cache.fused_sdpa(queries, scale=scale, mask=mask)

        # Fall through to original (handles prefill, short context, standard)
        return _original_sdpa(
            queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks,
        )

    base_module.scaled_dot_product_attention = _patched_sdpa

    # Also patch any already-imported model modules that copied the reference
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and mod is not None:
            if hasattr(mod, "scaled_dot_product_attention"):
                if mod.scaled_dot_product_attention is _original_sdpa:
                    mod.scaled_dot_product_attention = _patched_sdpa


def unpatch_sdpa():
    """Restore original SDPA."""
    global _original_sdpa
    if _original_sdpa is None:
        return

    import mlx_lm.models.base as base_module
    base_module.scaled_dot_product_attention = _original_sdpa

    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and mod is not None:
            if hasattr(mod, "scaled_dot_product_attention"):
                # Only restore if it's our patched version
                if mod.scaled_dot_product_attention is not _original_sdpa:
                    mod.scaled_dot_product_attention = _original_sdpa

    _original_sdpa = None


def make_fused_cache(model, bits: int = 3) -> list:
    """Create cache instances for each layer and patch SDPA.

    For hybrid models (e.g. Qwen3.5) that mix standard and linear attention,
    only standard attention layers get TurboQuantKVCache. Linear attention
    layers keep their native ArraysCache.

    Args:
        model: mlx-lm model (must have .layers)
        bits: PolarQuant bits per coordinate (2-4)

    Returns:
        List of caches, one per layer
    """
    patch_sdpa()
    caches = []
    for layer in model.layers:
        if hasattr(layer, 'is_linear') and layer.is_linear:
            from mlx_lm.models.cache import ArraysCache
            caches.append(ArraysCache(size=2))
        else:
            caches.append(TurboQuantKVCache(bits=bits, fused=True))
    return caches
