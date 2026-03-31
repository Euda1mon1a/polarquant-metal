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
            # During prefill (L_q > 1) with quantized cache, update_and_fetch
            # returns packed stubs. Dequantize for standard SDPA.
            if cache._quantized:
                keys = cache.keys
                values = cache.values

        # Fall through to original (handles prefill, short context, standard)
        return _original_sdpa(
            queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks,
        )

    base_module.scaled_dot_product_attention = _patched_sdpa

    # Also patch any already-imported model modules that copied the reference
    # Scan both mlx_lm and mlx_vlm modules (VLM models import SDPA from mlx_lm.models.base)
    for name, mod in list(sys.modules.items()):
        if (name.startswith("mlx_lm.models.") or name.startswith("mlx_vlm.models.")) and mod is not None:
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
        if (name.startswith("mlx_lm.models.") or name.startswith("mlx_vlm.models.")) and mod is not None:
            if hasattr(mod, "scaled_dot_product_attention"):
                # Only restore if it's our patched version
                if mod.scaled_dot_product_attention is not _original_sdpa:
                    mod.scaled_dot_product_attention = _original_sdpa

    _original_sdpa = None


def adaptive_threshold(default: int = 512, context_file: str = None) -> int:
    """Return a context-aware lazy quantization threshold.

    For long-running sessions (>30 min), compress earlier (256 tokens).
    For quick queries, stay FP16 longer (default 512).
    Reads session duration from OpenClaw context.json if available.
    """
    if context_file is None:
        context_file = os.path.expanduser("~/.openclaw/state/context.json")
    try:
        import json
        with open(context_file) as f:
            ctx = json.load(f)
        # If session has been running >30 min, compress earlier
        session_minutes = ctx.get("session", {}).get("duration_minutes", 0)
        if session_minutes > 30:
            return 256
    except Exception:
        pass
    return default


def make_fused_cache(model, bits: int = 3, bits_v: int = None,
                     boundary_layers: int = 2,
                     min_fused_context: int = 512) -> list:
    """Create cache instances for each layer and patch SDPA.

    For hybrid models (e.g. Qwen3.5) that mix standard and linear attention,
    only standard attention layers get TurboQuantKVCache. Linear attention
    layers keep their native ArraysCache.

    Boundary layer protection: first N and last N standard attention layers
    use FP16 KVCache (no quantization) to preserve quality at the layers
    that matter most. Middle layers get asymmetric K/V compression.

    Args:
        model: mlx-lm model (must have .layers)
        bits: PolarQuant bits for K (default 3)
        bits_v: PolarQuant bits for V (default same as bits). Lower is
            safe because Sparse V skips near-zero positions.
        boundary_layers: number of first/last standard attention layers
            to keep at FP16 (default 2). Set to 0 to compress all.

    Returns:
        List of caches, one per layer
    """
    from mlx_lm.models.cache import KVCache
    patch_sdpa()

    # Identify standard attention layer indices
    std_indices = [i for i, l in enumerate(model.layers)
                   if not (hasattr(l, 'is_linear') and l.is_linear)]
    n_std = len(std_indices)

    # Boundary layers: first N and last N standard attention layers stay FP16
    boundary_set = set()
    if boundary_layers > 0 and n_std > 2 * boundary_layers:
        boundary_set = set(std_indices[:boundary_layers] + std_indices[-boundary_layers:])

    caches = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'is_linear') and layer.is_linear:
            from mlx_lm.models.cache import ArraysCache
            caches.append(ArraysCache(size=2))
        elif i in boundary_set:
            caches.append(KVCache())
        else:
            caches.append(TurboQuantKVCache(bits=bits, bits_v=bits_v, fused=True,
                                            min_fused_context=min_fused_context))
    return caches
