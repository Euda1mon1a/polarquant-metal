"""
PolarQuant Metal: Fused Metal kernels for PolarQuant KV cache on Apple Silicon.

Provides fused dequantize-matmul kernels that avoid the dequantize-on-fetch
bottleneck in PolarQuant KV cache implementations.
"""

from .kernels import (
    polarquant_qk_matmul,
    polarquant_sv_matmul,
)
from .cache import FusedPolarQuantKVCache

__all__ = [
    "polarquant_qk_matmul",
    "polarquant_sv_matmul",
    "FusedPolarQuantKVCache",
]
