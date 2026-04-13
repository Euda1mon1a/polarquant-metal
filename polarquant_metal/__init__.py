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
from .tree_search import MCTSTree, MCTSNode, fork_layer_caches, draft_logprob_evaluator
from .prm import ProcessRewardModel

__all__ = [
    "polarquant_qk_matmul",
    "polarquant_sv_matmul",
    "FusedPolarQuantKVCache",
    "MCTSTree",
    "MCTSNode",
    "fork_layer_caches",
    "draft_logprob_evaluator",
    "ProcessRewardModel",
]
