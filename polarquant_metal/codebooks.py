"""
Precomputed Lloyd-Max codebooks for PolarQuant.

After random orthogonal rotation, each coordinate of a d-dimensional unit vector
is approximately N(0, 1/d). We store codebooks for N(0,1) and scale by 1/sqrt(d)
at load time.

Codebooks are computed once and cached. If scipy is not available, falls back
to hardcoded optimal codebooks for bits 1-4.
"""

import os
from functools import lru_cache

import numpy as np
import mlx.core as mx


# Hardcoded Lloyd-Max codebooks for N(0,1) — computed offline
# These are the optimal scalar quantizer centroids and boundaries
# for a standard normal distribution.
_HARDCODED = {
    1: {
        "centroids": np.array([-0.7978846, 0.7978846], dtype=np.float32),
        "boundaries": np.array([-5.0, 0.0, 5.0], dtype=np.float32),
    },
    2: {
        "centroids": np.array(
            [-1.5104176, -0.4527800, 0.4527800, 1.5104176], dtype=np.float32
        ),
        "boundaries": np.array(
            [-5.0, -0.9815988, 0.0, 0.9815988, 5.0], dtype=np.float32
        ),
    },
    3: {
        "centroids": np.array(
            [-2.1519732, -1.3439093, -0.7560052, -0.2451210,
             0.2451210, 0.7560052, 1.3439093, 2.1519732],
            dtype=np.float32,
        ),
        "boundaries": np.array(
            [-5.0, -1.7479413, -1.0500073, -0.5005631, 0.0,
             0.5005631, 1.0500073, 1.7479413, 5.0],
            dtype=np.float32,
        ),
    },
    4: {
        "centroids": np.array(
            [-2.7326374, -2.0690790, -1.6180005, -1.2562147,
             -0.9423403, -0.6567590, -0.3880823, -0.1284378,
             0.1284378, 0.3880823, 0.6567590, 0.9423403,
             1.2562147, 1.6180005, 2.0690790, 2.7326374],
            dtype=np.float32,
        ),
        "boundaries": np.array(
            [-5.0, -2.4008582, -1.8435397, -1.4371076, -1.0992775,
             -0.7995497, -0.5224206, -0.2582601, 0.0,
             0.2582601, 0.5224206, 0.7995497, 1.0992775,
             1.4371076, 1.8435397, 2.4008582, 5.0],
            dtype=np.float32,
        ),
    },
}


@lru_cache(maxsize=32)
def load_codebook(bits: int, dim: int) -> tuple[mx.array, mx.array]:
    """Load Lloyd-Max codebook scaled for a given dimension.

    After rotation, each coordinate of a unit vector in R^d is approximately
    N(0, 1/d). Codebooks are precomputed for N(0,1) and scaled by 1/sqrt(d).

    Args:
        bits: Quantization bits per coordinate (1-4)
        dim: Head dimension

    Returns:
        centroids: (2^bits,) MLX array of reconstruction values
        boundaries: (2^bits + 1,) MLX array of decision boundaries
    """
    if bits not in _HARDCODED:
        raise ValueError(f"No codebook for {bits}-bit. Supported: 1-4")

    raw = _HARDCODED[bits]
    scale = 1.0 / np.sqrt(dim)

    centroids = mx.array(raw["centroids"] * scale)
    boundaries = mx.array(raw["boundaries"] * scale)
    return centroids, boundaries


@lru_cache(maxsize=32)
def load_codebook_f32(bits: int, dim: int) -> mx.array:
    """Load just the centroids as float32 for the Metal kernel.

    Args:
        bits: Quantization bits per coordinate (1-4)
        dim: Head dimension

    Returns:
        centroids: (2^bits,) float32 MLX array
    """
    centroids, _ = load_codebook(bits, dim)
    return centroids.astype(mx.float32)
