"""
PolarQuant quantizer: Stage 1 of TurboQuant.

Random orthogonal rotation + Lloyd-Max codebook quantization.
"""

import mlx.core as mx
import numpy as np

from .codebooks import load_codebook


class PolarQuant:
    """PolarQuant quantizer for vectors of a fixed dimension.

    Args:
        bits: Bits per coordinate (1-4).
        dim: Vector dimension.
        seed: Random seed for rotation matrix.
    """

    def __init__(self, bits: int, dim: int, seed: int = 42):
        self.bits = bits
        self.dim = dim
        self.n_levels = 2 ** bits

        self.centroids, self.boundaries = load_codebook(bits, dim)

        # Generate fixed random orthogonal rotation (Haar-distributed)
        self.rotation = _generate_rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.T

    def quantize(self, vectors: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize vectors.

        Args:
            vectors: (..., dim) float array

        Returns:
            indices: (..., dim) uint8 codebook indices
            norms: (..., 1) vector norms
        """
        norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
        unit = vectors / mx.maximum(norms, 1e-8)

        # Rotate to decorrelated basis
        rotated = unit @ self.rotation_t

        # Quantize: find nearest centroid for each coordinate
        inner_bounds = self.boundaries[1:-1]
        indices = mx.zeros(rotated.shape, dtype=mx.uint8)
        for i in range(self.n_levels - 1):
            indices = indices + (rotated > inner_bounds[i]).astype(mx.uint8)

        return indices, norms

    def dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        """Reconstruct vectors from quantized form.

        Args:
            indices: (..., dim) uint8 codebook indices
            norms: (..., 1) vector norms

        Returns:
            reconstructed: (..., dim) float array
        """
        rotated_recon = self.centroids[indices]
        unit_recon = rotated_recon @ self.rotation
        return unit_recon * norms

    def quantize_and_reconstruct(
        self, vectors: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize then immediately reconstruct."""
        indices, norms = self.quantize(vectors)
        reconstructed = self.dequantize(indices, norms)
        return reconstructed, indices, norms


def _generate_rotation_matrix(dim: int, seed: int) -> mx.array:
    """Generate Haar-distributed random orthogonal matrix via QR.

    Uses mx.linalg.qr to match rachittshah's turboquant.py (PR #1059).
    """
    key = mx.random.key(seed)
    g = mx.random.normal(shape=(dim, dim), key=key)
    q, r = mx.linalg.qr(g, stream=mx.cpu)
    sign = mx.sign(mx.diag(r))
    sign = mx.where(sign == 0, 1, sign)
    return q * sign
