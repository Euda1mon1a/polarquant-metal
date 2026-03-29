"""
Bit-packing utilities for PolarQuant indices.

Pack b-bit codebook indices into uint32 for compact storage and
efficient Metal kernel access.
"""

import mlx.core as mx
import numpy as np


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack b-bit indices into uint32 words.

    Args:
        indices: (..., D) uint8 array with values in [0, 2^bits)
        bits: bits per index (2, 3, or 4)

    Returns:
        packed: (..., D_packed) uint32 array
        where D_packed = ceil(D / vals_per_int), vals_per_int = 32 // bits
    """
    assert bits in (2, 3, 4), f"Unsupported: {bits}"
    vals_per_int = 32 // bits

    shape = indices.shape
    D = shape[-1]
    flat = indices.reshape(-1, D)
    N = flat.shape[0]
    D_packed = (D + vals_per_int - 1) // vals_per_int

    # Pad D to multiple of vals_per_int
    if D % vals_per_int != 0:
        pad_width = vals_per_int - (D % vals_per_int)
        flat = mx.concatenate(
            [flat, mx.zeros((N, pad_width), dtype=flat.dtype)], axis=-1
        )

    # Reshape to (N, D_packed, vals_per_int)
    flat = flat.reshape(N, D_packed, vals_per_int)

    # Pack: shift each value by its bit position and OR together
    # Use numpy for the packing since it's a one-time operation
    flat_np = np.array(flat, dtype=np.uint32)
    packed = np.zeros((N, D_packed), dtype=np.uint32)
    for i in range(vals_per_int):
        packed |= flat_np[:, :, i] << (i * bits)

    result = mx.array(packed)
    return result.reshape(*shape[:-1], D_packed)


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 words back to b-bit indices.

    Args:
        packed: (..., D_packed) uint32 array
        bits: bits per index (2, 3, or 4)
        dim: original dimension D

    Returns:
        indices: (..., D) uint8 array
    """
    vals_per_int = 32 // bits
    mask = (1 << bits) - 1

    shape = packed.shape
    D_packed = shape[-1]
    flat = packed.reshape(-1, D_packed)
    N = flat.shape[0]

    # Unpack using numpy
    flat_np = np.array(flat, dtype=np.uint32)
    total_vals = D_packed * vals_per_int
    unpacked = np.zeros((N, total_vals), dtype=np.uint8)
    for i in range(vals_per_int):
        unpacked[:, i::vals_per_int] = ((flat_np >> (i * bits)) & mask).astype(
            np.uint8
        )

    # The packing order is: val0 at bits [0:b), val1 at [b:2b), etc.
    # So we need to reorder: for word w, vals are at positions
    # w*vals_per_int + 0, w*vals_per_int + 1, ...
    reordered = np.zeros_like(unpacked)
    for w in range(D_packed):
        for v in range(vals_per_int):
            src_col = v * D_packed + w  # from the strided unpack above
            if src_col < total_vals:
                dst_col = w * vals_per_int + v
                if dst_col < total_vals:
                    reordered[:, dst_col] = flat_np[:, w] >> (v * bits) & mask

    result = mx.array(reordered[:, :dim].astype(np.uint8))
    return result.reshape(*shape[:-1], dim)
