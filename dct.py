"""Discrete Cosine Transform (DCT) wrapper using scipy."""

import numpy as np
from scipy.fft import dct as scipy_dct, idct as scipy_idct


def dct(x, norm='ortho', axis=-1):
    """
    Apply 1D DCT to input array.
    Args:
        x: numpy array of shape (..., length)
        norm: normalization ('ortho' or None)
        axis: axis along which to apply DCT
    Returns:
        DCT coefficients (same shape as x)
    """
    return scipy_dct(x, type=2, norm=norm, axis=axis)


def idct(x, norm='ortho', axis=-1):
    """
    Apply inverse 1D DCT to input array.
    Args:
        x: numpy array of shape (..., length)
        norm: normalization ('ortho' or None)
        axis: axis along which to apply inverse DCT
    Returns:
        Reconstructed signal (same shape as x)
    """
    return scipy_idct(x, type=2, norm=norm, axis=axis)


def dct2(x, norm='ortho'):
    """2D DCT (not used in FAST, kept for completeness)."""
    return scipy_dct(scipy_dct(x, type=2, norm=norm, axis=0), type=2, norm=norm, axis=1)


def idct2(x, norm='ortho'):
    """2D inverse DCT."""
    return scipy_idct(scipy_idct(x, type=2, norm=norm, axis=0), type=2, norm=norm, axis=1)