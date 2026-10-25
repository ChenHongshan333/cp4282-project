"""PSNR, SSIM, and PSNR inside and outside a mask. Images are float arrays in [0, 1], (..., H, W, 3)."""

from __future__ import annotations

import numpy as np


def psnr(image, target) -> float:
    return float(-10.0 * np.log10(np.mean((np.clip(image, 0, 1) - target) ** 2)))


def psnr_per_frame(images, targets) -> np.ndarray:
    mse = np.mean((np.clip(images, 0, 1) - targets) ** 2, axis=(1, 2, 3))
    return -10.0 * np.log10(mse)


def masked_psnr(image, target, mask) -> float:
    """PSNR over the pixels where ``mask`` (H, W) is true; NaN if the mask is empty."""
    if not mask.any():
        return float("nan")
    return float(-10.0 * np.log10(np.mean((np.clip(image, 0, 1)[mask] - target[mask]) ** 2)))


def _blur(x, sigma=1.5, radius=5):
    """Separable Gaussian filter over the two image axes, valid region only."""
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 0, x)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, x)


def ssim(image, target) -> float:
    """SSIM of Wang et al. (2004) with an 11x11 Gaussian window, averaged over the RGB channels."""
    image = np.clip(image, 0, 1).astype(np.float64)
    target = target.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = _blur(image), _blur(target)
    sxx = _blur(image * image) - mx * mx
    syy = _blur(target * target) - my * my
    sxy = _blur(image * target) - mx * my
    value = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(value.mean())
