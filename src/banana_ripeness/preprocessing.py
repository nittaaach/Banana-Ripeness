"""Preprocessing module: resizing, Gaussian filtering, and contrast stretching."""

from typing import Tuple, Union
import cv2
import numpy as np

from .config import Config, PreprocessingConfig


def resize(img: np.ndarray, size: Tuple[int, int] = (600, 600)) -> np.ndarray:
    """Resize image to standard dimensions (width, height)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def gaussian_blur(
    img: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian blur to reduce high-frequency texture noise."""
    # Ensure kernel dimensions are odd
    kw = ksize[0] if ksize[0] % 2 == 1 else ksize[0] + 1
    kh = ksize[1] if ksize[1] % 2 == 1 else ksize[1] + 1
    return cv2.GaussianBlur(img, (kw, kh), sigmaX=sigma, sigmaY=sigma)


def contrast_stretch(
    img: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Apply percentile-based contrast stretching on the Value (V) channel in HSV space.

    Preserves color hues and saturation while boosting dynamic range when sufficient
    variance exists (v_max - v_min > 30).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    v_min, v_max = np.percentile(v, (p_low, p_high))
    if v_max - v_min > 30:
        v_stretched = np.clip((v - v_min) * (255.0 / (v_max - v_min)), 0, 255)
        hsv[:, :, 2] = v_stretched.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def preprocess(
    img: np.ndarray,
    cfg: Union[Config, PreprocessingConfig],
) -> np.ndarray:
    """Execute complete preprocessing stage: resize -> (optional) contrast stretch -> Gaussian blur."""
    if isinstance(cfg, Config):
        size = cfg.image.size
        p_cfg = cfg.preprocessing
    else:
        size = (600, 600)
        p_cfg = cfg

    # 1. Resize
    out = resize(img, size)

    # 2. Contrast stretching
    if p_cfg.contrast_stretch:
        low_p, high_p = p_cfg.stretch_percentiles
        out = contrast_stretch(out, p_low=low_p, p_high=high_p)

    # 3. Gaussian Blur
    out = gaussian_blur(out, ksize=p_cfg.gaussian_kernel, sigma=p_cfg.gaussian_sigma)
    return out
