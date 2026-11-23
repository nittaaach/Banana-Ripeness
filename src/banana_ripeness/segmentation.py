"""Segmentation module: HSV color thresholding, Otsu thresholding, and mask combination."""

from typing import Any, Dict, Sequence, Union
import cv2
import numpy as np

from .config import Config, SegmentationConfig


def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV color space."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def hsv_mask(
    hsv: np.ndarray,
    lower: Sequence[int],
    upper: Sequence[int],
) -> np.ndarray:
    """Create a binary mask isolating pixels within HSV bounds."""
    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower_np, upper_np)


def otsu_mask(channel: np.ndarray, auto_invert: bool = True) -> np.ndarray:
    """Calculate Otsu automatic threshold on a single grayscale/channel image.

    If auto_invert is enabled, analyzes border boundary pixels. If border pixels are
    predominantly foreground (white), inverts the mask so the main central object
    becomes the foreground.
    """
    _, mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if auto_invert:
        # Check boundary pixels along borders (top, bottom, left, right)
        border_pixels = np.concatenate([
            mask[0, :],
            mask[-1, :],
            mask[:, 0],
            mask[:, -1],
        ])
        # If over 50% of the image perimeter is 255, the background was thresholded as object
        if np.mean(border_pixels > 0) > 0.5:
            mask = cv2.bitwise_not(mask)

    return mask


def combine_masks(m_hsv: np.ndarray, m_otsu: np.ndarray, mode: str = "or") -> np.ndarray:
    """Combine HSV mask and Otsu mask using logical OR or AND."""
    mode_clean = mode.strip().lower()
    if mode_clean == "and":
        return cv2.bitwise_and(m_hsv, m_otsu)
    # Default to logical OR
    return cv2.bitwise_or(m_hsv, m_otsu)


def segment(
    img_pre: np.ndarray,
    cfg: Union[Config, SegmentationConfig],
) -> Dict[str, Any]:
    """Execute hybrid segmentation: HSV inRange + Otsu -> Combined Mask."""
    s_cfg = cfg.segmentation if isinstance(cfg, Config) else cfg

    hsv = to_hsv(img_pre)

    # 1. HSV color bounds mask
    m_hsv = hsv_mask(hsv, s_cfg.hsv_lower, s_cfg.hsv_upper)

    # 2. Otsu thresholding on selected channel (Saturation S=1 or Value V=2)
    ch_idx = 1 if s_cfg.otsu_channel.upper() == "S" else 2
    otsu_ch = hsv[:, :, ch_idx]
    m_otsu = otsu_mask(otsu_ch, auto_invert=s_cfg.auto_invert_otsu)

    # 3. Combine masks
    m_combined = combine_masks(m_hsv, m_otsu, mode=s_cfg.combine_mode)

    return {
        "hsv_img": hsv,
        "otsu_channel": otsu_ch,
        "mask_hsv": m_hsv,
        "mask_otsu": m_otsu,
        "combined": m_combined,
    }
