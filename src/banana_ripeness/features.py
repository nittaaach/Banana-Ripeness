"""Feature extraction module: masked color statistics and hue/saturation distribution."""

from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

from .config import Config, FeaturesConfig


def extract_features(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    cfg: Optional[Union[Config, FeaturesConfig]] = None,
) -> Dict[str, Any]:
    """Extract masked color distribution features only on foreground banana pixels.

    Metrics:
    - mean_h, mean_s, mean_v: average channel values
    - ratio_green: proportion of pixels in green hue band
    - ratio_yellow: proportion of pixels in yellow hue band
    - ratio_brown: proportion of pixels in brown/dark overripe spots
    - area_pixels: total foreground pixel count
    - area_ratio: foreground area divided by image area
    """
    total_img_pixels = mask.shape[0] * mask.shape[1]
    foreground_mask = mask > 0
    area_pixels = int(np.sum(foreground_mask))

    if cfg is None:
        f_cfg = FeaturesConfig()
    elif isinstance(cfg, Config):
        f_cfg = cfg.features
    else:
        f_cfg = cfg

    if area_pixels == 0:
        return {
            "mean_h": 0.0,
            "mean_s": 0.0,
            "mean_v": 0.0,
            "ratio_green": 0.0,
            "ratio_yellow": 0.0,
            "ratio_brown": 0.0,
            "area_pixels": 0,
            "area_ratio": 0.0,
        }

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, :, 0][foreground_mask]
    s_vals = hsv[:, :, 1][foreground_mask]
    v_vals = hsv[:, :, 2][foreground_mask]

    # Mean HSV
    mean_h = float(np.mean(h_vals))
    mean_s = float(np.mean(s_vals))
    mean_v = float(np.mean(v_vals))

    # 1. Green Ratio (unripe banana skin)
    g_low, g_high = f_cfg.green_hue_range
    green_mask = (h_vals >= g_low) & (h_vals <= g_high) & (s_vals >= 25)
    ratio_green = float(np.sum(green_mask) / area_pixels)

    # 2. Yellow Ratio (ripe banana skin)
    y_low, y_high = f_cfg.yellow_hue_range
    yellow_mask = (h_vals >= y_low) & (h_vals <= y_high) & (s_vals >= 35) & (v_vals >= 70)
    ratio_yellow = float(np.sum(yellow_mask) / area_pixels)

    # 3. Brown / Necrotic / Dark Ratio (overripe spots or rot)
    b_low, b_high = f_cfg.brown_hue_range
    brown_hue = (h_vals >= b_low) & (h_vals <= b_high)
    brown_spec = brown_hue & (v_vals <= f_cfg.brown_max_v) & (s_vals >= f_cfg.brown_min_s)
    # Also capture deep decay dark pixels (low value regardless of hue)
    deep_dark = (v_vals <= 60)
    rotten_pixels = brown_spec | deep_dark
    ratio_brown = float(np.sum(rotten_pixels) / area_pixels)

    area_ratio = float(area_pixels / total_img_pixels) if total_img_pixels > 0 else 0.0

    return {
        "mean_h": round(mean_h, 2),
        "mean_s": round(mean_s, 2),
        "mean_v": round(mean_v, 2),
        "ratio_green": round(ratio_green, 4),
        "ratio_yellow": round(ratio_yellow, 4),
        "ratio_brown": round(ratio_brown, 4),
        "area_pixels": area_pixels,
        "area_ratio": round(area_ratio, 4),
    }
