"""Classification module: 4-class rule-based decision logic and extensible classifier."""

from typing import Any, Dict, Tuple, Union

from .config import ClassificationConfig, Config

CANONICAL_CLASSES = ["Mentah", "Matang", "Terlalu Matang", "Busuk"]


def classify(
    features: Dict[str, Any],
    cfg: Union[Config, ClassificationConfig],
) -> str:
    """Classify banana ripeness into 4 classes based on calibrated color distributions.

    Rule Priority:
    1. Mentah: dominant green spectrum (ratio_green >= green_ratio_unripe or mean_h >= hue_green_max)
    2. Matang: dominant yellow spectrum (ratio_yellow >= yellow_ratio_ripe and mean_h >= 20 and ratio_brown < 0.45)
    3. Busuk: extensive decay (ratio_brown >= brown_ratio_rotten and (mean_h <= 11.0 or mean_v <= 65 or mean_s <= rotten_max_sat))
    4. Terlalu Matang: overripe spotting (ratio_brown >= brown_ratio_overripe or fallback)
    """
    label, _ = classify_with_details(features, cfg)
    return label


def classify_with_details(
    features: Dict[str, Any],
    cfg: Union[Config, ClassificationConfig],
) -> Tuple[str, Dict[str, Any]]:
    """Classify banana ripeness and return diagnostic reasons for rule triggers."""
    c_cfg = cfg.classification if isinstance(cfg, Config) else cfg

    r_brown = features.get("ratio_brown", 0.0)
    r_green = features.get("ratio_green", 0.0)
    r_yellow = features.get("ratio_yellow", 0.0)
    mean_h = features.get("mean_h", 0.0)
    mean_s = features.get("mean_s", 0.0)
    mean_v = features.get("mean_v", 0.0)

    # 1. Check Unripe (Mentah)
    if r_green >= c_cfg.green_ratio_unripe or (mean_h >= c_cfg.hue_green_max and r_yellow < 0.30):
        return "Mentah", {
            "rule": "ratio_green >= green_ratio_unripe or dominant green hue",
            "threshold": c_cfg.green_ratio_unripe,
            "value": r_green,
        }

    # 2. Check Rotten (Busuk)
    # Busuk exhibits severe decay (ratio_brown >= brown_ratio_rotten) with low saturation, low value, or deep hue
    sat_thresh = getattr(c_cfg, "rotten_max_sat", 100.0)
    if r_brown >= c_cfg.brown_ratio_rotten and (mean_h <= 11.0 or mean_v <= 65.0 or mean_s <= sat_thresh):
        return "Busuk", {
            "rule": "ratio_brown >= brown_ratio_rotten and (mean_h <= 11.0 or mean_v <= 65 or mean_s <= 100)",
            "threshold": c_cfg.brown_ratio_rotten,
            "value": r_brown,
        }

    # 3. Check Overripe (Terlalu Matang)
    if r_brown >= c_cfg.brown_ratio_overripe:
        return "Terlalu Matang", {
            "rule": "ratio_brown >= brown_ratio_overripe",
            "threshold": c_cfg.brown_ratio_overripe,
            "value": r_brown,
        }

    # 4. Check Ripe (Matang)
    return "Matang", {
        "rule": "dominant yellow / within ripe boundaries",
        "threshold": None,
        "value": r_yellow,
    }
