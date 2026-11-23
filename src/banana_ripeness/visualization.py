"""Visualization utilities: overlay blending and multi-stage pipeline debug panels."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


LABEL_COLORS = {
    "Mentah": (0, 180, 0),          # Green
    "Matang": (0, 215, 255),        # Yellow/Gold (BGR: B=0, G=215, R=255)
    "Terlalu Matang": (0, 100, 230),# Orange
    "Busuk": (30, 30, 140),         # Dark Red/Brown
}


def overlay(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Create transparent colored mask overlay with distinct perimeter contour."""
    if color is None:
        color = (0, 255, 0)  # Default neon green

    out = img_bgr.copy()
    m_bool = mask > 0

    colored_layer = np.zeros_like(img_bgr)
    colored_layer[m_bool] = color

    # Alpha blend inside mask
    out[m_bool] = cv2.addWeighted(img_bgr, 1.0 - alpha, colored_layer, alpha, 0)[m_bool]

    # Draw crisp contour border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def stage_panel(
    stages: Dict[str, np.ndarray],
    label: str = "",
    features: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Generate a 2x3 grid displaying all 6 pipeline stages with labels.

    Stages:
    Row 1: 1. Citra Asli | 2. Preprocessed | 3. Mask HSV
    Row 2: 4. Mask Otsu | 5. Morfologi (Final) | 6. Overlay & Prediksi
    """
    def ensure_bgr_600(arr: np.ndarray) -> np.ndarray:
        out = arr
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        if out.shape[:2] != (600, 600):
            out = cv2.resize(out, (600, 600), interpolation=cv2.INTER_AREA)
        return out

    def annotate(img: np.ndarray, text: str, subtext: str = "") -> np.ndarray:
        canvas = img.copy()
        # Top banner
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if subtext:
            cv2.rectangle(canvas, (0, canvas.shape[0] - 28), (canvas.shape[1], canvas.shape[0]), (20, 20, 20), -1)
            cv2.putText(canvas, subtext, (10, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return canvas

    p1 = annotate(ensure_bgr_600(stages.get("raw", np.zeros((600, 600, 3), dtype=np.uint8))), "1. Citra Asli", "Ukuran Awal")
    p2 = annotate(ensure_bgr_600(stages.get("preprocessed", np.zeros((600, 600, 3), dtype=np.uint8))), "2. Preprocessing", "Gaussian (5x5) + Kontras")
    p3 = annotate(ensure_bgr_600(stages.get("mask_hsv", np.zeros((600, 600), dtype=np.uint8))), "3. Masker HSV", "inRange Hijau-Kuning-Coklat")
    p4 = annotate(ensure_bgr_600(stages.get("mask_otsu", np.zeros((600, 600), dtype=np.uint8))), "4. Masker Otsu", "Auto-threshold Kanal S/V")
    p5 = annotate(ensure_bgr_600(stages.get("final", np.zeros((600, 600), dtype=np.uint8))), "5. Morfologi", "Open, Close, Kontur Terbesar")

    overlay_img = stages.get("overlay", np.zeros((600, 600, 3), dtype=np.uint8))
    pred_sub = f"Prediksi: {label.upper()}" if label else "Overlay"
    if features:
        sub_info = f"{pred_sub} | G:{features.get('ratio_green',0):.2f} Y:{features.get('ratio_yellow',0):.2f} B:{features.get('ratio_brown',0):.2f}"
    else:
        sub_info = pred_sub
    p6 = annotate(ensure_bgr_600(overlay_img), "6. Hasil & Diagnosis", sub_info)

    # Tile 2 rows of 3 columns
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    grid = np.vstack([row1, row2])

    if output_path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), grid)

    return grid
