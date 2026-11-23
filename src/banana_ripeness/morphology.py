"""Morphological refinement module: opening, closing, largest component selection, and hole filling."""

from typing import Tuple, Union
import cv2
import numpy as np

from .config import Config, MorphologyConfig


def make_kernel(
    shape: str = "ellipse",
    size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Construct OpenCV structuring element kernel."""
    shape_map = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
    }
    morph_shape = shape_map.get(shape.lower(), cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(morph_shape, size)


def opening(mask: np.ndarray, kernel: np.ndarray, iters: int = 1) -> np.ndarray:
    """Apply morphological Opening (erosion followed by dilation) to discard isolated noise."""
    if iters <= 0:
        return mask
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)


def closing(mask: np.ndarray, kernel: np.ndarray, iters: int = 2) -> np.ndarray:
    """Apply morphological Closing (dilation followed by erosion) to seal internal dark spots."""
    if iters <= 0:
        return mask
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Retain only the largest connected foreground component, discarding stray clusters."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    # Exclude background label 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0 or np.max(areas) == 0:
        return mask

    largest_label = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill all internal holes within external foreground contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def refine(
    mask: np.ndarray,
    cfg: Union[Config, MorphologyConfig],
) -> np.ndarray:
    """Execute complete morphological refinement: Open -> Close -> Largest Component -> Fill Holes."""
    m_cfg = cfg.morphology if isinstance(cfg, Config) else cfg

    kernel = make_kernel(m_cfg.kernel_shape, m_cfg.kernel_size)

    # 1. Opening
    out = opening(mask, kernel, iters=m_cfg.open_iterations)

    # 2. Closing
    out = closing(out, kernel, iters=m_cfg.close_iterations)

    # 3. Largest connected component
    if m_cfg.keep_largest_component:
        out = largest_component(out)

    # 4. Fill internal holes
    if m_cfg.fill_holes:
        out = fill_holes(out)

    return out
