"""Pipeline orchestrator for end-to-end banana ripeness segmentation and classification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

from .classifier import classify_with_details
from .config import Config, load_config
from .features import extract_features
from .io_utils import read_image
from .morphology import refine
from .preprocessing import preprocess
from .segmentation import segment
from .visualization import overlay


@dataclass
class PipelineResult:
    """Encapsulates full inference results and intermediate stages."""

    label: str
    mask: np.ndarray
    features: Dict[str, Any]
    decision_info: Dict[str, Any]
    stages: Dict[str, np.ndarray]
    gpu_accelerated: bool = False


class BananaRipenessPipeline:
    """Full hybrid pipeline: Preprocessing -> Segmentation -> Morphology -> Features -> Classification."""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = load_config(config_path)

        # Configure hardware acceleration (GPU via OpenCV OpenCL)
        self.gpu_active = False
        if hasattr(self.cfg, "device") and self.cfg.device.use_gpu:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                self.gpu_active = cv2.ocl.useOpenCL()

    def run(self, img_bgr: np.ndarray) -> PipelineResult:
        """Execute pipeline on a BGR image array."""
        # 1. Preprocess
        pre = preprocess(img_bgr, self.cfg)

        # 2. Hybrid Segmentation (HSV + Otsu)
        seg = segment(pre, self.cfg)

        # 3. Morphology Refinement
        final_mask = refine(seg["combined"], self.cfg)

        # 4. Color Feature Extraction
        features = extract_features(pre, final_mask, self.cfg)

        # 5. Ripeness Classification
        label, decision_info = classify_with_details(features, self.cfg)

        # 6. Visualization Overlay
        vis_overlay = overlay(pre, final_mask, alpha=0.35)

        stages = {
            "raw": img_bgr,
            "preprocessed": pre,
            "mask_hsv": seg["mask_hsv"],
            "mask_otsu": seg["mask_otsu"],
            "combined": seg["combined"],
            "final": final_mask,
            "overlay": vis_overlay,
        }

        return PipelineResult(
            label=label,
            mask=final_mask,
            features=features,
            decision_info=decision_info,
            stages=stages,
            gpu_accelerated=self.gpu_active,
        )

    def run_file(self, image_path: Union[str, Path]) -> PipelineResult:
        """Load image from path and execute pipeline."""
        img = read_image(image_path)
        return self.run(img)
