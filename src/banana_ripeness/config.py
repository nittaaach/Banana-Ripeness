"""Configuration data structures and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import yaml


@dataclass
class DeviceConfig:
    use_gpu: bool = True
    device_type: str = "gpu"


@dataclass
class ImageConfig:
    size: Tuple[int, int] = (600, 600)


@dataclass
class PreprocessingConfig:
    gaussian_kernel: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 0.0
    contrast_stretch: bool = True
    stretch_percentiles: Tuple[float, float] = (2.0, 98.0)


@dataclass
class SegmentationConfig:
    hsv_lower: List[int] = field(default_factory=lambda: [0, 20, 20])
    hsv_upper: List[int] = field(default_factory=lambda: [100, 255, 255])
    otsu_channel: str = "S"  # "S" atau "V"
    combine_mode: str = "or"  # "or" atau "and"
    auto_invert_otsu: bool = True


@dataclass
class MorphologyConfig:
    kernel_shape: str = "ellipse"  # "ellipse", "rect", atau "cross"
    kernel_size: Tuple[int, int] = (5, 5)
    open_iterations: int = 1
    close_iterations: int = 3
    keep_largest_component: bool = True
    fill_holes: bool = True


@dataclass
class FeaturesConfig:
    green_hue_range: Tuple[int, int] = (36, 85)
    yellow_hue_range: Tuple[int, int] = (20, 35)
    brown_hue_range: Tuple[int, int] = (0, 22)
    brown_max_v: int = 110
    brown_min_s: int = 20


@dataclass
class ClassificationConfig:
    hue_green_max: int = 35
    hue_yellow_max: int = 28
    green_ratio_unripe: float = 0.35
    yellow_ratio_ripe: float = 0.40
    brown_ratio_overripe: float = 0.40
    brown_ratio_rotten: float = 0.70
    rotten_max_sat: float = 100.0


@dataclass
class EvaluationConfig:
    success_iou_threshold: float = 0.80


@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        if "device" in data:
            d = data["device"]
            cfg.device.use_gpu = bool(d.get("use_gpu", True))
            cfg.device.device_type = str(d.get("device_type", "gpu")).lower()

        if "image" in data:
            sz = data["image"].get("size", [600, 600])
            cfg.image.size = tuple(sz)  # type: ignore

        if "preprocessing" in data:
            p = data["preprocessing"]
            cfg.preprocessing.gaussian_kernel = tuple(p.get("gaussian_kernel", [5, 5]))  # type: ignore
            cfg.preprocessing.gaussian_sigma = float(p.get("gaussian_sigma", 0))
            cfg.preprocessing.contrast_stretch = bool(p.get("contrast_stretch", True))
            cfg.preprocessing.stretch_percentiles = tuple(p.get("stretch_percentiles", [2, 98]))  # type: ignore

        if "segmentation" in data:
            s = data["segmentation"]
            cfg.segmentation.hsv_lower = list(s.get("hsv_lower", [0, 20, 20]))
            cfg.segmentation.hsv_upper = list(s.get("hsv_upper", [100, 255, 255]))
            cfg.segmentation.otsu_channel = str(s.get("otsu_channel", "S"))
            cfg.segmentation.combine_mode = str(s.get("combine_mode", "or")).lower()
            cfg.segmentation.auto_invert_otsu = bool(s.get("auto_invert_otsu", True))

        if "morphology" in data:
            m = data["morphology"]
            cfg.morphology.kernel_shape = str(m.get("kernel_shape", "ellipse"))
            cfg.morphology.kernel_size = tuple(m.get("kernel_size", [5, 5]))  # type: ignore
            cfg.morphology.open_iterations = int(m.get("open_iterations", 1))
            cfg.morphology.close_iterations = int(m.get("close_iterations", 3))
            cfg.morphology.keep_largest_component = bool(m.get("keep_largest_component", True))
            cfg.morphology.fill_holes = bool(m.get("fill_holes", True))

        if "features" in data:
            f = data["features"]
            cfg.features.green_hue_range = tuple(f.get("green_hue_range", [36, 85]))  # type: ignore
            cfg.features.yellow_hue_range = tuple(f.get("yellow_hue_range", [20, 35]))  # type: ignore
            cfg.features.brown_hue_range = tuple(f.get("brown_hue_range", [0, 22]))  # type: ignore
            cfg.features.brown_max_v = int(f.get("brown_max_v", 110))
            cfg.features.brown_min_s = int(f.get("brown_min_s", 20))

        if "classification" in data:
            c = data["classification"]
            cfg.classification.hue_green_max = int(c.get("hue_green_max", 35))
            cfg.classification.hue_yellow_max = int(c.get("hue_yellow_max", 28))
            cfg.classification.green_ratio_unripe = float(c.get("green_ratio_unripe", 0.35))
            cfg.classification.yellow_ratio_ripe = float(c.get("yellow_ratio_ripe", 0.40))
            cfg.classification.brown_ratio_overripe = float(c.get("brown_ratio_overripe", 0.40))
            cfg.classification.brown_ratio_rotten = float(c.get("brown_ratio_rotten", 0.70))
            cfg.classification.rotten_max_sat = float(c.get("rotten_max_sat", 100.0))

        if "evaluation" in data:
            e = data["evaluation"]
            cfg.evaluation.success_iou_threshold = float(e.get("success_iou_threshold", 0.80))

        return cfg


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or return defaults."""
    if config_path is None:
        default_yaml = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if default_yaml.exists():
            config_path = default_yaml
        else:
            return Config()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config.from_dict(data)
