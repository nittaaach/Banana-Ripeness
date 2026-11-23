"""I/O utilities: reading, writing images, directory discovery, and filename metadata parsing."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


CANONICAL_CLASSES = ["Mentah", "Matang", "Terlalu Matang", "Busuk"]


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image from disk in BGR format with proper Windows path handling."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found at: {p}")

    # Use cv2.imdecode to support paths with non-ASCII or space characters reliably on Windows
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from: {p}")
    return img


def write_image(path: Union[str, Path], img: np.ndarray) -> bool:
    """Write an image to disk, creating parent directories if needed."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if not ext:
        ext = ".png"
        p = p.with_suffix(ext)

    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    with open(p, "wb") as f:
        f.write(buf.tobytes())
    return True


def list_images(
    folder: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> List[Path]:
    """List and sort all image files in a folder matching extensions."""
    p = Path(folder).resolve()
    if not p.exists():
        return []
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    return sorted(files, key=lambda x: x.name)


def parse_filename_metadata(filename_or_path: Union[str, Path]) -> Dict[str, Optional[str]]:
    """Parse banana metadata (device, specimen, side, canonical condition) from filename.

    Example:
        'Realme_11_H11F1_Kanan_matang.jpg'
        -> {
            'device': 'Realme 11',
            'specimen': 'H11F1',
            'day': 'H11',
            'angle': 'F1',
            'side': 'Kanan',
            'label': 'Matang',
            'stem': 'Realme_11_H11F1_Kanan_matang'
        }
    """
    stem = Path(filename_or_path).stem
    parts = stem.split("_")

    meta: Dict[str, Optional[str]] = {
        "device": None,
        "specimen": None,
        "day": None,
        "angle": None,
        "side": None,
        "label": None,
        "stem": stem,
    }

    # Normalize lower case for condition parsing
    lower_stem = stem.lower()
    if "terlalu-matang" in lower_stem or "terlalu_matang" in lower_stem or "terlalumatang" in lower_stem:
        meta["label"] = "Terlalu Matang"
    elif "mentah" in lower_stem:
        meta["label"] = "Mentah"
    elif "busuk" in lower_stem:
        meta["label"] = "Busuk"
    elif "matang" in lower_stem:
        meta["label"] = "Matang"

    if "kanan" in lower_stem:
        meta["side"] = "Kanan"
    elif "kiri" in lower_stem:
        meta["side"] = "Kiri"

    if len(parts) >= 2:
        meta["device"] = f"{parts[0]} {parts[1]}"
    if len(parts) >= 3:
        spec = parts[2]
        meta["specimen"] = spec
        if "F" in spec:
            d_part, a_part = spec.split("F", 1)
            meta["day"] = d_part
            meta["angle"] = f"F{a_part}"
        else:
            meta["day"] = spec

    return meta
