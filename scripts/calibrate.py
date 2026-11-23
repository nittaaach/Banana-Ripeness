"""Calibration tool: extract and display color distributions across the test dataset."""

from pathlib import Path
import sys

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
from banana_ripeness.config import load_config
from banana_ripeness.io_utils import list_images, parse_filename_metadata
from banana_ripeness.morphology import refine
from banana_ripeness.features import extract_features
from banana_ripeness.preprocessing import preprocess
from banana_ripeness.segmentation import segment
from banana_ripeness.io_utils import read_image


def main():
    raw_dir = Path("Data/Raw").resolve()
    images = list_images(raw_dir)
    cfg = load_config()

    print(f"Menganalisis distribusi fitur warna pada {len(images)} citra...\n")
    rows = []
    for img_p in images:
        meta = parse_filename_metadata(img_p)
        img = read_image(img_p)
        pre = preprocess(img, cfg)
        seg = segment(pre, cfg)
        mask = refine(seg["combined"], cfg)
        feats = extract_features(pre, mask, cfg)

        row = {
            "file": img_p.name,
            "condition": meta.get("label"),
            "side": meta.get("side"),
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df[["file", "condition", "mean_h", "ratio_green", "ratio_yellow", "ratio_brown"]].to_string())

    print("\n=== RINGKASAN RATA-RATA FITUR PER KONDISI ===")
    summary = df.groupby("condition")[["mean_h", "mean_s", "mean_v", "ratio_green", "ratio_yellow", "ratio_brown"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
