"""CLI tool to process a batch of banana images from a directory and export results.csv."""

import argparse
from pathlib import Path
import sys
import time

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
from banana_ripeness.config import load_config
from banana_ripeness.io_utils import list_images, parse_filename_metadata, write_image
from banana_ripeness.pipeline import BananaRipenessPipeline
from banana_ripeness.visualization import stage_panel


def main():
    parser = argparse.ArgumentParser(description="Deteksi Tingkat Kematangan Pisang (Batch Processing)")
    parser.add_argument("input_dir", nargs="?", default="Data/Raw", help="Direktori berisi citra masukan")
    parser.add_argument("-o", "--out", type=str, default="Data/output", help="Direktori output")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path ke file konfigurasi YAML")
    parser.add_argument("--save-panels", action="store_true", default=True, help="Simpan visualisasi panel 6-tahap")

    args = parser.parse_args()

    in_dir = Path(args.input_dir).resolve()
    if not in_dir.exists():
        print(f"[ERROR] Direktori masukan tidak ditemukan: {in_dir}")
        sys.exit(1)

    images = list_images(in_dir)
    if not images:
        print(f"[ERROR] Tidak ada citra JPG/PNG ditemukan di: {in_dir}")
        sys.exit(1)

    out_dir = Path(args.out).resolve()
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    panels_dir = out_dir / "panels"

    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.save_panels:
        panels_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    pipeline = BananaRipenessPipeline(cfg=cfg)

    print(f"\n=======================================================")
    print(f" Memproses Batch: {len(images)} citra dari '{in_dir.name}'")
    print(f" Output tujuan  : {out_dir}")
    print(f"=======================================================\n")

    records = []
    t_start = time.time()

    for idx, img_path in enumerate(images, 1):
        t0 = time.time()
        res = pipeline.run_file(img_path)
        dt = time.time() - t0

        meta = parse_filename_metadata(img_path)
        match_flag = (meta["label"] == res.label) if meta["label"] else None

        # Save artifacts
        write_image(masks_dir / f"{img_path.stem}.png", res.mask)
        write_image(overlays_dir / f"{img_path.stem}.png", res.stages["overlay"])
        if args.save_panels:
            stage_panel(res.stages, label=res.label, features=res.features, output_path=panels_dir / f"{img_path.stem}.png")

        record = {
            "no": idx,
            "filename": img_path.name,
            "device": meta.get("device"),
            "specimen": meta.get("specimen"),
            "side": meta.get("side"),
            "ground_truth": meta.get("label"),
            "pred_label": res.label,
            "is_correct": match_flag,
            "mean_h": res.features["mean_h"],
            "mean_s": res.features["mean_s"],
            "mean_v": res.features["mean_v"],
            "ratio_green": res.features["ratio_green"],
            "ratio_yellow": res.features["ratio_yellow"],
            "ratio_brown": res.features["ratio_brown"],
            "area_pixels": res.features["area_pixels"],
            "area_ratio": res.features["area_ratio"],
            "elapsed_ms": round(dt * 1000, 2),
        }
        records.append(record)

        match_badge = "[OK]" if match_flag else ("[FAIL]" if match_flag is False else "[-]")
        print(f"[{idx:02d}/{len(images):02d}] {img_path.name:<40} -> {res.label:<15} {match_badge} ({dt*1000:.1f} ms)")

    total_time = time.time() - t_start
    avg_time = total_time / len(images)

    df = pd.DataFrame(records)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n=======================================================")
    print(f" RINGKASAN BATCH")
    print(f"=======================================================")
    print(f"  Total Citra Diproses : {len(images)}")
    print(f"  Waktu Total          : {total_time:.2f} detik (Rata-rata: {avg_time*1000:.1f} ms/citra)")
    if "is_correct" in df.columns and df["is_correct"].notna().any():
        valid_df = df[df["is_correct"].notna()]
        correct_count = int(valid_df["is_correct"].sum())
        acc = (correct_count / len(valid_df)) * 100.0
        print(f"  Akurasi Klasifikasi  : {correct_count}/{len(valid_df)} ({acc:.1f}%)")
    print(f"  File CSV Disimpan    : {csv_path}")
    print(f"  Masker Disimpan      : {masks_dir}/")
    print(f"  Overlay Disimpan     : {overlays_dir}/")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
