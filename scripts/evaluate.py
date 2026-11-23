"""CLI tool to evaluate segmentation (IoU, Dice) and classification metrics against ground truth."""

import argparse
from pathlib import Path
import sys

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from banana_ripeness.config import load_config
from banana_ripeness.evaluation import evaluate_folder
from banana_ripeness.io_utils import CANONICAL_CLASSES
from banana_ripeness.pipeline import BananaRipenessPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Sistem Deteksi Kematangan Pisang terhadap Ground Truth")
    parser.add_argument("--raw", type=str, default="Data/Raw", help="Folder citra asli")
    parser.add_argument("--gt", type=str, default="Data/ground_truth", help="Folder masker ground truth")
    parser.add_argument("-o", "--out", type=str, default="Data/output/evaluation_report.csv", help="File CSV laporan")
    parser.add_argument("-c", "--config", type=str, default=None, help="File konfigurasi YAML")

    args = parser.parse_args()

    raw_dir = Path(args.raw).resolve()
    gt_dir = Path(args.gt).resolve()
    cfg = load_config(args.config)
    pipeline = BananaRipenessPipeline(cfg=cfg)

    print(f"\n=======================================================")
    print(f" MENJALANKAN EVALUASI TERHADAP GROUND TRUTH")
    print(f" Raw Directory: {raw_dir}")
    print(f" GT Directory : {gt_dir}")
    print(f"=======================================================\n")

    df, summary = evaluate_folder(raw_dir, gt_dir, pipeline=pipeline)

    # Save detailed evaluation DataFrame
    out_file = Path(args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"{'No':<3} | {'Nama File':<38} | {'GT':<14} | {'Prediksi':<14} | {'IoU':<6} | {'Dice':<6} | {'Status'}")
    print("-" * 100)
    for idx, row in df.iterrows():
        match_str = "BENAR" if row.get("label_match") else "SALAH"
        iou_val = f"{row['iou']:.4f}" if pd_not_na(row.get("iou")) else "N/A"
        dice_val = f"{row['dice']:.4f}" if pd_not_na(row.get("dice")) else "N/A"
        print(f"{idx+1:<3} | {row['filename']:<38} | {str(row['true_label']):<14} | {str(row['pred_label']):<14} | {iou_val:<6} | {dice_val:<6} | {match_str}")

    print("\n" + "=" * 60)
    print(" RINGKASAN METRIK KINERJA SISTEM")
    print("=" * 60)
    print(f" Total Spesimen Citra       : {summary['total_images']}")
    if summary['mean_iou'] is not None:
        print(f" Rata-rata IoU (Segmentasi) : {summary['mean_iou']:.4f}  (Target PRD >= 0.85, Acuan: 0.934)")
        print(f" Rata-rata Dice             : {summary['mean_dice']:.4f}")
        print(f" Keberhasilan Segmentasi    : {summary['success_rate']:.1f}%  (IoU >= {cfg.evaluation.success_iou_threshold:.2f}, Target PRD >= 90%)")
    if summary['accuracy'] is not None:
        print(f" Akurasi Klasifikasi        : {summary['accuracy']:.1f}%  (Target PRD >= 85%)")

    print("\n[Rincian Per Kondisi Kematangan]")
    for cond, stats in summary.get("condition_stats", {}).items():
        iou_str = f"{stats['mean_iou']:.4f}" if stats['mean_iou'] is not None else "N/A"
        succ_str = f"{stats['success_rate']*100:.1f}%" if stats['success_rate'] is not None else "N/A"
        print(f"  * {cond:<14}: N={stats['count']:<2} | Akurasi={stats['accuracy']*100:.1f}% | IoU={iou_str} | Sukses={succ_str}")

    if summary.get("confusion_matrix"):
        print("\n[Confusion Matrix]")
        print(f"  Labels: {CANONICAL_CLASSES}")
        for label_name, row in zip(CANONICAL_CLASSES, summary["confusion_matrix"]):
            print(f"  {label_name:<15}: {row}")

    print(f"\nLaporan rinci telah disimpan ke: {out_file}\n")


def pd_not_na(val):
    import pandas as pd
    return pd.notna(val)


if __name__ == "__main__":
    main()
