"""CLI tool to process a single banana image, perform hybrid segmentation, and classify ripeness."""

import argparse
from pathlib import Path
import sys

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
from banana_ripeness.config import load_config
from banana_ripeness.io_utils import parse_filename_metadata, write_image
from banana_ripeness.pipeline import BananaRipenessPipeline
from banana_ripeness.visualization import stage_panel


def main():
    parser = argparse.ArgumentParser(description="Deteksi Tingkat Kematangan Pisang (Single Image Inference)")
    parser.add_argument("image_path", type=str, help="Path ke berkas citra pisang (JPG/PNG)")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path ke file konfigurasi YAML kustom")
    parser.add_argument("-o", "--out", type=str, default=None, help="Folder tujuan untuk menyimpan citra hasil")
    parser.add_argument("--show", action="store_true", help="Tampilkan visualisasi panel tahapan di jendela popup")
    parser.add_argument("--save-stages", action="store_true", help="Simpan semua citra antara tiap tahap secara terpisah")

    args = parser.parse_args()

    img_path = Path(args.image_path).resolve()
    if not img_path.exists():
        print(f"[ERROR] Berkas citra tidak ditemukan: {img_path}")
        sys.exit(1)

    cfg = load_config(args.config)
    pipeline = BananaRipenessPipeline(cfg=cfg)

    print(f"\n=======================================================")
    print(f" Memproses: {img_path.name}")
    print(f"=======================================================")

    res = pipeline.run_file(img_path)
    meta = parse_filename_metadata(img_path)

    print(f"\n[HASIL PREDIKSI]")
    print(f"  Label Kematangan : {res.label.upper()}")
    if meta.get("label"):
        match_str = "[SESUAI]" if meta["label"] == res.label else "[TIDAK SESUAI]"
        print(f"  Ground Truth     : {meta['label']} {match_str}")
    print(f"  Alasan Aturan    : {res.decision_info.get('rule')}")

    print(f"\n[FITUR WARNA & AREA]")
    print(f"  Luas Objek (px)  : {res.features['area_pixels']:,} px ({res.features['area_ratio']*100:.1f}% citra)")
    print(f"  Rata-rata HSV    : H={res.features['mean_h']:.1f}, S={res.features['mean_s']:.1f}, V={res.features['mean_v']:.1f}")
    print(f"  Rasio Hijau      : {res.features['ratio_green']*100:.2f}%")
    print(f"  Rasio Kuning     : {res.features['ratio_yellow']*100:.2f}%")
    print(f"  Rasio Coklat/Busuk: {res.features['ratio_brown']*100:.2f}%")

    out_dir = Path(args.out) if args.out else Path("Data/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = out_dir / f"{img_path.stem}_panel.png"
    panel_img = stage_panel(res.stages, label=res.label, features=res.features, output_path=panel_path)
    print(f"\n[FILE DISIMPAN]")
    print(f"  Panel 6-Tahap    : {panel_path}")

    mask_path = out_dir / f"{img_path.stem}_mask.png"
    write_image(mask_path, res.mask)
    print(f"  Masker Biner     : {mask_path}")

    overlay_path = out_dir / f"{img_path.stem}_overlay.png"
    write_image(overlay_path, res.stages["overlay"])
    print(f"  Overlay Visual   : {overlay_path}")

    if args.save_stages:
        stages_dir = out_dir / f"{img_path.stem}_stages"
        stages_dir.mkdir(parents=True, exist_ok=True)
        for stage_name, stage_arr in res.stages.items():
            st_path = stages_dir / f"{stage_name}.png"
            write_image(st_path, stage_arr)
        print(f"  Tahapan Terpisah : {stages_dir}/")

    if args.show:
        try:
            cv2.imshow("Panel Tahapan Pipeline Kematangan Pisang", panel_img)
            print("\nTekan sembarang tombol pada jendela citra untuk menutup...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[INFO] Tampilan GUI tidak tersedia di environment ini ({e}).")


if __name__ == "__main__":
    main()
