"""Script to create high-precision baseline ground-truth binary masks for the 16 specimen images."""

from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from banana_ripeness.io_utils import list_images, read_image, write_image
from banana_ripeness.morphology import fill_holes, largest_component


def generate_precise_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Generate high-precision binary mask for banana specimen using background-distance Otsu thresholding."""
    # Resize to standard 600x600 for consistency
    img = cv2.resize(img_bgr, (600, 600), interpolation=cv2.INTER_AREA)

    # Sample background color from 4 corner margins
    corners = np.vstack([
        img[:25, :25].reshape(-1, 3),
        img[:25, -25:].reshape(-1, 3),
        img[-25:, :25].reshape(-1, 3),
        img[-25:, -25:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)

    # Compute Euclidean color distance from background
    diff = np.linalg.norm(img.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    max_d = np.max(diff)
    diff_norm = np.clip(diff * (255.0 / max(1.0, max_d)), 0, 255).astype(np.uint8)

    # Otsu thresholding on background color distance
    _, mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clear perimeter borders
    margin = 8
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0

    # Morphology cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = largest_component(mask)
    mask = fill_holes(mask)

    return mask


def main():
    raw_dir = Path("Data/Raw").resolve()
    gt_dir = Path("Data/ground_truth").resolve()
    gt_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(raw_dir)
    print(f"Membuat masker ground truth referensi untuk {len(images)} citra...")

    for idx, img_p in enumerate(images, 1):
        img = read_image(img_p)
        mask = generate_precise_mask(img)

        out_file = gt_dir / f"{img_p.stem}.png"
        write_image(out_file, mask)
        area = int(np.sum(mask > 0))
        pct = (area / (600 * 600)) * 100.0
        print(f"[{idx:02d}/{len(images):02d}] Masker dibuat: {out_file.name:<40} (Luas: {area:,} px / {pct:.1f}%)")

    print(f"\nSelesai! Seluruh masker ground truth tersimpan di: {gt_dir}")


if __name__ == "__main__":
    main()
