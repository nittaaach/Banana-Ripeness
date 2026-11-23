"""Evaluation metrics: IoU, Dice, Success Rate, Confusion Matrix, and Batch Directory Evaluator."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .config import Config, load_config
from .io_utils import CANONICAL_CLASSES, list_images, parse_filename_metadata, read_image
from .pipeline import BananaRipenessPipeline


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate Intersection over Union (Jaccard Index) for binary masks."""
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate Dice Similarity Coefficient (F1-score for spatial segmentation)."""
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2.0 * inter / total)


def success_rate(iou_list: Sequence[float], threshold: float = 0.80) -> float:
    """Calculate proportion of segmentations achieving IoU >= threshold."""
    if not iou_list:
        return 0.0
    successful = sum(1 for val in iou_list if val >= threshold)
    return float(successful / len(iou_list))


def confusion(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Calculate confusion matrix for ripeness classes."""
    if labels is None:
        labels = CANONICAL_CLASSES
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def evaluate_folder(
    raw_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    pipeline: Optional[BananaRipenessPipeline] = None,
    cfg: Optional[Config] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate pipeline on all raw images with corresponding ground truth masks.

    Returns:
    - per_image_df: DataFrame with individual results (filename, IoU, Dice, true_label, pred_label, match)
    - summary: Aggregated metrics (mean_iou, success_rate, accuracy, per_condition_stats)
    """
    if pipeline is None:
        pipeline = BananaRipenessPipeline(cfg=cfg or load_config())

    raw_path = Path(raw_dir).resolve()
    gt_path = Path(gt_dir).resolve()

    raw_images = list_images(raw_path)
    if not raw_images:
        raise FileNotFoundError(f"No images found in raw directory: {raw_path}")

    records = []
    y_true = []
    y_pred = []
    iou_scores = []
    dice_scores = []

    thresh = pipeline.cfg.evaluation.success_iou_threshold

    for img_file in raw_images:
        meta = parse_filename_metadata(img_file)
        stem = img_file.stem

        # Locate corresponding ground truth file
        gt_candidates = [
            gt_path / f"{stem}.png",
            gt_path / f"{stem}.jpg",
            gt_path / f"{stem}_mask.png",
            gt_path / f"{stem}_gt.png",
        ]
        gt_file = next((c for c in gt_candidates if c.exists()), None)

        # Run pipeline
        res = pipeline.run_file(img_file)

        record: Dict[str, Any] = {
            "filename": img_file.name,
            "device": meta.get("device"),
            "specimen": meta.get("specimen"),
            "side": meta.get("side"),
            "true_label": meta.get("label"),
            "pred_label": res.label,
            "label_match": (meta.get("label") == res.label) if meta.get("label") else None,
            "area_ratio": res.features.get("area_ratio"),
            "ratio_green": res.features.get("ratio_green"),
            "ratio_yellow": res.features.get("ratio_yellow"),
            "ratio_brown": res.features.get("ratio_brown"),
        }

        if gt_file is not None:
            gt_img = read_image(gt_file)
            gt_gray = gt_img[:, :, 0] if len(gt_img.shape) == 3 else gt_img
            # Resize gt if necessary
            if gt_gray.shape != res.mask.shape:
                import cv2
                gt_gray = cv2.resize(gt_gray, (res.mask.shape[1], res.mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            score_iou = iou(res.mask, gt_gray)
            score_dice = dice(res.mask, gt_gray)
            record["iou"] = round(score_iou, 4)
            record["dice"] = round(score_dice, 4)
            record["iou_success"] = bool(score_iou >= thresh)
            iou_scores.append(score_iou)
            dice_scores.append(score_dice)
        else:
            record["iou"] = None
            record["dice"] = None
            record["iou_success"] = None

        if meta.get("label"):
            y_true.append(meta["label"])
            y_pred.append(res.label)

        records.append(record)

    df = pd.DataFrame(records)

    # Calculate overall summary metrics
    mean_iou = float(np.mean(iou_scores)) if iou_scores else None
    mean_dice = float(np.mean(dice_scores)) if dice_scores else None
    overall_success = success_rate(iou_scores, threshold=thresh) if iou_scores else None

    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else None
    conf_mat = confusion(y_true, y_pred, labels=CANONICAL_CLASSES) if y_true else None

    # Condition breakdown (Mentah, Matang, Terlalu Matang, Busuk)
    condition_stats = {}
    if "true_label" in df.columns:
        for cond, group in df.groupby("true_label"):
            g_iou = group["iou"].dropna().tolist()
            condition_stats[cond] = {
                "count": len(group),
                "accuracy": float(np.mean(group["true_label"] == group["pred_label"])),
                "mean_iou": float(np.mean(g_iou)) if g_iou else None,
                "success_rate": success_rate(g_iou, threshold=thresh) if g_iou else None,
            }

    summary = {
        "total_images": len(df),
        "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
        "mean_dice": round(mean_dice, 4) if mean_dice is not None else None,
        "success_rate": round(overall_success * 100.0, 2) if overall_success is not None else None,
        "accuracy": round(accuracy * 100.0, 2) if accuracy is not None else None,
        "confusion_matrix": conf_mat.tolist() if conf_mat is not None else None,
        "condition_stats": condition_stats,
    }

    return df, summary
