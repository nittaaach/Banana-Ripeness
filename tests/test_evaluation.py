"""Unit tests for evaluation module."""

import numpy as np
import pytest
from banana_ripeness.evaluation import confusion, dice, iou, success_rate


def test_iou_and_dice_identical_masks():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255

    assert iou(mask, mask) == 1.0
    assert dice(mask, mask) == 1.0


def test_iou_and_dice_disjoint_masks():
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m2 = np.zeros((100, 100), dtype=np.uint8)

    m1[10:30, 10:30] = 255
    m2[60:80, 60:80] = 255

    assert iou(m1, m2) == 0.0
    assert dice(m1, m2) == 0.0


def test_success_rate():
    ious = [0.85, 0.92, 0.75, 0.81]
    # 3 out of 4 are >= 0.80 -> 75%
    assert success_rate(ious, threshold=0.80) == 0.75


def test_confusion_matrix():
    y_true = ["Mentah", "Matang", "Busuk"]
    y_pred = ["Mentah", "Matang", "Matang"]
    labels = ["Mentah", "Matang", "Terlalu Matang", "Busuk"]
    cm = confusion(y_true, y_pred, labels=labels)
    assert cm.shape == (4, 4)
    # Mentah correctly predicted as Mentah
    assert cm[0, 0] == 1
    # Busuk mispredicted as Matang
    assert cm[3, 1] == 1
