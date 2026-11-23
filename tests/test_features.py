"""Unit tests for feature extraction module."""

import numpy as np
import pytest
from banana_ripeness.config import Config
from banana_ripeness.features import extract_features


def test_extract_features_empty_mask():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    feats = extract_features(img, mask)
    assert feats["area_pixels"] == 0
    assert feats["area_ratio"] == 0.0
    assert feats["ratio_green"] == 0.0


def test_extract_features_pure_green():
    # Pure green image (BGR: B=0, G=255, R=0 -> HSV: H=60, S=255, V=255)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 255, 0]
    mask = np.full((100, 100), 255, dtype=np.uint8)

    feats = extract_features(img, mask)
    assert feats["area_pixels"] == 10000
    assert feats["ratio_green"] > 0.90
    assert feats["ratio_yellow"] == 0.0
    assert feats["ratio_brown"] == 0.0


def test_extract_features_pure_yellow():
    # Pure yellow image (BGR: B=0, G=255, R=255 -> HSV: H=30, S=255, V=255)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 255, 255]
    mask = np.full((100, 100), 255, dtype=np.uint8)

    feats = extract_features(img, mask)
    assert feats["ratio_yellow"] > 0.90
    assert feats["ratio_green"] == 0.0
    assert feats["ratio_brown"] == 0.0


def test_extract_features_dark_rotten_spots():
    # Mostly dark decay pixels (low value V < 60)
    img = np.full((100, 100, 3), 30, dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)

    feats = extract_features(img, mask)
    assert feats["ratio_brown"] > 0.90
