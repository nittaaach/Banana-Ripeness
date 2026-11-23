"""Unit tests for classifier module."""

import pytest
from banana_ripeness.classifier import classify
from banana_ripeness.config import ClassificationConfig, Config


def test_classify_rotten():
    cfg = ClassificationConfig(brown_ratio_rotten=0.40)
    feats = {"ratio_brown": 0.55, "ratio_green": 0.05, "ratio_yellow": 0.20}
    assert classify(feats, cfg) == "Busuk"


def test_classify_overripe():
    cfg = ClassificationConfig(brown_ratio_rotten=0.40, brown_ratio_overripe=0.15)
    feats = {"ratio_brown": 0.25, "ratio_green": 0.05, "ratio_yellow": 0.60}
    assert classify(feats, cfg) == "Terlalu Matang"


def test_classify_unripe():
    cfg = ClassificationConfig(brown_ratio_rotten=0.40, brown_ratio_overripe=0.15, green_ratio_unripe=0.35)
    feats = {"ratio_brown": 0.02, "ratio_green": 0.60, "ratio_yellow": 0.30}
    assert classify(feats, cfg) == "Mentah"


def test_classify_ripe():
    cfg = ClassificationConfig(brown_ratio_rotten=0.40, brown_ratio_overripe=0.15, green_ratio_unripe=0.35)
    feats = {"ratio_brown": 0.03, "ratio_green": 0.10, "ratio_yellow": 0.85}
    assert classify(feats, cfg) == "Matang"
