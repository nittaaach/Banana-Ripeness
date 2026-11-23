"""Integration tests for full BananaRipenessPipeline."""

import numpy as np
import pytest
from banana_ripeness.config import Config
from banana_ripeness.pipeline import BananaRipenessPipeline


def test_pipeline_synthetic_yellow_banana():
    cfg = Config()
    pipeline = BananaRipenessPipeline(cfg=cfg)

    # Synthetic white background (240) with yellow central object (banana simulation)
    img = np.full((600, 600, 3), 245, dtype=np.uint8)
    # Draw yellow banana curved / centered block
    img[200:400, 150:450] = [20, 220, 240]  # BGR Yellow

    result = pipeline.run(img)

    assert result.label in ["Matang", "Mentah", "Terlalu Matang", "Busuk"]
    assert result.mask.shape == (600, 600)
    assert np.sum(result.mask > 0) > 0
    assert "preprocessed" in result.stages
    assert "overlay" in result.stages
    assert result.features["area_pixels"] > 0
