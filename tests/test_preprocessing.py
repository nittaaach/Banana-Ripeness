"""Unit tests for preprocessing module."""

import numpy as np
import pytest
from banana_ripeness.config import PreprocessingConfig
from banana_ripeness.preprocessing import contrast_stretch, gaussian_blur, preprocess, resize


def test_resize_dimensions():
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    resized = resize(img, size=(600, 600))
    assert resized.shape == (600, 600, 3)


def test_gaussian_blur_variance_reduction():
    # Create an image with high-frequency noise
    np.random.seed(42)
    noise_img = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
    var_before = np.var(noise_img)

    blurred = gaussian_blur(noise_img, ksize=(7, 7), sigma=1.5)
    var_after = np.var(blurred)

    assert var_after < var_before


def test_contrast_stretch_dynamic_range():
    # Low contrast image in middle intensity range [100, 150]
    low_contrast = np.full((100, 100, 3), 120, dtype=np.uint8)
    low_contrast[20:40, 20:40] = 105
    low_contrast[60:80, 60:80] = 145

    stretched = contrast_stretch(low_contrast, p_low=2, p_high=98)
    assert stretched.shape == low_contrast.shape
    assert stretched.dtype == np.uint8
    # Min value should decrease and max value should increase
    assert np.min(stretched) <= np.min(low_contrast)
    assert np.max(stretched) >= np.max(low_contrast)


def test_preprocess_pipeline():
    cfg = PreprocessingConfig(gaussian_kernel=(5, 5), contrast_stretch=True)
    img = np.random.randint(50, 200, size=(300, 300, 3), dtype=np.uint8)
    out = preprocess(img, cfg)
    assert out.shape == (600, 600, 3)
