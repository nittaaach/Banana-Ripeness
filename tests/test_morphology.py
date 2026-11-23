"""Unit tests for morphology module."""

import numpy as np
import pytest
from banana_ripeness.config import MorphologyConfig
from banana_ripeness.morphology import closing, fill_holes, largest_component, make_kernel, opening, refine


def test_opening_removes_small_isolated_dots():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Big object in center
    mask[20:80, 20:80] = 255
    # Isolated 1-pixel noise
    mask[5, 5] = 255
    mask[10, 90] = 255

    kernel = make_kernel("ellipse", (5, 5))
    opened = opening(mask, kernel, iters=1)

    assert opened[5, 5] == 0
    assert opened[10, 90] == 0
    assert opened[50, 50] == 255


def test_closing_fills_internal_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    # Small 3x3 hole in center
    mask[49:52, 49:52] = 0

    kernel = make_kernel("ellipse", (7, 7))
    closed = closing(mask, kernel, iters=2)

    assert closed[50, 50] == 255


def test_largest_component_keeps_single_dominant_object():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Large object (area ~ 2500)
    mask[20:70, 20:70] = 255
    # Small object (area ~ 100)
    mask[80:90, 80:90] = 255

    single = largest_component(mask)

    assert np.all(single[20:70, 20:70] == 255)
    assert np.all(single[80:90, 80:90] == 0)


def test_fill_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    # Deep hole inside
    mask[40:60, 40:60] = 0

    filled = fill_holes(mask)
    assert filled[50, 50] == 255
    assert filled[5, 5] == 0
