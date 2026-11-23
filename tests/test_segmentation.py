"""Unit tests for segmentation module."""

import cv2
import numpy as np
import pytest
from banana_ripeness.config import Config
from banana_ripeness.segmentation import combine_masks, hsv_mask, otsu_mask, segment, to_hsv


def test_to_hsv_conversion():
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    bgr[:, :] = [0, 255, 255]  # Yellow in BGR (B=0, G=255, R=255)
    hsv = to_hsv(bgr)
    assert hsv.shape == (50, 50, 3)
    # Yellow Hue in OpenCV is around 30
    assert 25 <= hsv[0, 0, 0] <= 35


def test_hsv_mask_synthetic_object():
    # Black background with yellow square
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [0, 255, 255]  # Yellow
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = hsv_mask(hsv, lower=[20, 100, 100], upper=[40, 255, 255])
    assert mask[50, 50] == 255
    assert mask[10, 10] == 0


def test_otsu_mask_and_auto_invert():
    # Light background (240), dark object in center (30)
    channel = np.full((100, 100), 240, dtype=np.uint8)
    channel[30:70, 30:70] = 30

    mask = otsu_mask(channel, auto_invert=True)
    # Foreground object in center should be white (255), outer border black (0)
    assert mask[50, 50] == 255
    assert mask[5, 5] == 0


def test_combine_masks_modes():
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10), dtype=np.uint8)

    m1[0:5, 0:5] = 255
    m2[3:8, 3:8] = 255

    m_or = combine_masks(m1, m2, mode="or")
    m_and = combine_masks(m1, m2, mode="and")

    assert np.sum(m_or > 0) > np.sum(m_and > 0)
    assert m_and[4, 4] == 255
    assert m_and[1, 1] == 0
