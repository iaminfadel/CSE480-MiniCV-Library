"""Unit tests for shawwaf.features — Feature extractors."""

import numpy as np
import pytest

from shawwaf.features import (
    color_histogram_descriptor,
    pixel_statistics_descriptor,
    hog_descriptor,
    edge_orientation_histogram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_image():
    return np.random.randint(0, 256, (32, 32), dtype=np.uint8)


@pytest.fixture
def rgb_image():
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# color_histogram_descriptor
# ---------------------------------------------------------------------------

class TestColorHistogramDescriptor:
    def test_grayscale(self, gray_image):
        feat = color_histogram_descriptor(gray_image, bins=16)
        assert feat.ndim == 1
        assert feat.shape[0] == 16
        np.testing.assert_allclose(feat.sum(), 1.0, atol=1e-10)

    def test_rgb(self, rgb_image):
        feat = color_histogram_descriptor(rgb_image, bins=16)
        assert feat.shape[0] == 48  # 3 channels * 16 bins
        np.testing.assert_allclose(feat.sum(), 1.0, atol=1e-10)

    def test_invalid_bins(self, gray_image):
        with pytest.raises(ValueError):
            color_histogram_descriptor(gray_image, bins=0)


# ---------------------------------------------------------------------------
# pixel_statistics_descriptor
# ---------------------------------------------------------------------------

class TestPixelStatisticsDescriptor:
    def test_grayscale(self, gray_image):
        feat = pixel_statistics_descriptor(gray_image)
        assert feat.shape == (6,)

    def test_rgb(self, rgb_image):
        feat = pixel_statistics_descriptor(rgb_image)
        assert feat.shape == (18,)

    def test_type_error(self):
        with pytest.raises(TypeError):
            pixel_statistics_descriptor([1, 2, 3])


# ---------------------------------------------------------------------------
# hog_descriptor
# ---------------------------------------------------------------------------

class TestHogDescriptor:
    def test_grayscale(self, gray_image):
        feat = hog_descriptor(gray_image, cell_size=8, bins=9)
        n_cells = (32 // 8) * (32 // 8)  # 4 * 4 = 16 cells
        assert feat.shape == (n_cells * 9,)

    def test_rgb(self, rgb_image):
        feat = hog_descriptor(rgb_image, cell_size=8, bins=9)
        assert feat.ndim == 1

    def test_invalid_cell_size(self, gray_image):
        with pytest.raises(ValueError):
            hog_descriptor(gray_image, cell_size=0)


# ---------------------------------------------------------------------------
# edge_orientation_histogram
# ---------------------------------------------------------------------------

class TestEdgeOrientationHistogram:
    def test_grayscale(self, gray_image):
        feat = edge_orientation_histogram(gray_image, bins=36)
        assert feat.shape == (36,)
        np.testing.assert_allclose(feat.sum(), 1.0, atol=1e-10)

    def test_rgb(self, rgb_image):
        feat = edge_orientation_histogram(rgb_image, bins=18)
        assert feat.shape == (18,)

    def test_invalid_bins(self, gray_image):
        with pytest.raises(ValueError):
            edge_orientation_histogram(gray_image, bins=0)
