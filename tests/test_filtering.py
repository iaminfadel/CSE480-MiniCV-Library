"""Unit tests for shawwaf.filtering — Convolution and spatial filters."""

import numpy as np
import pytest

from shawwaf.filtering import (
    convolve2d,
    spatial_filter,
    mean_filter,
    gaussian_kernel,
    gaussian_filter,
    median_filter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_image():
    return np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
        [130, 140, 150, 160],
    ], dtype=np.uint8)


@pytest.fixture
def rgb_image():
    return np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)


@pytest.fixture
def identity_kernel():
    """3×3 identity (impulse) kernel."""
    k = np.zeros((3, 3), dtype=np.float64)
    k[1, 1] = 1.0
    return k


# ---------------------------------------------------------------------------
# convolve2d
# ---------------------------------------------------------------------------

class TestConvolve2d:
    def test_identity_kernel(self, gray_image, identity_kernel):
        result = convolve2d(gray_image.astype(np.float64), identity_kernel)
        np.testing.assert_allclose(result, gray_image, atol=1e-10)

    def test_output_shape(self, gray_image, identity_kernel):
        result = convolve2d(gray_image, identity_kernel)
        assert result.shape == gray_image.shape

    def test_non_array_image(self, identity_kernel):
        with pytest.raises(TypeError):
            convolve2d([[1, 2], [3, 4]], identity_kernel)

    def test_even_kernel(self, gray_image):
        with pytest.raises(ValueError, match="odd"):
            convolve2d(gray_image, np.ones((2, 2)))

    def test_empty_kernel(self, gray_image):
        with pytest.raises(ValueError, match="empty"):
            convolve2d(gray_image, np.array([]).reshape(0, 0))

    def test_non_numeric_kernel(self, gray_image):
        with pytest.raises(TypeError, match="numeric"):
            convolve2d(gray_image, np.array([["a", "b", "c"]] * 3))

    def test_rgb_image_raises(self, rgb_image, identity_kernel):
        with pytest.raises(ValueError):
            convolve2d(rgb_image, identity_kernel)


# ---------------------------------------------------------------------------
# spatial_filter
# ---------------------------------------------------------------------------

class TestSpatialFilter:
    def test_grayscale(self, gray_image, identity_kernel):
        result = spatial_filter(gray_image, identity_kernel)
        assert result.shape == gray_image.shape

    def test_rgb(self, rgb_image, identity_kernel):
        result = spatial_filter(rgb_image, identity_kernel)
        assert result.shape == rgb_image.shape


# ---------------------------------------------------------------------------
# mean_filter
# ---------------------------------------------------------------------------

class TestMeanFilter:
    def test_basic(self, gray_image):
        result = mean_filter(gray_image, 3)
        assert result.shape == gray_image.shape

    def test_even_size_raises(self, gray_image):
        with pytest.raises(ValueError, match="odd"):
            mean_filter(gray_image, 4)

    def test_negative_size_raises(self, gray_image):
        with pytest.raises(ValueError, match="positive"):
            mean_filter(gray_image, -1)


# ---------------------------------------------------------------------------
# gaussian_kernel
# ---------------------------------------------------------------------------

class TestGaussianKernel:
    def test_shape(self):
        k = gaussian_kernel(5, 1.0)
        assert k.shape == (5, 5)

    def test_normalized(self):
        k = gaussian_kernel(5, 1.0)
        np.testing.assert_allclose(k.sum(), 1.0, atol=1e-10)

    def test_even_size_raises(self):
        with pytest.raises(ValueError):
            gaussian_kernel(4, 1.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_kernel(5, -1.0)


# ---------------------------------------------------------------------------
# gaussian_filter
# ---------------------------------------------------------------------------

class TestGaussianFilter:
    def test_basic(self, gray_image):
        result = gaussian_filter(gray_image, 3, 1.0)
        assert result.shape == gray_image.shape


# ---------------------------------------------------------------------------
# median_filter
# ---------------------------------------------------------------------------

class TestMedianFilter:
    def test_basic(self, gray_image):
        result = median_filter(gray_image, 3)
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_rgb(self, rgb_image):
        result = median_filter(rgb_image, 3)
        assert result.shape == rgb_image.shape

    def test_even_size_raises(self, gray_image):
        with pytest.raises(ValueError, match="odd"):
            median_filter(gray_image, 4)
