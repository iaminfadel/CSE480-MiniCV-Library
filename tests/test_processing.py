"""Unit tests for shawwaf.processing — Thresholding, Sobel, histogram, etc."""

import numpy as np
import pytest

from shawwaf.processing import (
    threshold,
    sobel,
    bit_plane_slice,
    histogram,
    equalize_histogram,
    laplacian_sharpen,
    gamma_correction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_image():
    return np.array([
        [0, 50, 100, 150],
        [200, 250, 25, 75],
        [125, 175, 225, 30],
        [60, 110, 160, 210],
    ], dtype=np.uint8)


@pytest.fixture
def rgb_image():
    return np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_global(self, gray_image):
        result = threshold(gray_image, method="global", thresh=128)
        assert result.shape == gray_image.shape
        assert set(np.unique(result)).issubset({0, 255})

    def test_otsu(self, gray_image):
        result = threshold(gray_image, method="otsu")
        assert set(np.unique(result)).issubset({0, 255})

    def test_adaptive_mean(self, gray_image):
        result = threshold(gray_image, method="adaptive", block_size=3, adaptive_method="mean")
        assert set(np.unique(result)).issubset({0, 255})

    def test_adaptive_gaussian(self, gray_image):
        result = threshold(gray_image, method="adaptive", block_size=3, adaptive_method="gaussian")
        assert set(np.unique(result)).issubset({0, 255})

    def test_global_missing_thresh(self, gray_image):
        with pytest.raises(ValueError, match="thresh"):
            threshold(gray_image, method="global")

    def test_adaptive_missing_block_size(self, gray_image):
        with pytest.raises(ValueError, match="block_size"):
            threshold(gray_image, method="adaptive")

    def test_unknown_method(self, gray_image):
        with pytest.raises(ValueError, match="Unknown"):
            threshold(gray_image, method="unknown")

    def test_rgb_raises(self, rgb_image):
        with pytest.raises(ValueError):
            threshold(rgb_image, method="global", thresh=128)


# ---------------------------------------------------------------------------
# sobel
# ---------------------------------------------------------------------------

class TestSobel:
    def test_grayscale(self, gray_image):
        gx, gy, mag, direction = sobel(gray_image)
        assert gx.shape == gray_image.shape
        assert gy.shape == gray_image.shape
        assert mag.shape == gray_image.shape
        assert direction.shape == gray_image.shape

    def test_rgb_auto_convert(self, rgb_image):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gx, gy, mag, d = sobel(rgb_image)
        assert gx.shape == rgb_image.shape[:2]


# ---------------------------------------------------------------------------
# bit_plane_slice
# ---------------------------------------------------------------------------

class TestBitPlaneSlice:
    def test_lsb(self, gray_image):
        result = bit_plane_slice(gray_image, 0)
        assert set(np.unique(result)).issubset({0, 1})

    def test_msb(self, gray_image):
        result = bit_plane_slice(gray_image, 7)
        assert set(np.unique(result)).issubset({0, 1})

    def test_invalid_plane(self, gray_image):
        with pytest.raises(ValueError, match="0, 7"):
            bit_plane_slice(gray_image, 8)

    def test_wrong_dtype(self):
        img = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(TypeError, match="uint8"):
            bit_plane_slice(img, 0)


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------

class TestHistogram:
    def test_shape(self, gray_image):
        h = histogram(gray_image)
        assert h.shape == (256,)

    def test_sum(self, gray_image):
        h = histogram(gray_image)
        assert h.sum() == gray_image.size


# ---------------------------------------------------------------------------
# equalize_histogram
# ---------------------------------------------------------------------------

class TestEqualizeHistogram:
    def test_output_dtype(self, gray_image):
        result = equalize_histogram(gray_image)
        assert result.dtype == np.uint8

    def test_output_shape(self, gray_image):
        result = equalize_histogram(gray_image)
        assert result.shape == gray_image.shape

    def test_float_dtype_raises(self):
        with pytest.raises(TypeError):
            equalize_histogram(np.zeros((4, 4), dtype=np.float64))


# ---------------------------------------------------------------------------
# laplacian_sharpen
# ---------------------------------------------------------------------------

class TestLaplacianSharpen:
    def test_basic(self, gray_image):
        result = laplacian_sharpen(gray_image)
        assert result.shape == gray_image.shape
        assert result.dtype == np.uint8

    def test_rgb(self, rgb_image):
        result = laplacian_sharpen(rgb_image)
        assert result.shape == rgb_image.shape

    def test_negative_strength(self, gray_image):
        with pytest.raises(ValueError):
            laplacian_sharpen(gray_image, strength=-1)


# ---------------------------------------------------------------------------
# gamma_correction
# ---------------------------------------------------------------------------

class TestGammaCorrection:
    def test_identity(self, gray_image):
        result = gamma_correction(gray_image, gamma=1.0)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, gray_image)

    def test_brighten(self, gray_image):
        result = gamma_correction(gray_image, gamma=0.5)
        # gamma < 1 brightens: most pixels should be >= original
        assert result.mean() >= gray_image.mean() - 1

    def test_negative_gamma_raises(self, gray_image):
        with pytest.raises(ValueError):
            gamma_correction(gray_image, gamma=-1)

    def test_zero_gamma_raises(self, gray_image):
        with pytest.raises(ValueError):
            gamma_correction(gray_image, gamma=0)
