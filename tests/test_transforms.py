"""Unit tests for shawwaf.transforms — Resize, rotate, translate."""

import numpy as np
import pytest

from shawwaf.transforms import resize, rotate, translate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_image():
    return np.arange(64, dtype=np.uint8).reshape(8, 8)


@pytest.fixture
def rgb_image():
    return np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------

class TestResize:
    def test_upscale_nearest(self, gray_image):
        result = resize(gray_image, (16, 16), method="nearest")
        assert result.shape == (16, 16)
        assert result.dtype == gray_image.dtype

    def test_downscale_bilinear(self, gray_image):
        result = resize(gray_image, (4, 4), method="bilinear")
        assert result.shape == (4, 4)

    def test_rgb_resize(self, rgb_image):
        result = resize(rgb_image, (16, 16))
        assert result.shape == (16, 16, 3)

    def test_non_positive_dim(self, gray_image):
        with pytest.raises(ValueError, match="positive"):
            resize(gray_image, (0, 10))

    def test_unknown_method(self, gray_image):
        with pytest.raises(ValueError, match="Unknown"):
            resize(gray_image, (4, 4), method="cubic")

    def test_same_size(self, gray_image):
        result = resize(gray_image, (8, 8))
        assert result.shape == gray_image.shape


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

class TestRotate:
    def test_360_rotation(self, gray_image):
        result = rotate(gray_image, 360, interpolation="nearest")
        np.testing.assert_array_equal(result, gray_image)

    def test_output_shape(self, gray_image):
        result = rotate(gray_image, 45)
        assert result.shape == gray_image.shape

    def test_rgb_rotate(self, rgb_image):
        result = rotate(rgb_image, 90)
        assert result.shape == rgb_image.shape

    def test_unknown_interpolation(self, gray_image):
        with pytest.raises(ValueError, match="Unknown"):
            rotate(gray_image, 45, interpolation="cubic")


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_zero_shift(self, gray_image):
        result = translate(gray_image, 0, 0)
        np.testing.assert_array_equal(result, gray_image)

    def test_positive_shift(self, gray_image):
        result = translate(gray_image, 2, 3)
        assert result.shape == gray_image.shape
        # Top-left should be zero-filled
        assert result[0, 0] == 0

    def test_negative_shift(self, gray_image):
        result = translate(gray_image, -2, -3)
        assert result.shape == gray_image.shape

    def test_rgb_translate(self, rgb_image):
        result = translate(rgb_image, 1, 1)
        assert result.shape == rgb_image.shape
