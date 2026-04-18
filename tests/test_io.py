"""Unit tests for shawwaf.io — Image I/O and color conversion."""

import os
import tempfile

import numpy as np
import pytest

from shawwaf.io import read_image, save_image, to_grayscale, to_rgb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_image():
    """A small 4×4 RGB uint8 image."""
    return np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)


@pytest.fixture
def gray_image():
    """A small 4×4 grayscale uint8 image."""
    return np.random.randint(0, 256, (4, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# read_image
# ---------------------------------------------------------------------------

class TestReadImage:
    def test_read_png(self, rgb_image, tmp_path):
        path = str(tmp_path / "test.png")
        save_image(rgb_image, path)
        loaded = read_image(path)
        assert loaded.shape[:2] == rgb_image.shape[:2]
        assert loaded.dtype == np.uint8

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_image("nonexistent.png")

    def test_unsupported_format(self, tmp_path):
        path = str(tmp_path / "test.bmp")
        with open(path, "w") as f:
            f.write("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            read_image(path)


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_save_rgb(self, rgb_image, tmp_path):
        path = str(tmp_path / "out.png")
        save_image(rgb_image, path)
        assert os.path.isfile(path)

    def test_save_gray(self, gray_image, tmp_path):
        path = str(tmp_path / "out.png")
        save_image(gray_image, path)
        assert os.path.isfile(path)

    def test_save_invalid_type(self, tmp_path):
        with pytest.raises(TypeError):
            save_image("not_array", str(tmp_path / "out.png"))

    def test_save_unsupported_format(self, rgb_image, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            save_image(rgb_image, str(tmp_path / "out.bmp"))

    def test_save_wrong_channels(self, tmp_path):
        img = np.zeros((4, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            save_image(img, str(tmp_path / "out.png"))


# ---------------------------------------------------------------------------
# to_grayscale
# ---------------------------------------------------------------------------

class TestToGrayscale:
    def test_basic(self, rgb_image):
        gray = to_grayscale(rgb_image)
        assert gray.shape == rgb_image.shape[:2]
        assert gray.dtype == np.uint8

    def test_invalid_input(self, gray_image):
        with pytest.raises(ValueError):
            to_grayscale(gray_image)

    def test_type_error(self):
        with pytest.raises(TypeError):
            to_grayscale([1, 2, 3])


# ---------------------------------------------------------------------------
# to_rgb
# ---------------------------------------------------------------------------

class TestToRgb:
    def test_basic(self, gray_image):
        rgb = to_rgb(gray_image)
        assert rgb.shape == (*gray_image.shape, 3)
        assert np.all(rgb[:, :, 0] == gray_image)
        assert np.all(rgb[:, :, 1] == gray_image)
        assert np.all(rgb[:, :, 2] == gray_image)

    def test_invalid_rgb_input(self, rgb_image):
        with pytest.raises(ValueError):
            to_rgb(rgb_image)
