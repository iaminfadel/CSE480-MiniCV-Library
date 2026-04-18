"""Unit tests for shawwaf.drawing — Drawing primitives and text."""

import numpy as np
import pytest

from shawwaf.drawing import (
    draw_point,
    draw_line,
    draw_rectangle,
    draw_polygon,
    draw_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_canvas():
    return np.zeros((50, 50), dtype=np.uint8)


@pytest.fixture
def rgb_canvas():
    return np.zeros((50, 50, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# draw_point
# ---------------------------------------------------------------------------

class TestDrawPoint:
    def test_grayscale(self, gray_canvas):
        draw_point(gray_canvas, 25, 25, color=200)
        assert gray_canvas[25, 25] == 200

    def test_rgb(self, rgb_canvas):
        draw_point(rgb_canvas, 10, 10, color=(255, 0, 0))
        np.testing.assert_array_equal(rgb_canvas[10, 10], [255, 0, 0])

    def test_out_of_bounds(self, gray_canvas):
        # Should not raise
        draw_point(gray_canvas, 100, 100, color=255)
        assert gray_canvas.max() == 0

    def test_thickness(self, gray_canvas):
        draw_point(gray_canvas, 25, 25, color=255, thickness=3)
        assert gray_canvas[24, 24] == 255
        assert gray_canvas[26, 26] == 255


# ---------------------------------------------------------------------------
# draw_line
# ---------------------------------------------------------------------------

class TestDrawLine:
    def test_horizontal(self, gray_canvas):
        draw_line(gray_canvas, 0, 25, 49, 25, color=255)
        assert np.all(gray_canvas[25, :] == 255)

    def test_vertical(self, gray_canvas):
        draw_line(gray_canvas, 25, 0, 25, 49, color=255)
        assert np.all(gray_canvas[:, 25] == 255)

    def test_diagonal(self, gray_canvas):
        draw_line(gray_canvas, 0, 0, 49, 49, color=128)
        assert gray_canvas[0, 0] == 128
        assert gray_canvas[49, 49] == 128

    def test_rgb(self, rgb_canvas):
        draw_line(rgb_canvas, 0, 0, 49, 0, color=(0, 255, 0))
        np.testing.assert_array_equal(rgb_canvas[0, 0], [0, 255, 0])


# ---------------------------------------------------------------------------
# draw_rectangle
# ---------------------------------------------------------------------------

class TestDrawRectangle:
    def test_outline(self, gray_canvas):
        draw_rectangle(gray_canvas, 10, 10, 20, 15, color=200)
        assert gray_canvas[10, 10] == 200  # corner

    def test_filled(self, gray_canvas):
        draw_rectangle(gray_canvas, 5, 5, 10, 10, color=128, filled=True)
        assert np.all(gray_canvas[5:15, 5:15] == 128)

    def test_partial_clip(self, gray_canvas):
        # Rectangle partially outside canvas
        draw_rectangle(gray_canvas, 45, 45, 20, 20, color=100, filled=True)
        assert gray_canvas[45, 45] == 100

    def test_rgb_filled(self, rgb_canvas):
        draw_rectangle(rgb_canvas, 0, 0, 10, 10, color=(255, 128, 0), filled=True)
        np.testing.assert_array_equal(rgb_canvas[0, 0], [255, 128, 0])


# ---------------------------------------------------------------------------
# draw_polygon
# ---------------------------------------------------------------------------

class TestDrawPolygon:
    def test_triangle_outline(self, gray_canvas):
        pts = [(10, 10), (40, 10), (25, 40)]
        draw_polygon(gray_canvas, pts, color=255)
        assert gray_canvas[10, 10] == 255

    def test_too_few_points(self, gray_canvas):
        with pytest.raises(ValueError, match="3 points"):
            draw_polygon(gray_canvas, [(0, 0), (1, 1)], color=255)

    def test_filled_polygon(self, gray_canvas):
        pts = [(10, 10), (40, 10), (40, 40), (10, 40)]
        draw_polygon(gray_canvas, pts, color=200, filled=True)
        # Center should be filled
        assert gray_canvas[25, 25] == 200


# ---------------------------------------------------------------------------
# draw_text
# ---------------------------------------------------------------------------

class TestDrawText:
    def test_basic(self, gray_canvas):
        # Just verify it doesn't crash and returns the image
        result = draw_text(gray_canvas, "Hello", 5, 5)
        assert result.shape == gray_canvas.shape

    def test_rgb(self, rgb_canvas):
        result = draw_text(rgb_canvas, "Test", 5, 5, color=(255, 0, 0))
        assert result.shape == rgb_canvas.shape
