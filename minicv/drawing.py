"""
Drawing primitives and text placement for the MiniCV library.

Provides functions to draw points, lines, rectangles, polygons,
and text onto images. Uses Bresenham's line algorithm for rasterization
and Matplotlib font rendering for text.
"""

import numpy as np

from minicv.utils import validate_image


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _resolve_color(image, color):
    """Resolve a color value based on image dimensionality.

    Returns a scalar for grayscale, or a 3-element array for RGB.
    """
    if image.ndim == 2:
        if isinstance(color, (list, tuple, np.ndarray)):
            return int(np.mean(color))
        return int(color)
    else:
        if isinstance(color, (int, float, np.integer, np.floating)):
            return np.array([color, color, color], dtype=image.dtype)
        return np.array(color, dtype=image.dtype)[:3]


def _set_pixel(image, x, y, color):
    """Set a single pixel, silently clipping out-of-bounds coordinates."""
    h, w = image.shape[:2]
    if 0 <= y < h and 0 <= x < w:
        image[y, x] = color


# ---------------------------------------------------------------------------
# Draw Point
# ---------------------------------------------------------------------------

def draw_point(image, x, y, color=255, thickness=1):
    """Draw a colored point at a given pixel coordinate.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image. Modified in-place.
    x : int
        Horizontal coordinate (column).
    y : int
        Vertical coordinate (row).
    color : int, float, or tuple, optional
        Grayscale scalar or RGB tuple (default 255).
    thickness : int, optional
        Radius of the point in pixels (default 1).

    Returns
    -------
    numpy.ndarray
        The modified image (same object, modified in-place).

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.

    Notes
    -----
    Coordinates outside the canvas are clipped silently.
    """
    validate_image(image)
    c = _resolve_color(image, color)
    half = thickness // 2

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            _set_pixel(image, x + dx, y + dy, c)

    return image


# ---------------------------------------------------------------------------
# Draw Line (Bresenham's Algorithm)
# ---------------------------------------------------------------------------

def draw_line(image, x0, y0, x1, y1, color=255, thickness=1):
    """Draw a straight line between two points using Bresenham's algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image. Modified in-place.
    x0, y0 : int
        Start point coordinates.
    x1, y1 : int
        End point coordinates.
    color : int, float, or tuple, optional
        Line color (default 255).
    thickness : int, optional
        Line thickness in pixels (default 1).

    Returns
    -------
    numpy.ndarray
        The modified image.

    Notes
    -----
    Pixels outside the canvas boundary are clipped silently.
    Uses Bresenham's line rasterization algorithm for efficient,
    integer-only computation.
    """
    validate_image(image)
    c = _resolve_color(image, color)
    points = _bresenham(x0, y0, x1, y1)

    half = thickness // 2
    for px, py in points:
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                _set_pixel(image, px + dx, py + dy, c)

    return image


def _bresenham(x0, y0, x1, y1):
    """Generate pixel coordinates along a line using Bresenham's algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


# ---------------------------------------------------------------------------
# Draw Rectangle
# ---------------------------------------------------------------------------

def draw_rectangle(image, x, y, w, h, color=255, thickness=1, filled=False):
    """Draw a rectangle defined by a corner and size.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image. Modified in-place.
    x, y : int
        Top-left corner coordinates.
    w, h : int
        Width and height of the rectangle.
    color : int, float, or tuple, optional
        Rectangle color (default 255).
    thickness : int, optional
        Outline thickness in pixels (default 1). Ignored if ``filled=True``.
    filled : bool, optional
        If True, fill the interior with ``color``.

    Returns
    -------
    numpy.ndarray
        The modified image.

    Notes
    -----
    Partial rectangles are drawn where visible; boundary clipping is applied.
    """
    validate_image(image)
    c = _resolve_color(image, color)
    img_h, img_w = image.shape[:2]

    if filled:
        y0 = max(0, y)
        y1 = min(img_h, y + h)
        x0 = max(0, x)
        x1 = min(img_w, x + w)
        if y1 > y0 and x1 > x0:
            image[y0:y1, x0:x1] = c
    else:
        # Draw four edges
        draw_line(image, x, y, x + w - 1, y, color, thickness)           # top
        draw_line(image, x, y + h - 1, x + w - 1, y + h - 1, color, thickness)  # bottom
        draw_line(image, x, y, x, y + h - 1, color, thickness)           # left
        draw_line(image, x + w - 1, y, x + w - 1, y + h - 1, color, thickness)  # right

    return image


# ---------------------------------------------------------------------------
# Draw Polygon
# ---------------------------------------------------------------------------

def draw_polygon(image, points, color=255, thickness=1, filled=False):
    """Draw a polygon from a list of vertices.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image. Modified in-place.
    points : list of tuple
        List of ``(x, y)`` vertex coordinates. Must have >= 3 points.
    color : int, float, or tuple, optional
        Polygon color (default 255).
    thickness : int, optional
        Outline thickness (default 1).
    filled : bool, optional
        If True, fill the polygon interior using a scanline approach.
        This is an optional but implemented feature.

    Returns
    -------
    numpy.ndarray
        The modified image.

    Raises
    ------
    ValueError
        If ``points`` has fewer than 3 vertices.
    """
    validate_image(image)
    if len(points) < 3:
        raise ValueError(
            f"Polygon requires at least 3 points, got {len(points)}."
        )

    c = _resolve_color(image, color)

    if filled:
        _fill_polygon(image, points, c)
    else:
        # Draw edges connecting consecutive vertices
        n = len(points)
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            draw_line(image, x0, y0, x1, y1, color, thickness)

    return image


def _fill_polygon(image, points, color):
    """Fill a polygon using scanline rasterization."""
    pts = np.array(points)
    min_y = max(0, int(pts[:, 1].min()))
    max_y = min(image.shape[0] - 1, int(pts[:, 1].max()))

    n = len(points)
    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            if y0 == y1:
                continue
            if min(y0, y1) <= y < max(y0, y1):
                x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                intersections.append(x_int)

        intersections.sort()
        for j in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(np.ceil(intersections[j])))
            x_end = min(image.shape[1] - 1, int(np.floor(intersections[j + 1])))
            if x_end >= x_start:
                image[y, x_start:x_end + 1] = color


# ---------------------------------------------------------------------------
# Draw Text
# ---------------------------------------------------------------------------

def draw_text(image, text, x, y, font_scale=1.0, color=255):
    """Render a text string onto an image.

    Uses Matplotlib's font rendering engine to rasterize text and
    composites the result onto the target image. Text exceeding canvas
    boundaries is clipped.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image. Modified in-place.
    text : str
        The text string to render.
    x, y : int
        Position of the text baseline origin (column, row).
    font_scale : float, optional
        Scale factor controlling character size (default 1.0).
    color : int, float, or tuple, optional
        Text color (default 255).

    Returns
    -------
    numpy.ndarray
        The modified image.

    Notes
    -----
    Font rendering is handled by Matplotlib's Agg backend. The text is
    rendered into an off-screen buffer and then blended onto the image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    validate_image(image)
    c = _resolve_color(image, color)

    img_h, img_w = image.shape[:2]
    dpi = 100
    fig_w = img_w / dpi
    fig_h = img_h / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)  # origin at top-left
    ax.axis("off")
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    fontsize = 12 * font_scale

    if image.ndim == 3:
        text_color = tuple(c_val / 255.0 for c_val in c[:3])
    else:
        text_color = (c / 255.0,) * 3

    ax.text(x, y, text, fontsize=fontsize, color=text_color,
            ha="left", va="top", family="monospace")

    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(img_h, img_w, 4)

    # ARGB -> extract alpha as mask
    alpha = buf[:, :, 0].astype(np.float64) / 255.0
    rgb_buf = buf[:, :, 1:4]  # RGB channels

    if image.ndim == 2:
        gray_buf = (0.2989 * rgb_buf[:, :, 0] +
                    0.5870 * rgb_buf[:, :, 1] +
                    0.1140 * rgb_buf[:, :, 2])
        mask = alpha > 0.1
        image[mask] = gray_buf[mask].astype(image.dtype)
    else:
        mask = alpha > 0.1
        for ch in range(3):
            channel = image[:, :, ch].astype(np.float64)
            channel[mask] = rgb_buf[:, :, ch][mask].astype(np.float64)
            image[:, :, ch] = channel.astype(image.dtype)

    plt.close(fig)
    return image
