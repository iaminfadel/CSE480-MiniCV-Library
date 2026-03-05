"""
Geometric transformations for the MiniCV library.

Provides resize, rotate, and translate operations with configurable
interpolation methods. All transforms use vectorized NumPy coordinate
mapping — no per-pixel Python loops.
"""

import numpy as np

from minicv.utils import validate_image


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def resize(image, target_size, method="bilinear"):
    """Resize an image to a target resolution.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    target_size : tuple of int
        ``(height, width)`` of the output image.
    method : str, optional
        Interpolation method. One of ``'nearest'`` or ``'bilinear'``.

    Returns
    -------
    numpy.ndarray
        Resized image with shape ``(target_h, target_w)`` or
        ``(target_h, target_w, 3)``, same dtype as input.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``target_size`` has non-positive dimensions or ``method``
        is unknown.
    """
    validate_image(image)
    target_h, target_w = target_size

    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"target_size must have positive dimensions, got ({target_h}, {target_w})."
        )

    if method not in ("nearest", "bilinear"):
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            "Supported: 'nearest', 'bilinear'."
        )

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(_resize_2d(image[:, :, c], target_h, target_w, method))
        return np.stack(channels, axis=-1)

    return _resize_2d(image, target_h, target_w, method)


def _resize_2d(image, target_h, target_w, method):
    """Resize a single 2D channel."""
    src_h, src_w = image.shape
    img = image.astype(np.float64)

    # Map target coordinates to source coordinates
    row_ratio = src_h / target_h
    col_ratio = src_w / target_w

    rows = np.arange(target_h) * row_ratio
    cols = np.arange(target_w) * col_ratio
    col_grid, row_grid = np.meshgrid(cols, rows)

    if method == "nearest":
        src_rows = np.clip(np.round(row_grid).astype(int), 0, src_h - 1)
        src_cols = np.clip(np.round(col_grid).astype(int), 0, src_w - 1)
        return img[src_rows, src_cols].astype(image.dtype)

    else:  # bilinear
        r0 = np.clip(np.floor(row_grid).astype(int), 0, src_h - 1)
        r1 = np.clip(r0 + 1, 0, src_h - 1)
        c0 = np.clip(np.floor(col_grid).astype(int), 0, src_w - 1)
        c1 = np.clip(c0 + 1, 0, src_w - 1)

        dr = row_grid - r0
        dc = col_grid - c0

        top = img[r0, c0] * (1 - dc) + img[r0, c1] * dc
        bot = img[r1, c0] * (1 - dc) + img[r1, c1] * dc
        result = top * (1 - dr) + bot * dr
        return result.astype(image.dtype)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotate(image, angle, interpolation="bilinear"):
    """Rotate an image about its center by an arbitrary angle.

    Out-of-bounds pixels are filled with 0.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    angle : float
        Rotation angle in degrees (counter-clockwise positive).
    interpolation : str, optional
        Interpolation method: ``'nearest'`` or ``'bilinear'``.

    Returns
    -------
    numpy.ndarray
        Rotated image with the same shape, dtype as the input.
        Out-of-bounds regions are filled with 0.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``interpolation`` is unknown.
    """
    validate_image(image)
    if interpolation not in ("nearest", "bilinear"):
        raise ValueError(
            f"Unknown interpolation '{interpolation}'. "
            "Supported: 'nearest', 'bilinear'."
        )

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(_rotate_2d(image[:, :, c], angle, interpolation))
        return np.stack(channels, axis=-1)

    return _rotate_2d(image, angle, interpolation)


def _rotate_2d(image, angle, interpolation):
    """Rotate a single 2D channel."""
    h, w = image.shape
    img = image.astype(np.float64)

    # Center of the image
    cy, cx = h / 2.0, w / 2.0

    # Rotation angle in radians (negative for inverse mapping)
    theta = -np.radians(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Create output coordinate grids
    rows = np.arange(h)
    cols = np.arange(w)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Map output coords to input coords (inverse mapping)
    src_x = cos_t * (col_grid - cx) - sin_t * (row_grid - cy) + cx
    src_y = sin_t * (col_grid - cx) + cos_t * (row_grid - cy) + cy

    output = np.zeros_like(img)

    if interpolation == "nearest":
        src_cols = np.round(src_x).astype(int)
        src_rows = np.round(src_y).astype(int)
        valid = (src_rows >= 0) & (src_rows < h) & (src_cols >= 0) & (src_cols < w)
        output[valid] = img[src_rows[valid], src_cols[valid]]

    else:  # bilinear
        r0 = np.floor(src_y).astype(int)
        c0 = np.floor(src_x).astype(int)
        r1 = r0 + 1
        c1 = c0 + 1

        valid = (r0 >= 0) & (r1 < h) & (c0 >= 0) & (c1 < w)

        dr = src_y - r0
        dc = src_x - c0

        r0c = np.clip(r0, 0, h - 1)
        r1c = np.clip(r1, 0, h - 1)
        c0c = np.clip(c0, 0, w - 1)
        c1c = np.clip(c1, 0, w - 1)

        top = img[r0c, c0c] * (1 - dc) + img[r0c, c1c] * dc
        bot = img[r1c, c0c] * (1 - dc) + img[r1c, c1c] * dc
        interp = top * (1 - dr) + bot * dr

        output[valid] = interp[valid]

    return output.astype(image.dtype)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(image, tx, ty):
    """Translate an image by a pixel offset.

    Vacated regions are filled with 0. Supports both positive and
    negative offsets.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    tx : int
        Horizontal offset (positive = shift right).
    ty : int
        Vertical offset (positive = shift down).

    Returns
    -------
    numpy.ndarray
        Translated image with the same shape and dtype as the input.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    """
    validate_image(image)

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(_translate_2d(image[:, :, c], tx, ty))
        return np.stack(channels, axis=-1)

    return _translate_2d(image, tx, ty)


def _translate_2d(image, tx, ty):
    """Translate a single 2D channel."""
    h, w = image.shape
    output = np.zeros_like(image)

    # Compute source and destination slices
    src_y_start = max(0, -ty)
    src_y_end = min(h, h - ty)
    src_x_start = max(0, -tx)
    src_x_end = min(w, w - tx)

    dst_y_start = max(0, ty)
    dst_y_end = min(h, h + ty)
    dst_x_start = max(0, tx)
    dst_x_end = min(w, w + tx)

    # Only copy if the region is valid
    if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
        output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            image[src_y_start:src_y_end, src_x_start:src_x_end]

    return output
