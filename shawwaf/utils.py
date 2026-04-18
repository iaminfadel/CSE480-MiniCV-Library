"""
Shared utilities for the shawwaf library.

Provides normalization, clipping, padding, and validation helpers
used by all other modules. All shared logic is centralized here to
avoid duplication across the codebase.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_image(image):
    """Validate that the input is a NumPy array representing an image.

    Parameters
    ----------
    image : numpy.ndarray
        The input to validate.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``image`` has fewer than 2 dimensions or more than 3.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected a NumPy ndarray, got {type(image).__name__}."
        )
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(
            f"Image must be 2D (grayscale) or 3D (RGB), got {image.ndim}D."
        )


def validate_grayscale(image):
    """Validate that the input is a 2D grayscale NumPy array.

    Parameters
    ----------
    image : numpy.ndarray
        The input to validate.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``image`` is not 2-dimensional.
    """
    validate_image(image)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}."
        )


def validate_kernel(kernel):
    """Validate that a convolution kernel is well-formed.

    The kernel must be a 2D NumPy array with odd dimensions, non-empty,
    and numeric.

    Parameters
    ----------
    kernel : numpy.ndarray
        The convolution kernel to validate.

    Raises
    ------
    TypeError
        If ``kernel`` is not a NumPy ndarray or is non-numeric.
    ValueError
        If ``kernel`` is empty or has even dimensions.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError(
            f"Kernel must be a NumPy ndarray, got {type(kernel).__name__}."
        )
    if kernel.size == 0:
        raise ValueError("Kernel must not be empty.")
    if kernel.ndim != 2:
        raise ValueError(
            f"Kernel must be 2D, got {kernel.ndim}D."
        )
    if not np.issubdtype(kernel.dtype, np.number):
        raise TypeError(
            f"Kernel must be numeric, got dtype {kernel.dtype}."
        )
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(
            f"Kernel must have odd dimensions, got {kernel.shape}."
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(image, mode="minmax"):
    """Normalize pixel values using one of several modes.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or RGB).
    mode : str, optional
        Normalization mode. One of:

        - ``'minmax'`` — scale to [0.0, 1.0] (output dtype ``float64``).
        - ``'zscore'`` — zero mean, unit standard deviation (``float64``).
        - ``'uint8'`` — scale to [0, 255] (output dtype ``uint8``).

    Returns
    -------
    numpy.ndarray
        Normalized image.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``mode`` is not recognized or image is empty.
    """
    validate_image(image)
    img = image.astype(np.float64)

    if mode == "minmax":
        min_val, max_val = img.min(), img.max()
        if max_val - min_val == 0:
            return np.zeros_like(img, dtype=np.float64)
        return (img - min_val) / (max_val - min_val)

    elif mode == "zscore":
        std = img.std()
        if std == 0:
            return np.zeros_like(img, dtype=np.float64)
        return (img - img.mean()) / std

    elif mode == "uint8":
        min_val, max_val = img.min(), img.max()
        if max_val - min_val == 0:
            return np.zeros_like(img, dtype=np.uint8)
        scaled = (img - min_val) / (max_val - min_val) * 255.0
        return scaled.astype(np.uint8)

    else:
        raise ValueError(
            f"Unknown normalization mode '{mode}'. "
            "Supported modes: 'minmax', 'zscore', 'uint8'."
        )


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clip(image, min_val, max_val):
    """Clip pixel values to a defined range.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or RGB).
    min_val : int or float
        Lower bound of the clipping range.
    max_val : int or float
        Upper bound of the clipping range.

    Returns
    -------
    numpy.ndarray
        Clipped image with values in ``[min_val, max_val]``.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``min_val >= max_val``.
    """
    validate_image(image)
    if min_val >= max_val:
        raise ValueError(
            f"min_val must be less than max_val, got "
            f"min_val={min_val}, max_val={max_val}."
        )
    return np.clip(image, min_val, max_val)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad(image, pad_width, mode="zero"):
    """Pad a grayscale image using one of several boundary modes.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale input image with shape ``(H, W)``.
    pad_width : int
        Number of pixels to pad on each side.
    mode : str, optional
        Padding mode. One of:

        - ``'zero'`` — pad with constant 0.
        - ``'reflect'`` — mirror-reflect at boundaries.
        - ``'replicate'`` — repeat edge pixels.

    Returns
    -------
    numpy.ndarray
        Padded image with shape ``(H + 2*pad_width, W + 2*pad_width)``.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``pad_width`` is negative or ``mode`` is unknown.
    """
    validate_grayscale(image)
    if not isinstance(pad_width, (int, np.integer)) or pad_width < 0:
        raise ValueError(
            f"pad_width must be a non-negative integer, got {pad_width}."
        )

    pw = int(pad_width)
    if pw == 0:
        return image.copy()

    if mode == "zero":
        return np.pad(image, pw, mode="constant", constant_values=0)

    elif mode == "reflect":
        return np.pad(image, pw, mode="reflect")

    elif mode == "replicate":
        return np.pad(image, pw, mode="edge")

    else:
        raise ValueError(
            f"Unknown padding mode '{mode}'. "
            "Supported modes: 'zero', 'reflect', 'replicate'."
        )
