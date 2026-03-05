"""
Image processing techniques for the MiniCV library.

Provides thresholding (global, Otsu, adaptive), Sobel edge detection,
bit-plane slicing, histogram operations, Laplacian sharpening, and
gamma correction.
"""

import numpy as np

from minicv.utils import validate_image, validate_grayscale
from minicv.filtering import convolve2d, spatial_filter


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def threshold(image, method="global", **kwargs):
    """Binarize a grayscale image using a thresholding method.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image, dtype ``uint8``.
    method : str, optional
        Thresholding method:

        - ``'global'`` — requires ``thresh`` kwarg (int in [0, 255]).
        - ``'otsu'`` — automatic threshold via Otsu's method.
        - ``'adaptive'`` — local thresholding. Requires ``block_size``
          (positive odd int) and optional ``method`` key ``'mean'`` or
          ``'gaussian'``, plus optional ``C`` offset (default 0).

    Returns
    -------
    numpy.ndarray
        Binary image with shape ``(H, W)``, values 0 or 255, dtype ``uint8``.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``method`` is unknown or required kwargs are missing.
    """
    validate_grayscale(image)
    img = image.astype(np.float64)

    if method == "global":
        if "thresh" not in kwargs:
            raise ValueError("'global' method requires 'thresh' kwarg.")
        t = kwargs["thresh"]
        result = np.where(img >= t, 255, 0).astype(np.uint8)
        return result

    elif method == "otsu":
        t = _otsu_threshold(image)
        return np.where(img >= t, 255, 0).astype(np.uint8)

    elif method == "adaptive":
        if "block_size" not in kwargs:
            raise ValueError("'adaptive' method requires 'block_size' kwarg.")
        block_size = kwargs["block_size"]
        adaptive_method = kwargs.get("adaptive_method", "mean")
        c = kwargs.get("C", 0)
        return _adaptive_threshold(image, block_size, adaptive_method, c)

    else:
        raise ValueError(
            f"Unknown threshold method '{method}'. "
            "Supported: 'global', 'otsu', 'adaptive'."
        )


def _otsu_threshold(image):
    """Compute the optimal threshold using Otsu's method.

    Maximizes the inter-class variance across all possible thresholds
    for a uint8 image.
    """
    img = image.astype(np.uint8)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size

    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0.0
    best_thresh = 0
    best_var = 0.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    return best_thresh


def _adaptive_threshold(image, block_size, adaptive_method, c):
    """Apply adaptive (local) thresholding."""
    if not isinstance(block_size, (int, np.integer)) or block_size <= 0 or block_size % 2 == 0:
        raise ValueError(
            f"block_size must be a positive odd integer, got {block_size}."
        )

    img = image.astype(np.float64)

    if adaptive_method == "mean":
        kernel = np.ones((block_size, block_size), dtype=np.float64) / (block_size ** 2)
    elif adaptive_method == "gaussian":
        from minicv.filtering import gaussian_kernel
        kernel = gaussian_kernel(block_size, block_size / 6.0)
    else:
        raise ValueError(
            f"Unknown adaptive method '{adaptive_method}'. "
            "Supported: 'mean', 'gaussian'."
        )

    local_mean = convolve2d(img, kernel, pad_mode="replicate")
    result = np.where(img > local_mean - c, 255, 0).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Sobel Gradients
# ---------------------------------------------------------------------------

def sobel(image):
    """Compute Sobel gradients of a grayscale image.

    Uses standard 3×3 Sobel kernels via the convolution pipeline.
    If an RGB image is provided, it is auto-converted to grayscale.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Gx, Gy, magnitude, direction)`` where:

        - ``Gx`` — horizontal gradient.
        - ``Gy`` — vertical gradient.
        - ``magnitude`` — gradient magnitude ``sqrt(Gx² + Gy²)``.
        - ``direction`` — gradient angle in radians ``arctan2(Gy, Gx)``.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    """
    validate_image(image)
    if image.ndim == 3:
        import warnings
        from minicv.io import to_grayscale
        warnings.warn("RGB image auto-converted to grayscale for Sobel.")
        image = to_grayscale(image)

    # Standard 3×3 Sobel kernels
    kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

    ky = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

    gx = convolve2d(image.astype(np.float64), kx, pad_mode="replicate")
    gy = convolve2d(image.astype(np.float64), ky, pad_mode="replicate")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)

    return gx, gy, magnitude, direction


# ---------------------------------------------------------------------------
# Bit-Plane Slicing
# ---------------------------------------------------------------------------

def bit_plane_slice(image, plane):
    """Extract a single bit plane from a grayscale uint8 image.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image with dtype ``uint8``.
    plane : int
        Bit plane index in [0, 7], where 0 is the LSB and 7 is the MSB.

    Returns
    -------
    numpy.ndarray
        Binary image with shape ``(H, W)``, values 0 or 1, dtype ``uint8``.

    Raises
    ------
    TypeError
        If ``image`` dtype is not ``uint8``.
    ValueError
        If ``plane`` is not in [0, 7].
    """
    validate_grayscale(image)
    if image.dtype != np.uint8:
        raise TypeError(
            f"Image must be uint8, got {image.dtype}."
        )
    if not isinstance(plane, (int, np.integer)) or plane < 0 or plane > 7:
        raise ValueError(
            f"plane must be an integer in [0, 7], got {plane}."
        )

    return ((image >> plane) & 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def histogram(image):
    """Compute a 256-bin histogram for a grayscale uint8 image.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image with dtype ``uint8``.

    Returns
    -------
    numpy.ndarray
        1D array of length 256 containing bin counts.

    Raises
    ------
    TypeError
        If ``image`` is not grayscale.
    """
    validate_grayscale(image)
    if image.dtype != np.uint8:
        raise TypeError(
            f"Image must be uint8 for histogram, got {image.dtype}."
        )
    return np.bincount(image.ravel(), minlength=256)


def equalize_histogram(image):
    """Apply histogram equalization to enhance image contrast.

    Uses the CDF-based equalization method: maps each pixel value
    through the normalized cumulative distribution function.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image with dtype ``uint8``.

    Returns
    -------
    numpy.ndarray
        Contrast-enhanced grayscale image, dtype ``uint8``.

    Raises
    ------
    TypeError
        If ``image`` is not grayscale.
    """
    validate_grayscale(image)
    if image.dtype != np.uint8:
        raise TypeError(
            f"Image must be uint8 for equalization, got {image.dtype}."
        )

    hist = histogram(image)
    cdf = hist.cumsum()
    # Mask zeros to avoid division by zero
    cdf_min = cdf[cdf > 0].min()
    total = image.size

    # Normalize CDF to [0, 255]
    lut = ((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
    return lut[image]


# ---------------------------------------------------------------------------
# Bonus Technique 1 — Laplacian Sharpening
# ---------------------------------------------------------------------------

def laplacian_sharpen(image, strength=1.0):
    """Sharpen an image using the Laplacian operator.

    Computes the Laplacian (second derivative) of the image and adds
    it back to the original to enhance edges. Uses the standard 3×3
    Laplacian kernel ``[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]``.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    strength : float, optional
        Sharpening strength multiplier (default 1.0).

    Returns
    -------
    numpy.ndarray
        Sharpened image, clipped to valid range, same shape as input.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``strength`` is negative.

    Notes
    -----
    The Laplacian kernel highlights regions of rapid intensity change.
    The sharpened image is computed as:

        sharpened = original + strength × Laplacian(original)
    """
    validate_image(image)
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}.")

    laplacian_kernel = np.array([[0, -1,  0],
                                  [-1,  4, -1],
                                  [0, -1,  0]], dtype=np.float64)

    laplacian = spatial_filter(image.astype(np.float64), laplacian_kernel)
    sharpened = image.astype(np.float64) + strength * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Bonus Technique 2 — Gamma Correction
# ---------------------------------------------------------------------------

def gamma_correction(image, gamma):
    """Apply gamma correction (power-law transform) to an image.

    Adjusts brightness by mapping pixel intensities through the
    power function: ``output = 255 × (input / 255) ^ gamma``.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    gamma : float
        Gamma value. ``gamma < 1`` brightens; ``gamma > 1`` darkens.

    Returns
    -------
    numpy.ndarray
        Corrected image, dtype ``uint8``.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``gamma`` is non-positive.
    """
    validate_image(image)
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}.")

    normalized = image.astype(np.float64) / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return corrected.astype(np.uint8)
