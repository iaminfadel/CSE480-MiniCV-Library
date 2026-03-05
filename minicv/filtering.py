"""
Convolution and spatial filters for the MiniCV library.

Provides 2D convolution, a spatial filter dispatcher for RGB images,
and built-in mean, Gaussian, and median filters. All linear filters
use the convolution pipeline defined here.
"""

import numpy as np

from minicv.utils import validate_image, validate_grayscale, validate_kernel, pad


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

def convolve2d(image, kernel, pad_mode="zero"):
    """Perform true 2D convolution on a grayscale image.

    The kernel is flipped (180° rotation) before sliding, following
    the mathematical definition of convolution. Boundary handling is
    delegated to :func:`minicv.utils.pad`.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image with shape ``(H, W)``.
    kernel : numpy.ndarray
        2D convolution kernel with odd dimensions.
    pad_mode : str, optional
        Padding mode passed to :func:`minicv.utils.pad`.
        One of ``'zero'``, ``'reflect'``, ``'replicate'``.

    Returns
    -------
    numpy.ndarray
        Convolved image with shape ``(H, W)``, dtype ``float64``.

    Raises
    ------
    TypeError
        If ``image`` or ``kernel`` is not a NumPy ndarray.
    ValueError
        If the kernel has even dimensions, is empty, or image is not 2D.
    """
    validate_grayscale(image)
    validate_kernel(kernel)

    # Flip kernel for true convolution
    flipped = kernel[::-1, ::-1].astype(np.float64)
    kh, kw = flipped.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = pad(image.astype(np.float64), max(pad_h, pad_w), mode=pad_mode)

    # Handle asymmetric kernels by trimming excess padding
    if pad_h != pad_w:
        ph = max(pad_h, pad_w)
        pw = max(pad_h, pad_w)
        # Re-pad with the larger value, then we'll use appropriate slicing
        padded = pad(image.astype(np.float64), max(pad_h, pad_w), mode=pad_mode)

    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float64)

    # Vectorized convolution using shifted views
    for i in range(kh):
        for j in range(kw):
            row_start = i + (max(pad_h, pad_w) - pad_h)
            col_start = j + (max(pad_h, pad_w) - pad_w)
            output += flipped[i, j] * padded[row_start:row_start + h,
                                              col_start:col_start + w]

    return output


# ---------------------------------------------------------------------------
# Spatial filter (RGB dispatcher)
# ---------------------------------------------------------------------------

def spatial_filter(image, kernel, pad_mode="zero"):
    """Apply a convolution-based filter to grayscale or RGB images.

    For RGB images, convolution is applied independently to each channel
    and the results are merged back. This per-channel strategy preserves
    color information while applying the same spatial filter uniformly.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    kernel : numpy.ndarray
        2D convolution kernel with odd dimensions.
    pad_mode : str, optional
        Padding mode (``'zero'``, ``'reflect'``, ``'replicate'``).

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape as the input, dtype ``float64``.

    Raises
    ------
    TypeError
        If ``image`` or ``kernel`` is not a NumPy ndarray.
    ValueError
        If the kernel is invalid or image dimensions are wrong.
    """
    validate_image(image)
    validate_kernel(kernel)

    if image.ndim == 2:
        return convolve2d(image, kernel, pad_mode)

    # Per-channel convolution for RGB
    channels = []
    for c in range(image.shape[2]):
        channels.append(convolve2d(image[:, :, c], kernel, pad_mode))
    return np.stack(channels, axis=-1)


# ---------------------------------------------------------------------------
# Mean / Box Filter
# ---------------------------------------------------------------------------

def mean_filter(image, kernel_size=3):
    """Smooth an image with a box (mean) filter.

    Builds a normalized box kernel and applies it via
    :func:`spatial_filter`.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    kernel_size : int, optional
        Side length of the square box kernel. Must be a positive odd integer.

    Returns
    -------
    numpy.ndarray
        Smoothed image with the same shape as the input.

    Raises
    ------
    ValueError
        If ``kernel_size`` is not a positive odd integer.
    """
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}."
        )
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    return spatial_filter(image, kernel)


# ---------------------------------------------------------------------------
# Gaussian Filter
# ---------------------------------------------------------------------------

def gaussian_kernel(size, sigma):
    """Generate a normalized 2D Gaussian kernel.

    Parameters
    ----------
    size : int
        Side length of the square kernel. Must be a positive odd integer.
    sigma : float
        Standard deviation of the Gaussian distribution. Must be positive.

    Returns
    -------
    numpy.ndarray
        Normalized 2D Gaussian kernel with shape ``(size, size)``,
        dtype ``float64``.

    Raises
    ------
    ValueError
        If ``size`` is not a positive odd integer or ``sigma`` is
        non-positive.
    """
    if not isinstance(size, (int, np.integer)) or size <= 0 or size % 2 == 0:
        raise ValueError(
            f"size must be a positive odd integer, got {size}."
        )
    if sigma <= 0:
        raise ValueError(
            f"sigma must be positive, got {sigma}."
        )

    center = size // 2
    y, x = np.mgrid[-center:center + 1, -center:center + 1].astype(np.float64)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_filter(image, size=5, sigma=1.0):
    """Smooth an image with a Gaussian filter.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    size : int, optional
        Side length of the square Gaussian kernel.
    sigma : float, optional
        Standard deviation of the Gaussian.

    Returns
    -------
    numpy.ndarray
        Smoothed image with the same shape as the input.
    """
    kernel = gaussian_kernel(size, sigma)
    return spatial_filter(image, kernel)


# ---------------------------------------------------------------------------
# Median Filter
# ---------------------------------------------------------------------------

def median_filter(image, kernel_size=3):
    """Apply a median filter to remove salt-and-pepper noise.

    Each pixel is replaced by the median of its local neighborhood.
    A Python-level loop over the kernel neighborhood is used because
    the median operation is non-linear and cannot be expressed as a
    convolution. The loop iterates over the kernel footprint elements,
    not individual pixels, keeping computation tractable.

    For RGB images, the filter is applied independently to each channel.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    kernel_size : int, optional
        Side length of the square neighborhood. Must be a positive
        odd integer.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape and dtype as the input.

    Raises
    ------
    ValueError
        If ``kernel_size`` is not a positive odd integer.
    """
    validate_image(image)
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}."
        )

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(_median_filter_2d(image[:, :, c], kernel_size))
        return np.stack(channels, axis=-1)

    return _median_filter_2d(image, kernel_size)


def _median_filter_2d(image, kernel_size):
    """Apply median filter to a single 2D channel.

    Uses a neighborhood-collection approach: collects all shifted views
    into a stack and takes the median along the stack axis — avoiding
    a per-pixel Python loop.
    """
    half = kernel_size // 2
    padded = np.pad(image, half, mode="edge").astype(np.float64)
    h, w = image.shape

    # Collect all shifted neighborhoods into a 3D stack
    neighbors = np.empty((kernel_size * kernel_size, h, w), dtype=np.float64)
    idx = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            neighbors[idx] = padded[i:i + h, j:j + w]
            idx += 1

    result = np.median(neighbors, axis=0)
    return result.astype(image.dtype)
