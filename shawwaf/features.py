"""
Feature extractors for the shawwaf library.

Provides two global image descriptors (color histogram, pixel statistics)
and two gradient-based descriptors (HOG, edge orientation histogram).
All descriptors return 1D NumPy feature vectors suitable for downstream
classification or similarity tasks.
"""

import numpy as np

from shawwaf.utils import validate_image, validate_grayscale
from shawwaf.processing import sobel, histogram as _compute_histogram
from shawwaf.io import to_grayscale


# ---------------------------------------------------------------------------
# Global Descriptor 1 — Color Histogram Descriptor
# ---------------------------------------------------------------------------

def color_histogram_descriptor(image, bins=32):
    """Compute a color histogram feature vector.

    For RGB images, independent histograms are computed per channel
    and concatenated. For grayscale images, a single histogram is used.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image, dtype ``uint8``.
    bins : int, optional
        Number of histogram bins per channel (default 32).

    Returns
    -------
    numpy.ndarray
        1D feature vector of length ``bins`` (grayscale) or ``3 * bins``
        (RGB), dtype ``float64``, L1-normalized.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``bins`` is not a positive integer.
    """
    validate_image(image)
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}.")

    if image.ndim == 2:
        hist, _ = np.histogram(image.ravel(), bins=bins, range=(0, 256))
        feat = hist.astype(np.float64)
    else:
        hists = []
        for c in range(image.shape[2]):
            h, _ = np.histogram(image[:, :, c].ravel(), bins=bins, range=(0, 256))
            hists.append(h.astype(np.float64))
        feat = np.concatenate(hists)

    # L1 normalize
    total = feat.sum()
    if total > 0:
        feat /= total
    return feat


# ---------------------------------------------------------------------------
# Global Descriptor 2 — Pixel Statistics Descriptor
# ---------------------------------------------------------------------------

def pixel_statistics_descriptor(image):
    """Compute a feature vector of pixel-level statistics.

    For grayscale images, computes 6 statistics: mean, std, min, max,
    skewness, kurtosis. For RGB images, these are computed per channel
    and concatenated (18 values total).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.

    Returns
    -------
    numpy.ndarray
        1D feature vector of length 6 (grayscale) or 18 (RGB),
        dtype ``float64``.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    """
    validate_image(image)

    def _channel_stats(ch):
        ch = ch.astype(np.float64).ravel()
        mean = ch.mean()
        std = ch.std()
        mn = ch.min()
        mx = ch.max()
        # Skewness (Fisher's definition)
        if std > 0:
            skew = np.mean(((ch - mean) / std) ** 3)
            kurt = np.mean(((ch - mean) / std) ** 4) - 3.0
        else:
            skew = 0.0
            kurt = 0.0
        return np.array([mean, std, mn, mx, skew, kurt], dtype=np.float64)

    if image.ndim == 2:
        return _channel_stats(image)

    stats = [_channel_stats(image[:, :, c]) for c in range(image.shape[2])]
    return np.concatenate(stats)


# ---------------------------------------------------------------------------
# Gradient Descriptor 1 — HOG (Histogram of Oriented Gradients)
# ---------------------------------------------------------------------------

def hog_descriptor(image, cell_size=8, bins=9):
    """Compute a Histogram of Oriented Gradients (HOG) feature vector.

    Divides the image into cells and computes a histogram of gradient
    orientations within each cell. Uses the Sobel pipeline internally
    for gradient computation.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    cell_size : int, optional
        Side length of each square cell in pixels (default 8).
    bins : int, optional
        Number of orientation bins spanning [0°, 180°) (default 9).

    Returns
    -------
    numpy.ndarray
        1D feature vector of length ``n_cells_y × n_cells_x × bins``,
        dtype ``float64``, L2-normalized.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``cell_size`` or ``bins`` is not positive.
    """
    validate_image(image)
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}.")
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}.")

    if image.ndim == 3:
        image = to_grayscale(image)

    _, _, magnitude, direction = sobel(image)

    # Convert direction from [-pi, pi] to [0, pi)
    angle = direction % np.pi

    h, w = image.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size

    # Crop to exact multiple of cell_size
    magnitude = magnitude[:n_cells_y * cell_size, :n_cells_x * cell_size]
    angle = angle[:n_cells_y * cell_size, :n_cells_x * cell_size]

    features = []
    bin_width = np.pi / bins

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell_mag = magnitude[cy * cell_size:(cy + 1) * cell_size,
                                  cx * cell_size:(cx + 1) * cell_size]
            cell_angle = angle[cy * cell_size:(cy + 1) * cell_size,
                               cx * cell_size:(cx + 1) * cell_size]

            hist = np.zeros(bins, dtype=np.float64)
            bin_idx = np.clip((cell_angle / bin_width).astype(int), 0, bins - 1)

            for b in range(bins):
                mask = bin_idx == b
                hist[b] = cell_mag[mask].sum()

            features.append(hist)

    feat = np.concatenate(features)

    # L2 normalize
    norm = np.linalg.norm(feat) + 1e-6
    return feat / norm


# ---------------------------------------------------------------------------
# Gradient Descriptor 2 — Edge Orientation Histogram
# ---------------------------------------------------------------------------

def edge_orientation_histogram(image, bins=36):
    """Compute a global edge orientation histogram.

    Computes Sobel gradients and builds a magnitude-weighted histogram
    of gradient directions across the entire image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    bins : int, optional
        Number of orientation bins spanning [0°, 360°) (default 36).

    Returns
    -------
    numpy.ndarray
        1D feature vector of length ``bins``, dtype ``float64``,
        L1-normalized.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``bins`` is not positive.
    """
    validate_image(image)
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}.")

    if image.ndim == 3:
        image = to_grayscale(image)

    _, _, magnitude, direction = sobel(image)

    # Convert direction from [-pi, pi] to [0, 2*pi)
    angle = direction % (2 * np.pi)

    bin_width = 2 * np.pi / bins
    bin_idx = np.clip((angle / bin_width).astype(int), 0, bins - 1)

    hist = np.zeros(bins, dtype=np.float64)
    for b in range(bins):
        mask = bin_idx == b
        hist[b] = magnitude[mask].sum()

    # L1 normalize
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist
