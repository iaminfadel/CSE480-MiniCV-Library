"""
Image I/O and color conversion for the MiniCV library.

Provides functions to read/write images from/to disk and convert
between RGB and grayscale color spaces. Uses Matplotlib backends for
file I/O — no OpenCV, PIL, or scikit-image dependency.
"""

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from minicv.utils import validate_image


# Supported file extensions
_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_image(path):
    """Load a PNG or JPG image from disk into a NumPy array.

    Parameters
    ----------
    path : str
        Path to the image file. Must be a PNG or JPG file.

    Returns
    -------
    numpy.ndarray
        Image array with shape ``(H, W, 3)`` for RGB or ``(H, W)`` for
        grayscale, dtype ``uint8``, values in [0, 255].

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file extension is not supported.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: '{path}'.")

    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported formats: {_SUPPORTED_EXTENSIONS}."
        )

    img = mpimg.imread(path)

    # Matplotlib returns float32 [0,1] for PNGs, uint8 [0,255] for JPGs
    if img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8)

    # Drop alpha channel if present (RGBA -> RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def save_image(image, path):
    """Save a NumPy array as an image file on disk.

    Parameters
    ----------
    image : numpy.ndarray
        Image array with shape ``(H, W)`` for grayscale or ``(H, W, 3)``
        for RGB. Values should be in [0, 255] with dtype ``uint8``, or
        [0.0, 1.0] with a float dtype.
    path : str
        Destination file path. Extension determines the format.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If the array shape is incompatible or the format is unsupported.
    """
    validate_image(image)
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported formats: {_SUPPORTED_EXTENSIONS}."
        )

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(
            f"RGB image must have 3 channels, got {image.shape[2]}."
        )

    # Normalize to [0,1] float for Matplotlib's imsave
    img = image.astype(np.float64)
    min_val, max_val = img.min(), img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    if image.ndim == 2:
        plt.imsave(path, img, cmap="gray")
    else:
        plt.imsave(path, img)


def to_grayscale(image):
    """Convert an RGB image to grayscale using the luminance formula.

    Uses the standard Rec. 601 luminance weights:
    ``Y = 0.2989 * R + 0.5870 * G + 0.1140 * B``

    Parameters
    ----------
    image : numpy.ndarray
        RGB image with shape ``(H, W, 3)`` and dtype ``uint8``.

    Returns
    -------
    numpy.ndarray
        Grayscale image with shape ``(H, W)``, dtype ``uint8``.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy ndarray.
    ValueError
        If ``image`` is not a 3-channel RGB array.
    """
    validate_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an (H, W, 3) RGB image, got shape {image.shape}."
        )

    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(image.astype(np.float64), weights)
    return gray.astype(np.uint8)


def to_rgb(image):
    """Convert a grayscale image to RGB by stacking the single channel.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image with shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``(H, W, 3)``, same dtype as input.

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

    return np.stack([image, image, image], axis=-1)
