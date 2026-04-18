"""
Shawwaf — A from-scratch Python image-processing library.

Provides core image processing functionality using only NumPy, Pandas,
Matplotlib, and the Python standard library. Emulates a well-defined
subset of OpenCV.

Submodules
----------
io : Image I/O and color conversion.
utils : Shared utilities (normalization, clipping, padding, dtype helpers).
filtering : Convolution and spatial filters (mean, Gaussian, median).
processing : Thresholding, Sobel, bit-plane slicing, histogram operations.
transforms : Geometric transformations (resize, rotate, translate).
features : Feature extractors (global and gradient descriptors).
drawing : Drawing primitives and text placement.
"""

from shawwaf import io
from shawwaf import utils
from shawwaf import filtering
from shawwaf import processing
from shawwaf import transforms
from shawwaf import features
from shawwaf import drawing

__version__ = "0.1.0"

__all__ = [
    "io",
    "utils",
    "filtering",
    "processing",
    "transforms",
    "features",
    "drawing",
]
