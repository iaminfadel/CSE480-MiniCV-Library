"""
MiniCV — A from-scratch Python image-processing library.

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

from minicv import io
from minicv import utils
from minicv import filtering
from minicv import processing
from minicv import transforms
from minicv import features
from minicv import drawing

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
