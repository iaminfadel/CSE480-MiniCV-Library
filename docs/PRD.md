# Product Requirements Document
## shawwaf — Python Image Processing Library
### CSE480: Machine Vision | Milestone 1 | Spring 2026

---

## Overview

shawwaf is a reusable, from-scratch Python image-processing library that emulates a well-defined subset of OpenCV. It is implemented using only NumPy, Pandas, Matplotlib, and the Python standard library. The library must be structured as a proper, importable Python package with clean module separation, comprehensive docstrings, and rigorous input validation.

---

## Goals

- Deliver a maintainable, modular image-processing package usable as a drop-in backend.
- Implement core OpenCV-equivalent functionality without using OpenCV or any dedicated vision library.
- Uphold production-level code quality: validated inputs, vectorized operations, and full documentation.

---

## Non-Goals

- No dependency on OpenCV, scikit-image, PIL/Pillow, or any image-processing framework.
- No GUI or interactive visualization tooling beyond Matplotlib-based display.
- No Milestone 2 ML pipeline components (dataset loading, model training, etc.).

---

## Project Structure

The repository must be laid out as a proper Python project, not a single script. The package source, tests, documentation, and notebooks each live in their own top-level folder.

```
shawwaf-repo/
├── shawwaf/                  # Installable package source
│   ├── __init__.py
│   ├── io.py                # Image I/O & color conversion
│   ├── utils.py             # Shared utilities (normalization, clipping, padding, dtype helpers)
│   ├── filtering.py         # Convolution, spatial filters (mean, Gaussian, median)
│   ├── transforms.py        # Geometric transformations (resize, rotate, translate)
│   ├── features.py          # Feature extractors (global + gradient descriptors)
│   ├── drawing.py           # Drawing primitives + text placement
│   └── processing.py        # Thresholding, Sobel, bit-plane, histogram, extras
│
├── tests/                   # Unit & integration tests (one file per module)
│   ├── test_io.py
│   ├── test_filtering.py
│   ├── test_processing.py
│   ├── test_transforms.py
│   ├── test_features.py
│   └── test_drawing.py
│
├── docs/                    # Project documentation
│   ├── api.md               # API reference (generated from docstrings)
│   ├── math_notes.md        # Math & algorithm explanations with equations
│   └── results/             # Output images used in the results section
│       ├── filtering_results.png
│       ├── threshold_comparison.png
│       └── ...
│
├── notebooks/               # Demonstration & verification notebooks
│   └── demo.ipynb
│
├── README.md                # Setup instructions, quickstart, and results overview
├── requirements.txt         # numpy, pandas, matplotlib (and versions)
└── setup.py                 # Package install configuration
```

---

## User Stories

---

### Epic 1 — Packaging & Project Structure

**US-1.1 — Importable Package**
> As a developer, I want to import `shawwaf` and its submodules cleanly, so that I can use the library in any Python project without path hacks.

**Acceptance Criteria:**
- `import shawwaf` succeeds after a standard install or `sys.path` addition.
- Each submodule (`shawwaf.io`, `shawwaf.filtering`, etc.) is independently importable.
- No circular imports exist between modules.

---

**US-1.2 — Module Separation**
> As a maintainer, I want each functional area (I/O, filtering, transforms, features, drawing, utils) to live in its own module, so that the codebase is easy to navigate and extend.

**Acceptance Criteria:**
- No module contains logic belonging to another domain (e.g., no drawing code in `filtering.py`).
- Shared helpers (dtype conversion, validation, padding) are centralized in `utils.py`.
- No duplicated utility code across modules.

---

### Epic 2 — Image I/O & Core Utilities

**US-2.1 — Read Image**
> As a user, I want to load a PNG or JPG image from disk into a NumPy array, so that I can process it programmatically.

**Acceptance Criteria:**
- Function `read_image(path)` returns a NumPy array with shape `(H, W, 3)` for RGB or `(H, W)` for grayscale.
- Supports at minimum PNG and JPG formats via Matplotlib backends.
- Raises `FileNotFoundError` for missing paths and `ValueError` for unsupported formats.

---

**US-2.2 — Export Image**
> As a user, I want to save an in-memory NumPy array to a PNG or JPG file on disk, so that I can persist my processed results.

**Acceptance Criteria:**
- Function `save_image(array, path)` writes the array to disk in the correct format based on file extension.
- Supports both grayscale `(H, W)` and RGB `(H, W, 3)` arrays.
- Raises `ValueError` if the array shape or dtype is incompatible with the target format.

---

**US-2.3 — Color Conversion**
> As a user, I want to convert an image between RGB and grayscale, so that I can prepare it for grayscale-only processing functions.

**Acceptance Criteria:**
- `to_grayscale(image)` converts an `(H, W, 3)` RGB array to `(H, W)` using a standard luminance formula.
- `to_rgb(image)` converts a `(H, W)` grayscale array to `(H, W, 3)` by stacking channels.
- Output dtype and value range are documented and consistent.

---

### Epic 3 — Core Operations (Foundation Functions)

**US-3.1 — Image Normalization**
> As a user, I want to normalize pixel values using at least 3 different modes, so that I can prepare images for downstream algorithms with varying input expectations.

**Acceptance Criteria:**
- `normalize(image, mode)` supports at minimum: `'minmax'` (scale to [0, 1]), `'zscore'` (zero mean, unit std), and `'uint8'` (scale to [0, 255]).
- Raises `ValueError` for unknown modes.
- Output dtype matches the documented contract for each mode.

---

**US-3.2 — Pixel Clipping**
> As a user, I want to clip pixel values to a defined range, so that I can prevent overflow or underflow after arithmetic operations.

**Acceptance Criteria:**
- `clip(image, min_val, max_val)` clamps all values to `[min_val, max_val]`.
- Works on both grayscale and RGB arrays.
- Raises `ValueError` if `min_val >= max_val`.

---

**US-3.3 — Padding**
> As a user, I want to pad an image using at least 3 modes, so that convolution boundary handling is well-defined and configurable.

**Acceptance Criteria:**
- `pad(image, pad_width, mode)` supports at minimum: `'zero'` (constant 0), `'reflect'`, and `'replicate'` (edge repeat).
- Works on grayscale `(H, W)` arrays.
- Raises `ValueError` for invalid `pad_width` or unknown `mode`.

---

**US-3.4 — 2D Convolution**
> As a user, I want to convolve a grayscale image with a kernel, so that I can apply any linear spatial filter via a single unified function.

**Acceptance Criteria:**
- `convolve2d(image, kernel, pad_mode)` performs true 2D convolution (kernel is flipped) on a grayscale array.
- Boundary handling is implemented by calling the `pad()` function internally.
- Raises `ValueError` if the kernel has even dimensions, is empty, or is non-numeric.
- Raises `TypeError` if the image or kernel is not a NumPy array.

---

**US-3.5 — 2D Spatial Filtering (Grayscale + RGB)**
> As a user, I want to apply a convolution-based filter to both grayscale and RGB images, so that I can use any kernel filter regardless of image type.

**Acceptance Criteria:**
- `spatial_filter(image, kernel, pad_mode)` dispatches to per-channel convolution for RGB inputs.
- The channel-merging strategy (per-channel or alternative) is documented in the docstring.
- Output shape matches input shape.

---

### Epic 4 — Image Processing Techniques

**US-4.1 — Mean / Box Filter**
> As a user, I want to smooth an image with a box filter of configurable size, so that I can reduce noise with a simple averaging operation.

**Acceptance Criteria:**
- `mean_filter(image, kernel_size)` builds a normalized box kernel and applies it via `spatial_filter`.
- `kernel_size` must be a positive odd integer; raises `ValueError` otherwise.

---

**US-4.2 — Gaussian Filter**
> As a user, I want to smooth an image with a Gaussian filter defined by size and sigma, so that I can apply perceptually-weighted blur.

**Acceptance Criteria:**
- `gaussian_kernel(size, sigma)` generates a normalized 2D Gaussian kernel.
- `gaussian_filter(image, size, sigma)` applies the kernel via the convolution pipeline.
- Raises `ValueError` for non-positive `sigma` or even `size`.

---

**US-4.3 — Median Filter**
> As a user, I want to apply a median filter to remove salt-and-pepper noise, so that I can denoise images without blurring edges.

**Acceptance Criteria:**
- `median_filter(image, kernel_size)` replaces each pixel with the median of its neighborhood.
- Any Python-level loop used is documented and justified in the docstring (e.g., neighborhood iteration).
- Works on grayscale images; RGB support is documented.

---

**US-4.4 — Thresholding**
> As a user, I want to binarize a grayscale image using global, Otsu, or adaptive thresholding, so that I can segment foreground from background under different conditions.

**Acceptance Criteria:**
- `threshold(image, method, **kwargs)` supports `'global'` (user-supplied value), `'otsu'` (automatic), and `'adaptive'` (mean or Gaussian local window).
- Returns a binary `(H, W)` array with values 0 or 255.
- Raises `ValueError` for missing required kwargs per method.

---

**US-4.5 — Sobel Gradients**
> As a user, I want to compute horizontal and vertical Sobel gradients, so that I can detect edges and estimate gradient magnitude and direction.

**Acceptance Criteria:**
- `sobel(image)` returns `(Gx, Gy, magnitude, direction)` as NumPy arrays.
- Uses the `convolve2d` pipeline internally with the standard 3×3 Sobel kernels.
- Operates on grayscale images; RGB inputs trigger auto-conversion with a warning or documented behavior.

---

**US-4.6 — Bit-Plane Slicing**
> As a user, I want to extract individual bit planes from a grayscale image, so that I can analyze the contribution of each bit to the overall image.

**Acceptance Criteria:**
- `bit_plane_slice(image, plane)` extracts the specified bit plane (0–7) as a binary array.
- `plane` must be an integer in [0, 7]; raises `ValueError` otherwise.
- Input image must be `uint8`; raises `TypeError` for other dtypes.

---

**US-4.7 — Histogram & Histogram Equalization**
> As a user, I want to compute a grayscale histogram and apply histogram equalization, so that I can analyze and enhance image contrast.

**Acceptance Criteria:**
- `histogram(image)` returns a 256-bin count array for a grayscale uint8 image.
- `equalize_histogram(image)` returns a contrast-enhanced grayscale image using the CDF-based equalization method.
- Both functions raise `TypeError` for non-grayscale inputs.

---

**US-4.8 — Two Additional Techniques**
> As a developer, I want to include two additional image-processing techniques beyond the required set, so that the library demonstrates broader capability.

**Acceptance Criteria:**
- Two additional functions are implemented, each in the correct module.
- Each has a full docstring including description, parameters, return values, and algorithm notes.
- Suggested candidates: Laplacian sharpening, unsharp masking, morphological erosion/dilation, CLAHE, or gamma correction.

---

### Epic 5 — Geometric Transformations

**US-5.1 — Resize**
> As a user, I want to resize an image to a target resolution using at least two interpolation methods, so that I can scale images for display or model input.

**Acceptance Criteria:**
- `resize(image, target_size, method)` supports `'nearest'` and `'bilinear'` interpolation.
- `target_size` is a `(height, width)` tuple; raises `ValueError` for non-positive dimensions.
- Raises `ValueError` for unknown `method`.

---

**US-5.2 — Rotation**
> As a user, I want to rotate an image by an arbitrary angle about its center, so that I can apply orientation augmentations or correct skew.

**Acceptance Criteria:**
- `rotate(image, angle, interpolation)` rotates the image about the center in degrees.
- Fills out-of-bounds pixels with a documented fill value (e.g., 0).
- Uses the same interpolation enum as `resize`.

---

**US-5.3 — Translation**
> As a user, I want to translate an image by a given (tx, ty) pixel offset, so that I can shift image content.

**Acceptance Criteria:**
- `translate(image, tx, ty)` shifts the image and fills vacated regions with 0 (or documented fill value).
- Supports both positive and negative offsets.
- Output shape matches input shape.

---

### Epic 6 — Feature Extractors

**US-6.1 — Two Global Descriptors**
> As a user, I want to extract at least two global image descriptors, so that I can represent the overall statistical properties of an image as a feature vector.

**Acceptance Criteria:**
- Two distinct global descriptors are implemented (e.g., color histogram, pixel statistics, LBP histogram, Hu moments).
- Each returns a 1D NumPy feature vector.
- Both have full docstrings including output dimensionality.

---

**US-6.2 — Two Gradient Descriptors**
> As a user, I want to extract at least two gradient-based descriptors, so that I can capture structural and edge information for recognition tasks.

**Acceptance Criteria:**
- Two distinct gradient descriptors are implemented (e.g., HOG, edge histogram, gradient magnitude histogram, Gabor response summary).
- Each returns a 1D NumPy feature vector.
- Both use the `sobel` or `convolve2d` pipeline internally.

---

### Epic 7 — Drawing Primitives

**US-7.1 — Draw Point**
> As a user, I want to draw a colored point at a given pixel coordinate, so that I can mark keypoints or annotations on an image.

**Acceptance Criteria:**
- `draw_point(image, x, y, color, thickness)` draws a point in-place or on a copy.
- Color is a grayscale scalar or RGB tuple.
- Coordinates outside canvas boundaries are clipped silently (no error).

---

**US-7.2 — Draw Line**
> As a user, I want to draw a straight line between two points using Bresenham's algorithm or equivalent, so that I can annotate or render geometric overlays.

**Acceptance Criteria:**
- `draw_line(image, x0, y0, x1, y1, color, thickness)` rasterizes a line.
- Pixels outside the canvas are clipped.
- Thickness > 1 is handled by expanding the drawn pixels.

---

**US-7.3 — Draw Rectangle**
> As a user, I want to draw a rectangle defined by a corner and size, in both outline and filled modes, so that I can annotate bounding boxes.

**Acceptance Criteria:**
- `draw_rectangle(image, x, y, w, h, color, thickness, filled)` draws an outline or filled rectangle.
- `filled=True` fills the interior with `color`.
- Boundary clipping is applied; partial rectangles are drawn where visible.

---

**US-7.4 — Draw Polygon**
> As a user, I want to draw a polygon from a list of vertices (outline; filled is optional), so that I can annotate arbitrary shapes.

**Acceptance Criteria:**
- `draw_polygon(image, points, color, thickness, filled)` draws the outline by connecting consecutive vertices.
- `points` is a list or array of `(x, y)` pairs; raises `ValueError` for fewer than 3 points.
- Filled polygon support is noted as optional but valuable.

---

### Epic 8 — Text Placement

**US-8.1 — Draw Text**
> As a user, I want to render a text string onto an image at a given position with configurable scale and color, so that I can add labels and captions to images.

**Acceptance Criteria:**
- `draw_text(image, text, x, y, font_scale, color)` renders text using Matplotlib font rendering or a bitmap font approach.
- `font_scale` controls character size.
- Text that exceeds canvas boundaries is clipped.

---

### Epic 9 — Code Quality & Engineering

**US-9.1 — Docstrings**
> As a contributor, I want every public function to have a complete docstring, so that I can understand how to use each function without reading its implementation.

**Acceptance Criteria:**
- Every public function includes: description, parameters with types, return value and type, raised exceptions, and notes on expected input ranges/dtypes.
- Docstrings follow a consistent style (e.g., NumPy or Google format).

---

**US-9.2 — Input Validation**
> As a user, I want all functions to raise clear, specific errors for invalid inputs, so that I receive immediate and actionable feedback when I misuse the API.

**Acceptance Criteria:**
- `TypeError` is raised for wrong argument types.
- `ValueError` is raised for invalid values or shapes.
- Error messages specify what failed and what the expected format is (e.g., `"kernel must have odd dimensions, got (4, 4)"`).

---

**US-9.3 — Performance Practices**
> As a developer, I want the library to prefer NumPy vectorization over Python loops, so that processing is fast enough for interactive use on typical images.

**Acceptance Criteria:**
- No pixel-by-pixel Python loops appear in convolution, normalization, thresholding, or transformation functions.
- Any loop present (e.g., median filter neighborhood iteration) is documented and justified in the docstring.

---

**US-9.4 — Modularity & No Duplication**
> As a maintainer, I want shared logic centralized in `utils.py` and no copy-pasted code across modules, so that bug fixes only need to happen in one place.

**Acceptance Criteria:**
- Padding, dtype conversion, and input validation utilities are defined once in `utils.py` and imported elsewhere.
- A code review finds no duplicated functions or logic blocks across modules.

---

### Epic 10 — Documentation

**US-10.1 — GitHub Repository**
> As a reviewer, I want the project hosted on a public GitHub repository with a clean commit history, so that I can evaluate the development process.

**Acceptance Criteria:**
- Repository is public and contains the full `shawwaf` package.
- `README.md` provides installation instructions and a quickstart example.

---

**US-10.2 — API Reference**
> As a user, I want a browsable API reference generated from docstrings, so that I can look up any function's signature and behavior quickly.

**Acceptance Criteria:**
- Docstrings serve as the source of truth for all public APIs.
- A rendered or plaintext API reference is included in the repository (e.g., via `pdoc`, `sphinx`, or a hand-written `docs/api.md`).

---

**US-10.3 — Math & Algorithms Notes**
> As a student or reviewer, I want a dedicated section explaining the math and algorithms behind each technique, so that I can verify correctness and learn from the implementation.

**Acceptance Criteria:**
- Each major algorithm (Gaussian kernel, Otsu, histogram equalization, Sobel, convolution, etc.) has a concise written explanation.
- Equations and/or pseudocode are included where appropriate.
- Notes are organized to mirror the library's module structure.

---

**US-10.4 — Results & Verification**
> As a reviewer, I want a results section showing visual and quantitative output for all major APIs, so that I can confirm the library produces correct results.

**Acceptance Criteria:**
- Each major function has at least one before/after image or output sample included in the documentation.
- Test images and scripts used to generate the results are committed to the repository.

---

## Summary Table

| Epic      | Area                  | # Stories |
|-----------|-----------------------|-----------|
| 1         | Packaging & Structure | 2         |
| 2         | Image I/O & Utilities | 3         |
| 3         | Core Operations       | 5         |
| 4         | Processing Techniques | 8         |
| 5         | Geometric Transforms  | 3         |
| 6         | Feature Extractors    | 2         |
| 7         | Drawing Primitives    | 4         |
| 8         | Text Placement        | 1         |
| 9         | Code Quality          | 4         |
| 10        | Documentation         | 4         |
| **Total** |                       | **36**    |