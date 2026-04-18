# shawwaf API Reference

Complete API reference for the `shawwaf` package, organized by module.

---

## `shawwaf.io` — Image I/O & Color Conversion

### `read_image(path) → ndarray`

Load a PNG or JPG image from disk.

| Parameter | Type  | Description            |
|-----------|-------|------------------------|
| `path`    | `str` | Path to the image file |

**Returns:** `ndarray` with shape `(H, W, 3)` for RGB or `(H, W)` for grayscale, dtype `uint8`.

**Raises:** `FileNotFoundError` if path doesn't exist; `ValueError` for unsupported formats.

---

### `save_image(image, path)`

Save a NumPy array to disk as PNG or JPG.

| Parameter | Type      | Description                   |
|-----------|-----------|-------------------------------|
| `image`   | `ndarray` | `(H, W)` or `(H, W, 3)` array |
| `path`    | `str`     | Destination file path         |

**Raises:** `TypeError` for non-array input; `ValueError` for bad shape or unsupported format.

---

### `to_grayscale(image) → ndarray`

Convert RGB to grayscale using Rec. 601 luminance weights: `Y = 0.2989R + 0.5870G + 0.1140B`.

| Parameter | Type      | Description                    |
|-----------|-----------|--------------------------------|
| `image`   | `ndarray` | RGB image `(H, W, 3)`, `uint8` |

**Returns:** Grayscale `(H, W)`, dtype `uint8`.

---

### `to_rgb(image) → ndarray`

Convert grayscale to RGB by stacking the channel 3×.

| Parameter | Type      | Description              |
|-----------|-----------|--------------------------|
| `image`   | `ndarray` | Grayscale image `(H, W)` |

**Returns:** RGB `(H, W, 3)`, same dtype as input.

---

## `shawwaf.utils` — Core Utilities

### `normalize(image, mode='minmax') → ndarray`

Normalize pixel values.

| Parameter | Type      | Description                                                                                       |
|-----------|-----------|---------------------------------------------------------------------------------------------------|
| `image`   | `ndarray` | Input image                                                                                       |
| `mode`    | `str`     | `'minmax'` → [0, 1] float64; `'zscore'` → zero mean, unit std float64; `'uint8'` → [0, 255] uint8 |

---

### `clip(image, min_val, max_val) → ndarray`

Clamp pixel values to `[min_val, max_val]`.

**Raises:** `ValueError` if `min_val >= max_val`.

---

### `pad(image, pad_width, mode='zero') → ndarray`

Pad a grayscale image. Modes: `'zero'`, `'reflect'`, `'replicate'`.

| Parameter   | Type      | Description                |
|-------------|-----------|----------------------------|
| `image`     | `ndarray` | 2D grayscale image         |
| `pad_width` | `int`     | Pixels to pad on each side |
| `mode`      | `str`     | Padding mode               |

---

## `shawwaf.filtering` — Convolution & Spatial Filters

### `convolve2d(image, kernel, pad_mode='zero') → ndarray`

True 2D convolution (kernel is flipped) on a grayscale image.

| Parameter  | Type      | Description                   |
|------------|-----------|-------------------------------|
| `image`    | `ndarray` | 2D grayscale image            |
| `kernel`   | `ndarray` | 2D kernel with odd dimensions |
| `pad_mode` | `str`     | Padding mode                  |

**Returns:** Convolved image, `float64`.

---

### `spatial_filter(image, kernel, pad_mode='zero') → ndarray`

Apply convolution to grayscale or RGB (per-channel dispatch).

---

### `mean_filter(image, kernel_size=3) → ndarray`

Box (mean) smoothing filter.

---

### `gaussian_kernel(size, sigma) → ndarray`

Generate a normalized 2D Gaussian kernel.

---

### `gaussian_filter(image, size=5, sigma=1.0) → ndarray`

Gaussian blur via the convolution pipeline.

---

### `median_filter(image, kernel_size=3) → ndarray`

Non-linear median filter for salt-and-pepper noise removal. Works on grayscale and RGB.

---

## `shawwaf.processing` — Image Processing

### `threshold(image, method='global', **kwargs) → ndarray`

Binarize a grayscale image. Returns binary `uint8` with values 0 or 255.

| Method       | Required kwargs                                                                          |
|--------------|------------------------------------------------------------------------------------------|
| `'global'`   | `thresh` (int)                                                                           |
| `'otsu'`     | (none)                                                                                   |
| `'adaptive'` | `block_size` (odd int), optional `adaptive_method` (`'mean'`/`'gaussian'`), `C` (offset) |

---

### `sobel(image) → (Gx, Gy, magnitude, direction)`

Sobel edge detection. Returns four arrays: horizontal gradient, vertical gradient, magnitude, and direction (radians). RGB images are auto-converted to grayscale.

---

### `bit_plane_slice(image, plane) → ndarray`

Extract bit plane `[0, 7]` from a `uint8` grayscale image. Returns binary array (0/1).

---

### `histogram(image) → ndarray`

256-bin histogram for `uint8` grayscale images.

---

### `equalize_histogram(image) → ndarray`

CDF-based histogram equalization for contrast enhancement.

---

### `laplacian_sharpen(image, strength=1.0) → ndarray`

Sharpen using the Laplacian operator. `sharpened = original + strength × Laplacian(original)`.

---

### `gamma_correction(image, gamma) → ndarray`

Power-law transform: `output = 255 × (input / 255) ^ gamma`. `gamma < 1` brightens; `gamma > 1` darkens.

---

## `shawwaf.transforms` — Geometric Transformations

### `resize(image, target_size, method='bilinear') → ndarray`

Resize to `(height, width)`. Methods: `'nearest'`, `'bilinear'`.

---

### `rotate(image, angle, interpolation='bilinear') → ndarray`

Rotate about center by `angle` degrees (CCW positive). Out-of-bounds filled with 0.

---

### `translate(image, tx, ty) → ndarray`

Shift image by `(tx, ty)` pixels. Vacated regions filled with 0.

---

## `shawwaf.features` — Feature Extractors

### `color_histogram_descriptor(image, bins=32) → ndarray`

L1-normalized color histogram. Output: `bins` (gray) or `3×bins` (RGB).

---

### `pixel_statistics_descriptor(image) → ndarray`

Per-channel statistics: mean, std, min, max, skewness, kurtosis. Output: 6 (gray) or 18 (RGB).

---

### `hog_descriptor(image, cell_size=8, bins=9) → ndarray`

Histogram of Oriented Gradients. L2-normalized. Output: `n_cells × bins`.

---

### `edge_orientation_histogram(image, bins=36) → ndarray`

Magnitude-weighted edge orientation histogram over [0°, 360°). L1-normalized.

---

## `shawwaf.drawing` — Drawing Primitives & Text

### `draw_point(image, x, y, color=255, thickness=1) → ndarray`

Draw a point. Out-of-bounds coordinates are clipped silently. Modifies in-place.

---

### `draw_line(image, x0, y0, x1, y1, color=255, thickness=1) → ndarray`

Line rasterization using Bresenham's algorithm. Modifies in-place.

---

### `draw_rectangle(image, x, y, w, h, color=255, thickness=1, filled=False) → ndarray`

Draw outline or filled rectangle. Boundary-clipped. Modifies in-place.

---

### `draw_polygon(image, points, color=255, thickness=1, filled=False) → ndarray`

Polygon from vertex list (≥3 points). Filled mode uses scanline rasterization. Modifies in-place.

---

### `draw_text(image, text, x, y, font_scale=1.0, color=255) → ndarray`

Render text via Matplotlib font rendering. Clipped at canvas boundaries. Modifies in-place.
