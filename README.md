# Shawwaf — Python Image Processing Library

**CSE480: Machine Vision | Milestone 1 | Spring 2026**

A from-scratch Python image-processing library that emulates a subset of OpenCV using only NumPy, Pandas, Matplotlib, and the Python standard library.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd shawwaf

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quickstart

```python
import shawwaf

# Read an image
img = shawwaf.io.read_image("photo.png")

# Convert to grayscale
gray = shawwaf.io.to_grayscale(img)

# Apply Gaussian blur
blurred = shawwaf.filtering.gaussian_filter(gray, size=5, sigma=1.5)

# Detect edges with Sobel
gx, gy, magnitude, direction = shawwaf.processing.sobel(gray)

# Threshold using Otsu's method
binary = shawwaf.processing.threshold(gray, method="otsu")

# Draw a rectangle
shawwaf.drawing.draw_rectangle(img, 10, 20, 100, 50, color=(255, 0, 0), thickness=2)

# Save result
shawwaf.io.save_image(img, "output.png")
```

## Package Structure

```
shawwaf/
├── shawwaf/                 # Importable package
│   ├── __init__.py
│   ├── io.py                # Image I/O & color conversion
│   ├── utils.py             # Normalization, clipping, padding, validation
│   ├── filtering.py         # Convolution, mean, Gaussian, median filters
│   ├── transforms.py        # Resize, rotate, translate
│   ├── features.py          # HOG, edge histogram, color histogram, pixel stats
│   ├── drawing.py           # Points, lines, rectangles, polygons, text
│   └── processing.py        # Threshold, Sobel, bit-plane, histogram, extras
│
├── tests/                   # Unit tests (pytest)
│   ├── test_io.py
│   ├── test_filtering.py
│   ├── test_processing.py
│   ├── test_transforms.py
│   ├── test_features.py
│   └── test_drawing.py
│
├── docs/
│   ├── api.md               # API reference
│   ├── math_notes.md        # Algorithm & math explanations
│   └── PRD.md               # Product requirements
│
├── notebooks/
│   └── demo.ipynb           # Demonstration notebook
│
├── requirements.txt
├── setup.py
└── README.md
```

## Modules Overview

| Module       | Key Functions                                                                                                       |
|--------------|---------------------------------------------------------------------------------------------------------------------|
| `io`         | `read_image`, `save_image`, `to_grayscale`, `to_rgb`                                                                |
| `utils`      | `normalize`, `clip`, `pad`                                                                                          |
| `filtering`  | `convolve2d`, `spatial_filter`, `mean_filter`, `gaussian_filter`, `median_filter`                                   |
| `processing` | `threshold`, `sobel`, `bit_plane_slice`, `histogram`, `equalize_histogram`, `laplacian_sharpen`, `gamma_correction` |
| `transforms` | `resize`, `rotate`, `translate`                                                                                     |
| `features`   | `color_histogram_descriptor`, `pixel_statistics_descriptor`, `hog_descriptor`, `edge_orientation_histogram`         |
| `drawing`    | `draw_point`, `draw_line`, `draw_rectangle`, `draw_polygon`, `draw_text`                                            |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.24
- Pandas ≥ 2.0
- Matplotlib ≥ 3.7

## License

Academic project — CSE480, Spring 2026.