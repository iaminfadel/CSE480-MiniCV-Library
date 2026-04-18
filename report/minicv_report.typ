// Cover Page
#set page(paper: "a4", margin: 1.0cm)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)

#align(center)[
  #image("assets/ASU_LOGO.png", width: 4cm)
  #v(2cm)

  #text(size: 20pt, weight: "bold")[
    Shawwaf: Image Processing Library from Scratch
  ]

  #text(size: 18pt, weight: "bold")[
    Comprehensive Implementation Report
  ]

  #v(1cm)

  #text(size: 14pt, weight: "bold")[
    CSE480: Machine Vision
  ]

  #text(size: 14pt)[
    Spring 2026 - Milestone 1
  ]

  #v(3cm)

  #text(size: 14pt, weight: "bold")[
    Submitted By:
  ]

  #v(0.5cm)

  #table(
    columns: (2fr, 1fr),
    align: (left, center),
    stroke: 0.5pt,
    fill: (_, row) => if calc.odd(row) { rgb("#f5f5f5") } else { white },
    inset: 8pt,
    table.header(
      [*Name*], [*Student ID*],
    ),
    [Amin Moustafa Fadel],          [2100483],
    [Ahmed Osama Ahmed Nasr],       [2101119],
    [Mohammed Tarek Abdelwahab],    [2100287],
  )

  #v(3cm)
]

#pagebreak()

// Table of Contents
#outline(
  title: [Table of Contents],
  indent: auto,
  depth: 2,
)

#pagebreak()

// List of Figures
#outline(
  title: [List of Figures],
  target: figure.where(kind: image),
)

// Main Content - Single Column Layout
#set page(
  margin: (left: 1.5cm, right: 1.5cm, top: 2cm, bottom: 2cm),
  columns: 1,
  numbering: "1",
  number-align: center,
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt)
      #smallcaps[Shawwaf Image Processing Library]
      #h(1fr)
      #counter(page).display("1")
      #line(length: 100%, stroke: 0.5pt)
    ]
  },
)

#set text(size: 9pt)
#set par(justify: true, leading: 0.7em, spacing: 1em)
#set heading(numbering: "1.1")

// Better heading styles
#show heading.where(level: 1): it => {
  v(0.8em)
  block(
    fill: rgb("#e8f4f8"),
    inset: 8pt,
    radius: 3pt,
    width: 100%,
    [
      #set text(size: 12pt, weight: "bold")
      #counter(heading).display()
      #h(0.5em)
      #it.body
    ],
  )
  v(0.8em)
}

#show heading.where(level: 2): it => {
  v(0.6em)
  block(
    inset: (left: 6pt),
    [
      #set text(size: 10pt, weight: "bold")
      #counter(heading).display()
      #h(0.5em)
      #it.body
    ],
  )
  v(0.4em)
}

#show heading.where(level: 3): it => {
  v(0.5em)
  block(
    inset: (left: 3pt),
    [
      #set text(size: 9.5pt, weight: "bold", style: "italic")
      #counter(heading).display()
      #h(0.5em)
      #it.body
    ],
  )
  v(0.3em)
}

// Important note boxes
#let note(content) = {
  block(
    fill: rgb("#fffbea"),
    stroke: (left: 2pt + rgb("#f59e0b")),
    inset: 8pt,
    radius: 2pt,
    width: 100%,
    [
      #set text(size: 8.5pt)
      #content
    ],
  )
}

// Key insight boxes
#let insight(content) = {
  block(
    fill: rgb("#eff6ff"),
    stroke: (left: 2pt + rgb("#3b82f6")),
    inset: 8pt,
    radius: 2pt,
    width: 100%,
    [
      #set text(size: 8.5pt)
      *Key Insight:* #content
    ],
  )
}

= Introduction

Image processing techniques constitute the initial phase of any computer vision pipeline. The dependency on highly optimized, monolithic libraries such as OpenCV abstracts the mathematical intricacies of the domain, distancing practitioners from fundamental algorithmic constraints.

Shawwaf bridges this educational gap by re-engineering these capabilities using an arbitrary baseline of Python constraints. The library achieves parity with low-level OpenCV abstractions while operating entirely atop NumPy multidimensional arrays.

== Objectives

The design and development of Shawwaf were grounded in the following core objectives:

1. *Algorithm Mathematical Fidelity*: Demonstrating exact transformations without reliance on black-box wrappers.
2. *Architectural Modularity*: Establishing a package structure suitable for independent domain modules (I/O, Filtering, Transforms).
3. *Computational Efficiency*: Prioritizing array vectorization over scalar loops.
4. *Strict Input Validation*: Guaranteeing API robustness through pervasive assertion handling.

= Library Architecture

== Package Structure

Shawwaf adopts a traditional flat-package architecture conducive to straightforward imports:

```text
Shawwaf/
├── __init__.py
├── io.py             (Image read/write, formats)
├── utils.py          (Shared validation, padding)
├── filtering.py      (Convolution, spatial kernels)
├── processing.py     (Thresholds, histograms)
├── transforms.py     (Geometric manipulations)
├── features.py       (Feature descriptors)
└── drawing.py        (Rasterized geometric primitives)
```

== Core Abstraction: Vectorized Cross-Correlation

At the heart of any image processing library lies spatial filtering. Shawwaf implements its core kernel convolution mathematically analogous to OpenCV's `filter2D`.

For an input image $I$ and kernel $K$ of dimensions $(2k+1) times (2k+1)$, the linear convolution is bounded as:

$ S(x,y) = sum_(i=-k)^k sum_(j=-k)^k I(x-i, y-j) K(i, j) $

Unlike naive element-wise iteration mapping, Shawwaf enforces this mathematically through highly optimized view-based strides using NumPy array mechanics. Memory boundaries are secured via configurable padding mechanisms (Zero, Reflect, Edge Replicate).

#figure(
  image("assets/padding_demo.png", width: 100%),
  caption: [Demonstration of core padding techniques (Zero, Reflect, Replicate) utilized to maintain spatial dimensions during convolution.],
) <fig:padding>

= Spatial Filtering

== Mean, Gaussian, and Median

The `Shawwaf.filtering` module exposes fundamental discrete convolutions essential to noise reduction.

- *Gaussian Filtering*: Generated via numerical sampling of the 2D Gaussian distribution, preserving image energy.
$ G(x,y) = frac(1, 2 pi sigma^2) e^(-frac(x^2 + y^2, 2 sigma^2)) $

- *Median Filtering*: An inherently non-linear morphological operation leveraging sliding-window percentile extraction to eradicate salt-and-pepper impulses without compromising edge profiles.

#figure(
  image("assets/filters_comparison.png", width: 100%),
  caption: [Comparison of spatial filtering techniques applied to synthetic noise topologies.],
) <fig:filters>

= Advanced Pixel Processing

== Sobel Gradient Estimation

The `sobel()` algorithm extracts high-frequency structural changes by convolving discrete horizontal and vertical derivative approximations.

Gradient $G_x$ and $G_y$ are isolated to formulate the orientation magnitude:
$ |G| = sqrt(G_x^2 + G_y^2) $

#figure(
  image("assets/sobel_demo.png", width: 100%),
  caption: [Horizontal, Vertical, and Absolute Sobel gradient approximations isolating geometric edge features.],
) <fig:sobel>

== Adaptive and Global Thresholding

Shawwaf segregates pixels by generating binary masks via various evaluation constraints:

- *Global Formulation*: Defined against hard-coded luminance bounds.
- *Otsu's Method*: Procedurally evaluates discrete histogram variance to maximize inter-class separation.

#figure(
  image("assets/threshold_demo.png", width: 100%),
  caption: [Binarization applied over a uniform intensity gradient comparing static versus dynamically calculated Otsu thresholds.],
) <fig:thresholds>

== Histogram Equalization & Bit Plane Slicing

Contrast enhancement operations interact directly with pixel intensity distributions. Histogram Equalization flattens the cumulative distribution function (CDF) of pixel intensities, spreading local contrast across the available dynamic range.

#figure(
  image("assets/histogram_demo.png", width: 100%),
  caption: [Histogram Equalization improving contrast dynamics on a compressed-range synthetic image.],
) <fig:histogram>

Furthermore, the integration of bit plane slicing isolates discrete intensity vectors, segmenting structural boundaries and high-frequency noise distributed across numerical exponents.

#figure(
  image("assets/bitplane_demo.png", width: 100%),
  caption: [Sequential decomposition of an image into its most significant bit planes.],
) <fig:bitplane>

== Additional Rendering Techniques

To extend the baseline curriculum constraints, Shawwaf employs optional filtering pipelines including Laplacian sharpening to emphasize high-frequency edges, alongside parameterized Gamma correction for non-linear luminance mapping.

#figure(
  image("assets/extras_demo.png", width: 100%),
  caption: [Demonstration of Laplacian edge sharpening and Gamma correction ($gamma=2.0$).],
) <fig:extras>

= Feature Extraction Modules

Vector descriptors collapse arbitrary pixel spaces into statistically significant, low-dimensionality arrays for downstream classification algorithms. Shawwaf implements both Global and Gradient-based local feature extraction.

- *Global Extraction*: Captures macroscopic tone properties via normalized color histograms and central pixel statistics (Mean, Variance).
- *Gradient Extraction*: Evaluates regional complexity scaling via structural Histogram of Oriented Gradients (HOG) and Edge Orientation analysis.

#figure(
  image("assets/features_demo.png", width: 100%),
  caption: [Feature vectors extracted by Histogram of Oriented Gradients (HOG) and Edge Orientation modules.],
) <fig:features>

= Geometric Transformations

Image alignment, resizing, and warping are handled via the `transforms.py` module, utilizing configurable coordinate interpolations (Nearest-Neighbor, Bilinear).

#insight[
  Image rotation fundamentally breaks grid alignments. Instead of projecting input pixels to irrational output float topologies, Shawwaf projects an inverse coordinate map utilizing homographic rotation matrices centered on the origin.
]

#figure(
  image("assets/transforms_demo.png", width: 100%),
  caption: [Demonstration of inverse-mapped geometric interpolation retaining structure post-transformation.],
) <fig:transforms>

= Raster Drawing Primitives

The library features from-scratch geometry rasterization manipulating pixel regions in direct mutable memory space. Operations such as lines, rectangles, and complex polygons rely upon generalized algebraic bounding geometries. Bresenham's line algorithm guarantees aliased continuous rendering uncoupled from floating-point inaccuracy.

#figure(
  image("assets/drawing_demo.png", width: 100%),
  caption: [Canvas demonstrating discrete geometry intersection and rasterization rendering.],
) <fig:drawing>

= Conclusion

The Shawwaf project achieves robust equivalency with essential elements of traditional machine vision constraints. The constraints on utilizing Python loop structures fundamentally compelled the application of high-precision numerical techniques across the NumPy API.

The resulting framework operates as an educational testament to underlying CV engineering, fully documented and validated against extensive edge case validation pipelines. Future expansion avenues include morphological erosion algorithms, integral image optimization maps, and Hough space detection transforms.

= Repository & Implementation

The repository encapsulating all functional domains, continuous testing, testing scripts, and visualization notebooks is located at:

#link("https://github.com/iaminfadel/Shawwaf")

102 unit tests validate matrix boundary behaviors, dimension mismatch handling, and dtype assertion.
