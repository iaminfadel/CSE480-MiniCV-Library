# Design Documents — End-to-End Machine Vision Pipeline
## CSE480: Machine Vision | Milestone 2 | Spring 2026

---

## Table of Contents

1. [Epic 1 — Dataset](#epic-1--dataset)
2. [Epic 2 — Preprocessing](#epic-2--preprocessing)
3. [Epic 3 — Augmentation](#epic-3--augmentation)
4. [Epic 4 — Feature Extraction](#epic-4--feature-extraction)
5. [Epic 5 — Model Training](#epic-5--model-training)
6. [Epic 6 — Optimizer](#epic-6--optimizer)
7. [Epic 7 — Logging & Checkpointing](#epic-7--logging--checkpointing)
8. [Epic 8 — Evaluation & Comparison](#epic-8--evaluation--comparison)
9. [Epic 9 — Documentation](#epic-9--documentation)

---

## Epic 1 — Dataset

### Purpose
Define the structure, format, and loading contract for the image dataset that feeds the entire pipeline.

### Module: `pipeline/dataset.py`

### Dataset Requirements

| Property | Requirement |
|---|---|
| Minimum classes | 6 |
| Intra-class variability | Lighting, background, viewpoint/pose |
| Class balance | No class > 3× any other class in sample count |
| Annotation format | `annotations.csv` with `filepath`, `label` columns |
| Split ratio | 70% train / 15% val / 15% test (configurable) |
| Reproducibility | Fixed random seed for split generation |

### Directory Layout

```
data/
├── raw/
│   ├── class_a/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── class_b/
│       └── ...
├── annotations.csv          # filepath, label
└── splits/
    ├── train.csv            # subset of annotations.csv
    ├── val.csv
    └── test.csv
```

### `annotations.csv` Schema

```
filepath,label
data/raw/class_a/img_001.jpg,class_a
data/raw/class_a/img_002.jpg,class_a
data/raw/class_b/img_001.jpg,class_b
...
```

### Key Functions

#### `build_annotations(root_dir) -> pd.DataFrame`
Walks `data/raw/`, collects all image paths, infers label from parent folder name, and writes `annotations.csv`.

#### `split_dataset(annotations_df, train_ratio, val_ratio, seed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

**Stratified split algorithm:**
```
1. Group rows by label
2. For each label group:
       shuffle with fixed seed
       take first train_ratio fraction → train
       next val_ratio fraction         → val
       remainder                       → test
3. Concatenate across label groups
4. Shuffle each split with fixed seed
5. Write train.csv, val.csv, test.csv
```
Stratified splitting preserves class distribution across all three sets.

#### `ImageDataset(split_csv, transform=None)`

Iterable class that:
- Reads the split CSV.
- Loads each image via `shawwaf.io.read_image()`.
- Applies an optional `transform` callable (preprocessing + augmentation pipeline).
- Returns `(image_array, label_int)` pairs.

**Label encoding:** Labels are mapped to integers via a sorted `label_to_idx` dict, committed alongside the splits as `data/splits/label_map.json`.

---

## Epic 2 — Preprocessing

### Purpose
Normalize all images to a fixed spatial size and consistent pixel value range before any feature extraction or model input.

### Module: `pipeline/preprocessing.py`

### Design

```
raw image (H, W, 3) uint8
        │
        ▼
   shawwaf.resize(target_size, method='bilinear')
        │
        ▼
   shawwaf.normalize(mode='minmax')  →  float32 [0, 1]
        │
        ▼
preprocessed image (TARGET_H, TARGET_W, 3) float32
```

### Configuration Constants

```python
TARGET_SIZE  = (128, 128)   # (height, width) — configurable
NORM_MODE    = 'minmax'     # justified: CNN and feature extractors expect [0,1]
NORM_STATS   = 'train_only' # statistics computed from training set only
```

### `preprocess(image: np.ndarray) -> np.ndarray`

```python
def preprocess(image):
    image = shawwaf.resize(image, TARGET_SIZE, method='bilinear')
    image = shawwaf.normalize(image, mode=NORM_MODE)
    return image
```

**Critical rule:** Normalization statistics (min, max for minmax; μ, σ for zscore) must be computed on the training set and stored, then applied to val and test without recomputation.

---

## Epic 3 — Augmentation

### Purpose
Artificially expand the training set by applying random image transforms via shawwaf, improving model generalization without collecting more data.

### Module: `pipeline/augmentation.py`

### Design: Composable Transform Pipeline

```python
class Compose:
    def __init__(self, transforms): ...
    def __call__(self, image): 
        for t in self.transforms:
            image = t(image)
        return image
```

Each transform is a callable class with a configurable `probability` parameter.

### Required Transforms (≥ 5)

| Transform | shawwaf Call | Parameters |
|---|---|---|
| `RandomHorizontalFlip` | `image[:, ::-1]` (array slice) | `p=0.5` |
| `RandomRotation` | `shawwaf.rotate(image, angle)` | `max_angle=15`, `p=0.5` |
| `RandomTranslation` | `shawwaf.translate(image, tx, ty)` | `max_shift=10px`, `p=0.5` |
| `GaussianNoise` | Add `np.random.normal` noise, then `shawwaf.clip` | `sigma=0.02`, `p=0.3` |
| `RandomBrightnessJitter` | `shawwaf.normalize` after scaling | `factor_range=(0.7, 1.3)`, `p=0.5` |
| `RandomGaussianBlur` | `shawwaf.gaussian_filter(image, size, sigma)` | `sigma_range=(0.5, 1.5)`, `p=0.3` |

**Application rule:** Augmentation is instantiated inside `ImageDataset` only for the training split:

```python
train_transform = Compose([preprocess, augmentation_pipeline])
val_transform   = Compose([preprocess])   # no augmentation
test_transform  = Compose([preprocess])
```

### Before/After Panel Generation

A utility function `plot_augmentation_panel(image, transforms)` applies each transform independently to the same source image and produces a grid saved to `docs/results/augmentation_panel.png`.

---

## Epic 4 — Feature Extraction

### Purpose
Transform preprocessed images into fixed-length numerical vectors suitable for classical ML classifiers. Feature selection then reduces the vector to its most informative subset.

### Module: `pipeline/features.py`

### Feature Pool Design

All extraction calls shawwaf's feature API. No raw NumPy image manipulation allowed.

| Family | Extractor | Output Length | shawwaf Call |
|---|---|---|---|
| Color | RGB histogram (32 bins/channel) | 96 | `shawwaf.features.extract_color_histogram` |
| Statistics | Per-channel moments (mean, std, skew, kurt) | 12 | `shawwaf.features.extract_statistical_moments` |
| Gradient | Gradient magnitude histogram (32 bins) | 32 | `shawwaf.features.extract_gradient_histogram` |
| Texture | Simplified HOG (8×8 cells, 9 bins) | variable | `shawwaf.features.extract_hog` |

**Total default dimensionality:** documented in `FEATURE_INDEX` dict.

### `FEATURE_INDEX` Scheme

```python
FEATURE_INDEX = {
    'color_histogram':     (0,   96),
    'statistical_moments': (96,  108),
    'gradient_histogram':  (108, 140),
    'hog':                 (140, 140 + HOG_DIM),
}
```

### `extract_all(image) -> np.ndarray`

```python
def extract_all(image):
    gray = shawwaf.to_grayscale(image)
    v = np.concatenate([
        shawwaf.features.extract_color_histogram(image),
        shawwaf.features.extract_statistical_moments(image),
        shawwaf.features.extract_gradient_histogram(gray),
        shawwaf.features.extract_hog(gray),
    ])
    return v.astype(np.float32)
```

### MRMR Feature Selection

**What MRMR selects:** The top-K features that maximize relevance to the target class label while minimizing redundancy among selected features.

**Criterion (mutual information formulation):**
```
score(f) = MI(f; y) − (1/|S|) · Σ MI(f; s)  for s in already-selected set S
```

**Usage:**
```python
from mrmr import mrmr_classif   # only allowed library call in pipeline

selected_indices = mrmr_classif(
    X=pd.DataFrame(X_train),
    y=pd.Series(y_train),
    K=K
)
```

**K is configurable** (default: 64). The indices of selected features are logged to `logs/<run_id>/mrmr_selected.json` for traceability.

**All downstream models train on `X[:, selected_indices]`, not the full vector.**

---

## Epic 5 — Model Training

### Purpose
Implement four classifiers with increasing complexity, all operating on the MRMR-selected feature vectors (except the CNN which operates on raw image patches or pooled features as designed).

### Module: `pipeline/models/`

---

### 5.1 KNN — `pipeline/models/knn.py`

#### Design

```python
class KNN:
    def fit(self, X_train, y_train): ...     # store training data
    def predict(self, X_test, k): ...        # return predicted labels
    def k_sweep(self, X_val, y_val, k_range): ...  # return {k: val_accuracy}
```

#### Distance Metric

**Euclidean (default):**
```python
# Vectorized: no Python loop over samples
# X_test: (N_test, D), X_train: (N_train, D)
diffs = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]   # (N_test, N_train, D)
dists = np.sqrt(np.sum(diffs**2, axis=-1))                      # (N_test, N_train)
```

**Cosine (optional):**
```
d(a, b) = 1 − (a · b) / (‖a‖ · ‖b‖)
```

#### k-Sweep

```
for k in k_range:
    preds = KNN.predict(X_val, k)
    acc[k] = accuracy(preds, y_val)
best_k = argmax(acc)
```

---

### 5.2 Softmax Regression — `pipeline/models/softmax.py`

#### Architecture

```
Input x ∈ ℝᴰ  →  Linear(D → C)  →  Softmax  →  ŷ ∈ ℝᶜ
```

Where C = number of classes.

#### Forward Pass

```
Z = X @ W + b                   # (N, C)
P = softmax(Z)                  # (N, C), numerically stable
L = −(1/N) Σ log(P[i, y[i]] + ε)   # scalar, ε = 1e-9
```

**Numerically stable softmax:**
```python
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   # subtract row max
    exp_Z = np.exp(Z)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)
```

#### Backward Pass

```
dL/dZ = (P − one_hot(y)) / N
dL/dW = Xᵀ @ dL/dZ  +  λ·W     # L2 regularization term
dL/db = sum(dL/dZ, axis=0)
```

#### Training Loop

```
for epoch in range(max_epochs):
    shuffle(X_train, y_train)
    for batch in minibatches(X_train, y_train, batch_size):
        forward → compute loss
        backward → compute grads
        clip grads
        optimizer.step(W, b, grads)
    compute val_loss, val_acc
    lr_schedule.step()
    early_stopping.check(val_loss)
    logger.log(epoch, train_loss, val_loss, ...)
    checkpointer.save_if_best(val_loss)
```

---

### 5.3 CNN — `pipeline/models/cnn.py`

#### Architecture

```
Input (N, C, H, W)
    → Conv2D(in_ch, out_ch, kernel=3, pad=1)
    → ReLU
    → MaxPool2D(kernel=2, stride=2)
    → Conv2D(out_ch, out_ch*2, kernel=3, pad=1)
    → ReLU
    → MaxPool2D(kernel=2, stride=2)
    → Flatten
    → Linear(D_flat → 128)
    → ReLU
    → Linear(128 → C)
    → Softmax + Cross-Entropy Loss
```

All layer dimensions are configurable via constructor arguments.

#### Layer Interface Contract

Each layer implements:
```python
class Layer:
    def forward(self, x) -> np.ndarray: ...
    def backward(self, grad_out) -> np.ndarray: ...
    def params(self) -> list[np.ndarray]: ...
    def grads(self) -> list[np.ndarray]: ...
```

#### Conv2D Forward

```
For each sample n, each output channel f, each position (i, j):
    out[n, f, i, j] = sum(W[f] * in[n, :, i·s:i·s+kH, j·s:j·s+kW]) + b[f]
```

Implemented via `np.lib.stride_tricks` — no Python loop over spatial positions.

#### Conv2D Backward

```
dL/dW[f]   = sum over (n,i,j) of dL/dOut[n,f,i,j] * in_patch[n,:,i,j]
dL/dIn     = full convolution of dL/dOut with flipped W (transposed convolution)
dL/db[f]   = sum(dL/dOut[:, f, :, :])
```

#### MaxPool2D Forward & Backward

```
Forward:  out[n,c,i,j] = max(in[n,c, i*s:i*s+k, j*s:j*s+k])
Backward: gradient flows only to the max-position element (mask-based)
```

---

### 5.4 Research Paper Architecture — `pipeline/models/paper_model.py`

#### Selection Criteria

Choose one architecture from a paper published 2021–2026. Recommended candidates:

| Paper | Year | Why Suitable |
|---|---|---|
| ConvMixer | 2022 | Simple depthwise + pointwise design |
| MobileViT | 2022 | Lightweight hybrid CNN-Transformer |
| EfficientNetV2 | 2021 | Compound scaling, small variant |

#### Implementation Rules

- Framework (PyTorch / TensorFlow / Keras) is allowed for model definition and training loop.
- All dataset loading, preprocessing, and augmentation still go through shawwaf.
- Paper citation: title, authors, year, venue/arXiv ID — committed in `docs/math_notes.md`.
- Documented deviations from the original paper (e.g., reduced depth for compute budget) are noted.

---

## Epic 6 — Optimizer

### Purpose
Implement a reusable optimizer system shared by Softmax Regression and CNN-from-scratch, with configurable LR schedules and training safety features.

### Module: `pipeline/optimizers.py`

### Optimizer Interface

```python
class Optimizer:
    def step(self, params: list, grads: list) -> None: ...
    def state_dict(self) -> dict: ...         # for checkpointing
    def load_state_dict(self, d: dict): ...   # for resuming
```

---

### SGD

```
w ← w − lr · grad
```

---

### Adam

```
m ← β₁·m + (1−β₁)·grad           # first moment
v ← β₂·v + (1−β₂)·grad²          # second moment
m̂ = m / (1−β₁ᵗ)                  # bias correction
v̂ = v / (1−β₂ᵗ)
w ← w − lr · m̂ / (√v̂ + ε)
```

Default: `β₁=0.9, β₂=0.999, ε=1e-8`.

---

### LR Schedules

| Schedule | Formula | Parameters |
|---|---|---|
| Step decay | `lr = lr₀ · γ^⌊epoch/step⌋` | `gamma`, `step_size` |
| Exponential | `lr = lr₀ · γ^epoch` | `gamma` |
| Cosine | `lr = lr_min + 0.5(lr₀−lr_min)(1 + cos(πt/T))` | `T`, `lr_min` |
| Reduce-on-plateau | Halve lr if val_loss stagnates for `patience` epochs | `patience`, `factor` |

---

### Safety Features

#### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience): ...
    def check(self, val_loss) -> bool:
        # returns True (stop) if val_loss hasn't improved for `patience` epochs
```

#### Gradient Clipping
```python
# Applied before optimizer.step()
total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
if total_norm > max_norm:
    scale = max_norm / (total_norm + 1e-6)
    grads = [g * scale for g in grads]
```

#### L2 Regularization
Added to the loss before backprop:
```
L_total = L_CE + (λ/2) · Σ ‖W‖²
dL/dW  += λ · W
```

#### Mini-Batch Shuffling
```python
# At start of each epoch:
perm = np.random.permutation(N)
X_train, y_train = X_train[perm], y_train[perm]
```

---

## Epic 7 — Logging & Checkpointing

### Purpose
Produce fully reproducible, resumable training runs with structured per-epoch logs and best-model checkpoints.

### Module: `pipeline/trainer.py`

### Run ID Convention
Each training run gets a unique ID: `<model_name>_<timestamp>` (e.g., `cnn_20260415_143022`). All artifacts for a run are stored under `logs/<run_id>/` and `checkpoints/<run_id>/`.

---

### `logs.csv` Schema

```
epoch,train_loss,val_loss,train_acc,val_acc,learning_rate
1,2.3104,2.2891,0.1823,0.1901,0.001
2,2.1045,2.0912,0.2341,0.2289,0.001
...
```

**Append-only:** When resuming, new rows are appended. No row is overwritten.

---

### Checkpoint Format

```
checkpoints/<run_id>/
├── best.npz          # model weights (numpy arrays, keyed by layer name)
├── optimizer.json    # optimizer state (moments, step count)
└── config.json       # full run configuration
```

`config.json` example:
```json
{
  "model": "softmax",
  "lr": 0.001,
  "batch_size": 32,
  "optimizer": "adam",
  "lambda_l2": 1e-4,
  "max_norm": 5.0,
  "patience": 10,
  "K_mrmr": 64,
  "target_size": [128, 128],
  "seed": 42
}
```

---

### Resume Protocol

```
1. Load config.json → restore all hyperparameters
2. Load best.npz    → restore model weights
3. Load optimizer.json → restore optimizer state (moments, step)
4. Read last row of logs.csv → determine start_epoch
5. Continue training loop from start_epoch + 1
6. Append to existing logs.csv
```

---

### `Trainer` Class Interface

```python
class Trainer:
    def __init__(self, model, optimizer, config, run_id): ...
    def train(self, train_loader, val_loader): ...
    def save_checkpoint(self, val_loss): ...
    def load_checkpoint(self, checkpoint_dir): ...
```

---

## Epic 8 — Evaluation & Comparison

### Purpose
Compute all evaluation metrics from scratch on the held-out test set and produce a structured cross-model comparison.

### Module: `pipeline/evaluator.py`

### Metric Implementations (all from scratch, no sklearn)

#### Confusion Matrix
```python
def confusion_matrix(y_true, y_pred, n_classes) -> np.ndarray:
    C = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C
```

#### Per-Class Precision, Recall, F1
```
TP[c] = C[c, c]
FP[c] = C[:, c].sum() − TP[c]
FN[c] = C[c, :].sum() − TP[c]

Precision[c] = TP[c] / (TP[c] + FP[c] + ε)
Recall[c]    = TP[c] / (TP[c] + FN[c] + ε)
F1[c]        = 2 · Precision[c] · Recall[c] / (Precision[c] + Recall[c] + ε)
```

#### Macro-F1 and Weighted-F1
```
Macro-F1    = (1/C) · Σ F1[c]
Weighted-F1 = Σ (support[c] / N) · F1[c]
```

Where `support[c]` = number of true instances of class c.

---

### Cross-Model Comparison Table

Generated by `evaluator.compare_models(results_dict)`:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Inference Time (ms/sample) |
|---|---|---|---|---|
| KNN (best k) | | | | |
| Softmax Regression | | | | |
| CNN (from scratch) | | | | |
| Paper Architecture | | | | |

---

### Visualization Outputs

| Output | Path | Tool |
|---|---|---|
| Confusion matrix heatmap × 4 | `docs/results/cm_<model>.png` | Matplotlib `imshow` |
| Per-class F1 bar chart × 4 | `docs/results/f1_<model>.png` | Matplotlib `bar` |
| Training curves (loss + acc) | `docs/results/curves_<model>.png` | Matplotlib `plot` |
| Cross-model comparison table | `docs/results/comparison.md` | Markdown table |

---

## Epic 9 — Documentation

### `docs/math_notes.md` — Required Sections

| Section | Content |
|---|---|
| Dataset splits | Stratified split derivation |
| MRMR | Mutual information criterion formula |
| Numerically stable softmax | Derivation of the max-shift trick |
| Cross-entropy + epsilon clipping | Formula and why ε prevents log(0) |
| Backpropagation (Conv2D) | Chain rule derivation for dL/dW and dL/dIn |
| Backpropagation (MaxPool) | Gradient mask explanation |
| Adam optimizer | Full update rule with bias correction |
| Gradient clipping | L2 norm clipping formula |
| L2 regularization | Loss augmentation and weight update effect |
| LR schedules | All four schedule formulas with plots |
| Evaluation metrics | Precision, recall, F1, macro, weighted derivations |
| Paper architecture | Paper citation + key design decisions |

### `docs/results/` — Required Artifacts

| File | Content |
|---|---|
| `class_distribution.png` | Dataset class balance bar chart |
| `augmentation_panel.png` | Before/after panel for all 5+ transforms |
| `feature_importance.png` | MRMR-selected feature index bar chart |
| `curves_softmax.png` | Train/val loss + accuracy curves |
| `curves_cnn.png` | Train/val loss + accuracy curves |
| `curves_paper.png` | Train/val loss + accuracy curves |
| `cm_knn.png` | Confusion matrix heatmap |
| `cm_softmax.png` | Confusion matrix heatmap |
| `cm_cnn.png` | Confusion matrix heatmap |
| `cm_paper.png` | Confusion matrix heatmap |
| `comparison.md` | Cross-model metrics table |

---

## Cross-Epic Dependency Summary

```
Epic 1 (Dataset)
    └─► Epic 2 (Preprocessing)
            └─► Epic 3 (Augmentation)
                    └─► Epic 4 (Feature Extraction)
                            ├─► Epic 5.1 (KNN)
                            ├─► Epic 5.2 (Softmax) ──► Epic 6 (Optimizer)
                            ├─► Epic 5.3 (CNN)      ──► Epic 6 (Optimizer)
                            └─► Epic 5.4 (Paper)
                                    │
                            Epic 6 (Optimizer) ──► Epic 7 (Logging)
                                                         │
                                                   Epic 8 (Evaluation)
                                                         │
                                                   Epic 9 (Docs)
```

**Recommended Implementation Order:**
1. Epic 1 — build and validate the dataset
2. Epic 2 → 3 — preprocessing and augmentation
3. Epic 4 — feature extraction and MRMR selection
4. Epic 6 — optimizer (shared dependency for Epics 5.2 and 5.3)
5. Epic 7 — logging and checkpointing (shared by all trainable models)
6. Epic 5.1 — KNN (no optimizer needed, simplest baseline)
7. Epic 5.2 — Softmax Regression
8. Epic 5.3 — CNN from scratch
9. Epic 5.4 — Paper architecture
10. Epic 8 — evaluation (after all models are trained)
11. Epic 9 — documentation (written throughout, finalized last)
