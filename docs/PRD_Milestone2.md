# Product Requirements Document
## shawwaf — End-to-End Machine Vision Pipeline
### CSE480: Machine Vision | Milestone 2 | Spring 2026

---

## Overview

Milestone 2 builds a complete supervised machine vision pipeline on top of the shawwaf library from Milestone 1. It spans dataset preparation, preprocessing, augmentation, feature extraction, model training, and evaluation. All image processing must use the shawwaf library; NumPy is allowed only for model math and general array operations. A deep learning framework (PyTorch / TensorFlow / Keras) is permitted exclusively for the research paper benchmark architecture.

---

## Goals

- Build a reproducible, end-to-end image classification pipeline.
- Implement four distinct classifiers (KNN, Softmax Regression, CNN, Paper Architecture) from scratch or with a justified framework.
- Demonstrate rigorous model evaluation and fair comparison across all classifiers.
- Produce fully resumable training runs with structured logging and checkpointing.

---

## Non-Goals

- No image I/O, filtering, or transformation logic reimplemented outside shawwaf.
- No pre-trained model weights or transfer learning (except as defined by the chosen paper architecture).
- No deployment, serving, or inference API beyond local evaluation scripts.

---

## Project Structure

The repository must follow the same flat layout as Milestone 1, extended with pipeline-specific folders.

```
shawwaf-repo/
├── shawwaf/                        # shawwaf library (Milestone 1 — unchanged)
│
├── pipeline/                      # Milestone 2 source code
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loading, annotation parsing, train/val/test split
│   ├── preprocessing.py           # Resize, normalize
│   ├── augmentation.py            # 5+ augmentation transforms
│   ├── features.py                # Feature extraction, concatenation, MRMR selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── knn.py                 # KNN from scratch
│   │   ├── softmax.py             # Softmax regression from scratch
│   │   ├── cnn.py                 # CNN from scratch
│   │   └── paper_model.py         # Research paper architecture (framework allowed)
│   ├── optimizers.py              # SGD, advanced optimizer, LR schedule
│   ├── trainer.py                 # Training loop, logging, checkpointing
│   └── evaluator.py               # Accuracy, confusion matrix, precision/recall/F1
│
├── data/                          # Dataset root
│   ├── raw/                       # Original downloaded images
│   ├── annotations.csv            # Ground-truth class labels for every image
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── checkpoints/                   # Saved model weights and optimizer states
│
├── logs/                          # Per-run logs.csv files
│
├── tests/                         # Unit & integration tests
│   ├── test_dataset.py
│   ├── test_augmentation.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_evaluator.py
│
├── docs/                          # Project documentation
│   ├── api.md
│   ├── math_notes.md
│   └── results/                   # Output plots, confusion matrices, comparison tables
│
├── notebooks/
│   └── demo_pipeline.ipynb
│
├── README.md
├── requirements.txt
└── setup.py
```

---

## User Stories

---

### Epic 1 — Dataset

**US-1.1 — Dataset Composition**
> As a researcher, I want the dataset to contain at least 6 classes with strong intra-class variability and a roughly balanced class distribution, so that the models face a realistic and fair classification challenge.

**Acceptance Criteria:**
- Minimum 6 distinct classes are present.
- Each class exhibits variability across lighting, background, and viewpoint/pose.
- Class sample counts are approximately balanced; no single class dominates by more than 3×.

---

**US-1.2 — Class Distribution Plot**
> As a reviewer, I want a class distribution bar chart provided, so that I can immediately verify dataset balance at a glance.

**Acceptance Criteria:**
- A bar chart showing sample count per class is committed to `docs/results/`.
- The chart is generated programmatically and reproducible via a script or notebook cell.

---

**US-1.3 — Ground-Truth Annotation File**
> As a developer, I want every image to have a ground-truth label stored in a dedicated annotation file, so that the dataset is fully self-contained and reproducible.

**Acceptance Criteria:**
- `data/annotations.csv` exists with at minimum columns: `filepath`, `label`.
- Every image in the dataset has exactly one corresponding row.
- Labels are consistent strings (no mixed int/string representations).

---

**US-1.4 — Dataset Splits**
> As a developer, I want the dataset split into training, validation, and test sets with no leakage between them, so that model evaluation reflects true generalization.

**Acceptance Criteria:**
- `data/splits/train.csv`, `val.csv`, and `test.csv` are generated and committed.
- No image appears in more than one split.
- Split proportions are documented (e.g., 70/15/15).
- Splits are generated with a fixed random seed for reproducibility.

---

### Epic 2 — Preprocessing

**US-2.1 — Resize to Fixed Size**
> As a developer, I want all images resized to a single fixed spatial resolution before any model sees them, so that feature vectors and model inputs are consistently shaped.

**Acceptance Criteria:**
- `preprocess(image, target_size)` resizes every image to `target_size` using shawwaf's `resize()`.
- Target size is a single configurable constant documented in the pipeline config.
- Applied to training, validation, and test splits identically.

---

**US-2.2 — Normalization**
> As a developer, I want pixel values normalized before model input, with the normalization strategy justified, so that model training is numerically stable.

**Acceptance Criteria:**
- Normalization is applied using shawwaf's `normalize()`.
- The chosen mode (`minmax`, `zscore`, etc.) is documented with a justification (e.g., "minmax chosen because CNN inputs expect [0,1]").
- Applied identically to all splits using statistics computed from the training set only.

---

### Epic 3 — Augmentation

**US-3.1 — Training-Only Augmentation**
> As a developer, I want image augmentation applied exclusively to training samples, so that validation and test evaluation remain unbiased.

**Acceptance Criteria:**
- Augmentation functions are called only within the training data loader path.
- Validation and test images pass through preprocessing only (no augmentation).

---

**US-3.2 — Minimum Five Augmentation Transforms**
> As a developer, I want at least 5 distinct augmentation transforms implemented using shawwaf, so that the model sees sufficient variety to generalize.

**Acceptance Criteria:**
- At least 5 transforms are implemented (e.g., horizontal flip, rotation, brightness jitter, Gaussian noise, crop-and-resize, translation).
- Every transform is implemented via shawwaf library calls — no raw NumPy image manipulation.
- Each transform has a configurable probability of being applied per image.

---

**US-3.3 — Before/After Augmentation Panels**
> As a reviewer, I want before/after image panels showing the effect of each augmentation transform, so that I can verify correctness visually.

**Acceptance Criteria:**
- A panel showing the original image alongside each augmented variant is committed to `docs/results/`.
- Panel covers all 5+ transforms.

---

### Epic 4 — Feature Extraction

**US-4.1 — Multi-Family Feature Pool**
> As a developer, I want at least 3 distinct feature families extracted per image and concatenated into a single vector, so that the feature representation is rich enough for downstream classifiers.

**Acceptance Criteria:**
- Minimum 3 distinct feature families are used (e.g., color histogram, statistical moments, HOG, gradient histogram, LBP).
- All extraction is performed via shawwaf's feature API — no raw NumPy image operations.
- The concatenated vector's total dimensionality is documented.

---

**US-4.2 — Feature Naming & Indexing Scheme**
> As a developer, I want a clear naming and index map for the concatenated feature vector, so that selected features can be traced back to their source descriptor.

**Acceptance Criteria:**
- A `FEATURE_INDEX` dictionary (or equivalent) documents the start/end slice for each feature family in the full vector.
- This mapping is committed as part of `pipeline/features.py` and referenced in `docs/math_notes.md`.

---

**US-4.3 — MRMR Feature Selection**
> As a developer, I want MRMR applied to select the top K features from the full pool, so that redundant and irrelevant features are removed before model training.

**Acceptance Criteria:**
- An MRMR library may be used for the selection step only; feature extraction still uses shawwaf.
- K is a configurable hyperparameter documented in the pipeline config.
- The identity of the selected top-K features is logged (by name/index).
- Models are trained on the MRMR-selected feature subset, not the full vector.

---

**US-4.4 — Before/After Feature Selection Panels**
> As a reviewer, I want visual panels comparing feature distributions before and after MRMR selection, so that the effect of feature selection is demonstrable.

**Acceptance Criteria:**
- A panel (e.g., feature importance bar chart or distribution plot) is committed to `docs/results/`.

---

### Epic 5 — Model Training

**US-5.1 — KNN Classifier (From Scratch)**
> As a developer, I want a KNN classifier implemented from scratch with a configurable distance metric and k-sweep, so that I have a non-parametric baseline.

**Acceptance Criteria:**
- Distance metric (e.g., Euclidean, cosine) is implemented in NumPy — no `sklearn`.
- A k-sweep over a configurable range of k values is run on the validation set.
- Best k is selected and reported based on validation accuracy.
- Prediction is fully vectorized where possible.

---

**US-5.2 — Softmax Regression (From Scratch)**
> As a developer, I want a multiclass softmax regression classifier trained with gradient descent, so that I have a parametric linear baseline.

**Acceptance Criteria:**
- Numerically stable softmax is implemented: `exp(x − max(x)) / sum(exp(x − max(x)))`.
- Cross-entropy loss uses epsilon clipping to prevent `log(0)`.
- Training supports mini-batch gradient descent.
- Uses the shared optimizer and LR schedule from Epic 6.

---

**US-5.3 — CNN Classifier (From Scratch)**
> As a developer, I want a CNN implemented entirely from scratch including forward and backward passes, so that I demonstrate a deep understanding of convolutional networks.

**Acceptance Criteria:**
- `Conv2D` forward and backward pass supports multi-channel inputs.
- Includes ReLU non-linearity.
- Includes max or average pooling with forward and backward pass.
- Includes flatten and at least one fully connected layer.
- Softmax + cross-entropy loss is implemented from scratch.
- Training loop uses mini-batches and the shared optimizer from Epic 6.

---

**US-5.4 — Research Paper Architecture**
> As a developer, I want to implement an architecture from a paper published within the last 5 years using a deep learning framework, so that I can benchmark against a state-of-the-art design.

**Acceptance Criteria:**
- The chosen paper is cited (title, authors, year, venue).
- Framework (PyTorch / TensorFlow / Keras) may be used for the model definition and training loop only.
- All dataset loading, preprocessing, and augmentation still go through shawwaf.
- The architecture is faithfully reproduced from the paper (documented deviations noted).

---

### Epic 6 — Optimizer

**US-6.1 — SGD Optimizer**
> As a developer, I want SGD implemented as the baseline optimizer, so that I have a simple reference to compare advanced optimizers against.

**Acceptance Criteria:**
- `SGD(params, lr)` updates weights by `w ← w − lr · grad`.
- Applies to Softmax Regression and CNN-from-scratch.

---

**US-6.2 — Advanced Optimizer**
> As a developer, I want at least one advanced optimizer (Momentum, RMSProp, or Adam) implemented, so that training convergence is improved over vanilla SGD.

**Acceptance Criteria:**
- One of Momentum, RMSProp, or Adam is implemented from scratch.
- Hyperparameters (e.g., β₁, β₂, ε for Adam) are configurable.
- Applies to both Softmax Regression and CNN-from-scratch.

---

**US-6.3 — Learning Rate Schedule**
> As a developer, I want a learning rate schedule applied during training, so that the optimizer reduces the learning rate at appropriate points.

**Acceptance Criteria:**
- At least one schedule is implemented: step decay, exponential decay, cosine annealing, or reduce-on-plateau.
- Current LR is logged each epoch in `logs.csv`.

---

**US-6.4 — Training Safety Features**
> As a developer, I want early stopping, gradient clipping, L2 regularization, and mini-batch shuffling, so that training is robust to instability and overfitting.

**Acceptance Criteria:**
- Early stopping halts training when validation loss has not improved for `patience` epochs.
- Gradient clipping caps gradient norms above a configurable threshold.
- L2 weight decay is added to the loss before backpropagation.
- Training samples are shuffled at the start of each epoch.
- All four safety features have configurable hyperparameters.

---

### Epic 7 — Logging & Checkpointing

**US-7.1 — Per-Epoch Training Log**
> As a developer, I want a `logs.csv` file written for every training run containing per-epoch metrics, so that training curves can be inspected and reproduced.

**Acceptance Criteria:**
- `logs.csv` contains at minimum the columns: `epoch`, `train_loss`, `val_loss`, `train_acc`, `val_acc`, `learning_rate`.
- File is appended to (not overwritten) when training is resumed.
- One `logs.csv` per experiment run, stored in `logs/<run_id>/`.

---

**US-7.2 — Best Model Checkpoint**
> As a developer, I want the best model checkpoint saved based on minimum validation loss, so that the optimal weights are preserved even if later epochs overfit.

**Acceptance Criteria:**
- Checkpoint is saved whenever validation loss improves.
- Checkpoint file contains: model weights, optimizer state, and full run configuration (JSON).
- Checkpoint is stored in `checkpoints/<run_id>/best.pt` (or equivalent).

---

**US-7.3 — Resumable Training**
> As a developer, I want training to be fully resumable from a checkpoint with identical state, so that interrupted runs can continue without restarting from scratch.

**Acceptance Criteria:**
- Loading a checkpoint and resuming produces identical loss values as if training had never been interrupted.
- Log file continues appending from the resumed epoch (no duplicate rows).
- Resume is demonstrated in the notebook or a documented CLI flag.

---

### Epic 8 — Evaluation & Comparison

**US-8.1 — Per-Class Metrics**
> As a researcher, I want precision, recall, and F1 computed per class on the held-out test set, so that I can identify which classes each model handles well or poorly.

**Acceptance Criteria:**
- Precision, recall, and F1 are computed from scratch (no sklearn) for each class.
- Results are reported in a formatted table in `docs/results/`.

---

**US-8.2 — Aggregate Metrics**
> As a researcher, I want macro-F1 and weighted-F1 reported alongside overall accuracy, so that I can compare models with a single aggregate score.

**Acceptance Criteria:**
- Macro-F1 (unweighted average over classes) and weighted-F1 (weighted by class support) are computed from scratch.
- Overall accuracy is reported as the fraction of correctly classified test samples.

---

**US-8.3 — Confusion Matrix**
> As a researcher, I want a confusion matrix for each model on the test set, so that I can visually inspect misclassification patterns.

**Acceptance Criteria:**
- Confusion matrix is computed from scratch and visualized as a heatmap.
- Heatmap is committed to `docs/results/` for each model.

---

**US-8.4 — Cross-Model Comparison**
> As a reviewer, I want a side-by-side comparison table of all four models across all evaluation metrics, so that I can draw conclusions about the trade-offs between approaches.

**Acceptance Criteria:**
- A single comparison table covering all four models (KNN, Softmax, CNN, Paper Model) is produced.
- Metrics included: accuracy, macro-F1, weighted-F1, and inference time.
- Table is committed to `docs/results/` and reproduced in the README.

---

### Epic 9 — Documentation

**US-9.1 — Math & Algorithms Notes**
> As a reviewer, I want concise mathematical explanations with equations and pseudocode for every algorithm in the pipeline, so that I can verify the correctness of each implementation.

**Acceptance Criteria:**
- `docs/math_notes.md` covers: distance metrics (KNN), softmax + cross-entropy, backpropagation (Conv2D, pooling, FC), gradient clipping, L2 regularization, each LR schedule, MRMR selection criterion, and each evaluation metric.
- Equations are formatted in LaTeX-style markdown or clearly typeset.
- Pseudocode is provided where equations alone are insufficient.

---

**US-9.2 — Results & Model Comparisons**
> As a reviewer, I want a dedicated results section showing training curves, evaluation metrics, and comparison plots for all models, so that conclusions are supported by evidence.

**Acceptance Criteria:**
- Training loss and validation loss curves are plotted for Softmax, CNN, and Paper Model.
- All confusion matrices and per-class metric tables are included.
- The cross-model comparison table from US-8.4 is present.
- All figures are committed to `docs/results/` and referenced in the README.

---

## Summary Table

| Epic | Area | # Stories |
|---|---|---|
| 1 | Dataset | 4 |
| 2 | Preprocessing | 2 |
| 3 | Augmentation | 3 |
| 4 | Feature Extraction | 4 |
| 5 | Model Training | 4 |
| 6 | Optimizer | 4 |
| 7 | Logging & Checkpointing | 3 |
| 8 | Evaluation & Comparison | 4 |
| 9 | Documentation | 2 |
| **Total** | | **30** |
