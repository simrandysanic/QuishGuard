# QuishGuard

**Multimodal QR phishing (quishing) detection using visual + semantic evidence**

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1-orange) ![Accuracy 99.54%](https://img.shields.io/badge/Accuracy-99.54%25-brightgreen)

M.Tech Thesis — IIT Ropar, 2026
*Simran Prasad (2024CSM1018) · Supervisor: Dr. Basant Subba*

---

## Overview

QuishGuard is a 3-stage multimodal framework for detecting **quishing attacks** — phishing delivered via QR codes. It achieves **99.54% accuracy** and **AUC 99.90%** on a 98,509-sample holdout test set.

### The Problem

Visual-only QR classifiers trained on clean datasets fail on real-world images, dropping from **99.19% → 84.80%** accuracy. This failure stems from QR code determinism: the same URL always generates the same code, so appearance alone is not enough. QuishGuard solves this by combining visual evidence with URL-semantic evidence.

---

## Architecture

```
         QR Code Image
               │
    ┌──────────▼──────────┐
    │   Stage 0: Visual   │       ┌──────────────────────┐
    │ MobileNetV2 (1280-d)│       │   QR Decode Failure? │
    │ ShuffleNetV2(1024-d)│       │   → C₁ = 0.0         │
    │ Concat → 2000-d     │       │   (treat as phishing) │
    │ HHO (313 features)  │       └──────────┬───────────┘
    │ Quadratic SVM       │                  │
    └────────┬────────────┘    ┌─────────────▼──────────┐
             │  C₀ score       │   Stage 1: Semantic     │
             │                 │ BERT-base-cased on URL   │
             │                 │ → C₁ score               │
             └────────┬────────┘
                      │
           ┌──────────▼──────────┐
           │  Stage 2: Fusion    │
           │ Logistic Regression │
           │ β₀=6.49, β₁=12.57  │
           │ threshold τ = 0.94  │
           └──────────┬──────────┘
                      │
               Phishing / Benign
```

---

## Key Results

| Component | Accuracy | AUC |
|---|---|---|
| Stage 0 — visual SVM (HHO) | 93.98% | 97.77% |
| Stage 0 — end-to-end visual only | 96.38% | 98.91% |
| Stage 1 — BERT URL classifier | 99.54% | 99.93% |
| **QuishGuard — full fusion** | **99.54%** | **99.90%** |
| Wild-test visual-only baseline | 84.80% | — |

**Confusion matrix (test set, 98,509 samples):**

|  | Predicted Benign | Predicted Phishing |
|---|---|---|
| **Actual Benign** | TN = 49,521 | FP = 478 |
| **Actual Phishing** | FN = 645 | TP = 47,865 |

---

## Repository Structure

```
QuishGuard/
├── scripts/                              # Pipeline scripts — run in numbered order
│   ├── 1_balance_dataset.py              # Balance phishing/benign URL sets
│   ├── 2_generate_qr_codes_v2.py         # Render 1.08M QR code images
│   ├── 3_train_mobilenet_features.py     # Fine-tune MobileNetV2 (Stage 0a)
│   ├── 3b_train_shufflenet_features.py
│   ├── 3c_extract_shufflenet_features.py
│   ├── 4b_extract_features_canonical.py  # Extract 2000-d concat features
│   ├── 4c_hho_v2_feature_selection.py    # Harris Hawks Optimization (313 dims)
│   ├── 4d_train_svm_hho.py               # Train quadratic SVM on HHO features
│   ├── 5_train_bert.py                   # Fine-tune BERT for URL classification
│   ├── 6_extract_C1_scores.py            # Extract BERT confidence scores
│   ├── 7c_retrain_fusion_hho_clean.py    # Train Stage 2 logistic fusion
│   └── 8b_end_to_end_penalty_pipeline.py # Full pipeline evaluation
│
├── models/
│   ├── stage0/     # CNN weights (.pth) + SVM models (.pkl)
│   ├── stage1/     # BERT configs & tokenizer (weights excluded — see Dataset)
│   └── stage2/     # Fusion model coefficients (.json, .pkl)
│
├── outputs/
│   ├── stage0_v2/           # HHO selected feature indices, SVM result JSONs
│   ├── stage1/features/     # BERT C₁ scores per split
│   ├── stage2_models_clean/ # Final clean fusion model
│   └── e2e_test/            # End-to-end evaluation results
│
├── data/
│   ├── processed/  # train.csv, val.csv, test.csv (URL + label, ~25 MB)
│   └── raw/        # Source URL dataset (Kaggle phishing URLs)
│
└── requirements.txt
```

---

## Dataset

The QR image datasets are **not included in this repository** due to size (~5.9 GB total).

| Folder | Size | Description |
|---|---|---|
| `data/qr_images/` | 4.2 GB | 1.08M QR codes across 8 rendering tiers (Semester 1 study) |
| `data/qr_images_v2/` | 1.7 GB | Refined QR codes for train/val/test splits (QuishGuard training) |

**To request the dataset:** Open an [issue](https://github.com/simrandysanic/QuishGuard/issues) and the data can be shared via Google Drive.

Similarly, the fine-tuned BERT weights (`pytorch_model.bin`, ~414 MB × 11 checkpoints ≈ 4.5 GB) are excluded. Tokenizer configs and model architecture JSONs are included so the architecture is fully reproducible. Weights available on request.

---

## Setup

```bash
conda create -n qr_env python=3.10
conda activate qr_env
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0.1 (CUDA 11.7)
- HuggingFace Transformers 4.30.2
- scikit-learn 1.3.0
- OpenCV 4.x
- segno (QR code generation)

---

## Running the Pipeline

Run scripts in numbered order. Each saves outputs to `outputs/` for the next stage.

```bash
# Stage 0 — Visual
python scripts/1_balance_dataset.py
python scripts/2_generate_qr_codes_v2.py
python scripts/3_train_mobilenet_features.py
python scripts/3b_train_shufflenet_features.py
python scripts/4b_extract_features_canonical.py
python scripts/4c_hho_v2_feature_selection.py
python scripts/4d_train_svm_hho.py

# Stage 1 — Semantic
python scripts/5_train_bert.py
python scripts/6_extract_C1_scores.py

# Stage 2 — Fusion + Evaluation
python scripts/7c_retrain_fusion_hho_clean.py
python scripts/8b_end_to_end_penalty_pipeline.py
```

---

## Citation

If you use this work, please cite:

```
Simran Prasad. "QuishGuard: A Visual and Semantic Evidence-Based Multimodal
Framework for Quishing Detection." M.Tech Thesis, IIT Ropar, 2026.
Enrollment No: 2024CSM1018. Supervisor: Dr. Basant Subba.
```

---

## License

MIT
