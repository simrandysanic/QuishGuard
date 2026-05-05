"""
C0 Score Extraction (Script C)
================================
Extracts C0 (malicious probability) from the trained SVM using
canonically-ordered features produced by 4b_extract_features_canonical.py.

KEY FIX vs. original 5_extract_C0_from_svm.py:
  The original saved only 'label, C0' with NO URL column, making
  URL-based merging with C1 impossible.
  This script saves 'url, label, C0'  so every row is traceable.

INPUT:
  outputs/stage0_v2/{split}_features_1000d.npy   ← from Script B
  outputs/stage0_v2/{split}_manifest.csv         ← from Script B
  models/stage0/svm_quadratic.pkl               ← original trained SVM

OUTPUT:
  outputs/stage0_v2/{split}_with_C0.csv
  columns: url, label, C0
  where C0 = P(malicious) from SVM predict_proba[:,1]
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


# ============================================================
# CONFIG
# ============================================================
FEATURE_DIR = "outputs/stage0_v2"
SVM_PATH    = "models/stage0/svm_quadratic.pkl"
OUTPUT_DIR  = "outputs/stage0_v2"    # same dir, different file
SPLITS      = ["train", "val", "test"]


# ============================================================
# LOAD SVM
# ============================================================

def load_svm(path: str):
    assert os.path.exists(path), (
        f"SVM model not found: {path}\n"
        "Models/stage0/svm_quadratic.pkl must exist (trained by original pipeline)."
    )
    with open(path, "rb") as f:
        svm = pickle.load(f)
    print(f"✅ SVM loaded: {path}")
    print(f"   kernel={svm.kernel}  degree={svm.degree}  C={svm.C}")
    return svm


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split: str, svm):
    feat_path     = os.path.join(FEATURE_DIR, f"{split}_features_1000d.npy")
    manifest_path = os.path.join(FEATURE_DIR, f"{split}_manifest.csv")

    assert os.path.exists(feat_path), (
        f"Features not found: {feat_path}\nRun 4b_extract_features_canonical.py first."
    )
    assert os.path.exists(manifest_path), (
        f"Manifest not found: {manifest_path}\nRun 4b_extract_features_canonical.py first."
    )

    features = np.load(feat_path)          # shape (N, 1000)
    manifest = pd.read_csv(manifest_path)  # url, label, npy_index

    print(f"\n{split.upper()}: {len(features):,} samples  features={features.shape}")

    assert len(features) == len(manifest), (
        f"MISMATCH: features has {len(features)} rows but manifest has {len(manifest)} rows.\n"
        "Re-run 4b_extract_features_canonical.py to regenerate both files."
    )

    # ---- SVM predict_proba ----
    # class order: svm.classes_ tells us which column is which
    proba = svm.predict_proba(features)    # (N, 2)
    classes = list(svm.classes_)
    mal_idx = classes.index(1)             # column index for 'malicious'
    C0      = proba[:, mal_idx]            # P(malicious)

    # ---- Build output DataFrame ----
    out_df = pd.DataFrame({
        "url":   manifest["url"].values,
        "label": manifest["label"].values,
        "C0":    C0,
    })

    # ---- Quick diagnostics ----
    y_true     = manifest["label"].values.astype(int)
    y_pred_bin = (C0 >= 0.5).astype(int)
    acc        = accuracy_score(y_true, y_pred_bin)

    try:
        auc = roc_auc_score(y_true, C0)
        print(f"  SVM accuracy (thresh=0.5) : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  SVM AUC-ROC               : {auc:.4f}")
    except Exception:
        print(f"  SVM accuracy (thresh=0.5) : {acc:.4f}  (AUC needs both classes)")

    # ---- Save ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{split}_with_C0.csv")
    out_df.to_csv(out_path, index=False)

    print(f"  Saved : {out_path}")
    print(f"  Cols  : {out_df.columns.tolist()}")
    print(f"  Sample:")
    print(out_df.head(3).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("C0 SCORE EXTRACTION (Script C)")
    print("Using: existing models/stage0/svm_quadratic.pkl")
    print("Order: canonical (from 4b_extract_features_canonical.py)")
    print("Output cols: url, label, C0")
    print("=" * 60)

    svm = load_svm(SVM_PATH)

    for split in SPLITS:
        process_split(split, svm)

    print("\n" + "=" * 60)
    print("✅  C0 extraction complete.")
    print(f"Output dir : {OUTPUT_DIR}/")
    print("NEXT STEP  : Run 7_train_fusion_v2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
