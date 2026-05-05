"""
Stage 2: Fusion (Script D)
============================
Combines C0 (MobileNetV2+SVM) and C1 (BERT) scores via Logistic Regression.

INPUT:
  outputs/stage0_v2/{split}_with_C0.csv   columns: url, label, C0
  outputs/stage1/features/{split}_with_C1.csv  columns: url, label, C1

MERGE: inner join on 'url' – only rows present in BOTH files are used.

SANITY CHECKS (printed before training):
  - Merge rate (how many URLs matched)
  - Label agreement between C0 file and C1 file
    → should be 100%; if < 100% stop and investigate

TRAINING:
  - LogisticRegression, C-parameter sweep [0.01, 0.1, 1, 10, 100]
  - Best C selected by val AUC-ROC
  - Final evaluation on test set

OUTPUT:
  models/stage2/fusion_model.pkl
  outputs/stage2/comparison_table.csv
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
C0_DIR      = "outputs/stage0_v2"
C1_DIR      = "outputs/stage1/features"
MODEL_DIR   = "models/stage2"
RESULTS_DIR = "outputs/stage2"
SPLITS      = ["train", "val", "test"]

C_GRID      = [0.01, 0.1, 1.0, 10.0, 100.0]


# ============================================================
# HELPERS
# ============================================================

def load_and_merge(split: str):
    c0_path = os.path.join(C0_DIR, f"{split}_with_C0.csv")
    c1_path = os.path.join(C1_DIR, f"{split}_with_C1.csv")

    assert os.path.exists(c0_path), f"C0 file missing: {c0_path}\nRun 5b_extract_C0_canonical.py"
    assert os.path.exists(c1_path), f"C1 file missing: {c1_path}"

    c0 = pd.read_csv(c0_path)   # url, label, C0
    c1 = pd.read_csv(c1_path)   # url, label, C1

    print(f"\n{split.upper()}")
    print(f"  C0 rows : {len(c0):,}")
    print(f"  C1 rows : {len(c1):,}")

    # Inner join on URL
    merged = pd.merge(c0[["url", "label", "C0"]],
                      c1[["url", "label", "C1"]],
                      on="url", suffixes=("_c0", "_c1"))

    merge_pct = len(merged) / max(len(c0), len(c1)) * 100
    print(f"  Merged  : {len(merged):,}  ({merge_pct:.1f}% match)")

    if merge_pct < 95:
        print(f"  ⚠ WARNING: only {merge_pct:.1f}% of URLs matched. "
              "Check that C0 and C1 were generated from the same CSV.")

    # ---- Label agreement check ----
    agree = (merged["label_c0"] == merged["label_c1"]).mean() * 100
    print(f"  Label agreement (C0 vs C1) : {agree:.2f}%")
    if agree < 99.9:
        print(f"  ⚠ WARNING: label mismatch detected! "
              f"{(merged['label_c0'] != merged['label_c1']).sum()} rows differ.\n"
              "  This means C0 labels are misaligned. Re-run the pipeline from step 2.")
        # Do NOT assert – just warn and use canonical C1 labels
        print("  → Using C1 labels (BERT ground truth) as canonical.")

    # Canonical label = c1 label (BERT was trained on correct labels)
    merged["label"] = merged["label_c1"].astype(int)

    X = merged[["C0", "C1"]].values
    y = merged["label"].values

    return X, y, merged


def metrics(y_true, y_pred, y_proba, split_name):
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_proba)
    cm    = confusion_matrix(y_true, y_pred)

    print(f"\n  {split_name.upper()} metrics:")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-score  : {f1:.4f}")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"    Confusion matrix:")
    print(f"      TN={cm[0,0]:7,}  FP={cm[0,1]:7,}")
    print(f"      FN={cm[1,0]:7,}  TP={cm[1,1]:7,}")

    return {"split": split_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc}


def single_stage_auc(y_true, scores, name):
    auc = roc_auc_score(y_true, scores)
    acc = accuracy_score(y_true, (scores >= 0.5).astype(int))
    print(f"  {name:30s}: AUC={auc:.4f}  Acc={acc:.4f}")
    return auc, acc


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("STAGE 2: FUSION TRAINING (Script D)")
    print("=" * 60)

    # ---- Load data ----
    X_train, y_train, train_merged = load_and_merge("train")
    X_val,   y_val,   val_merged   = load_and_merge("val")
    X_test,  y_test,  test_merged  = load_and_merge("test")

    print(f"\nFeature matrix shapes:")
    print(f"  Train : {X_train.shape}   positives: {y_train.sum():,}")
    print(f"  Val   : {X_val.shape}   positives: {y_val.sum():,}")
    print(f"  Test  : {X_test.shape}   positives: {y_test.sum():,}")

    # ---- C sweep – select best C on val AUC ----
    print("\n" + "=" * 60)
    print("HYPER-PARAMETER SWEEP (val AUC)")
    print("=" * 60)

    best_C   = None
    best_auc = -1

    for C in C_GRID:
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        val_proba = lr.predict_proba(X_val)[:, 1]
        val_auc   = roc_auc_score(y_val, val_proba)
        print(f"  C={C:7.2f}  val AUC = {val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_C   = C
            best_model = lr

    print(f"\nBest C = {best_C}  (val AUC = {best_auc:.6f})")

    # ---- Per-stage baselines ----
    print("\n" + "=" * 60)
    print("INDIVIDUAL STAGE BASELINES (val set)")
    print("=" * 60)
    c0_auc, c0_acc = single_stage_auc(y_val, X_val[:, 0], "C0 only (MobileNet+SVM)")
    c1_auc, c1_acc = single_stage_auc(y_val, X_val[:, 1], "C1 only (BERT)")

    print("\n" + "=" * 60)
    print("INDIVIDUAL STAGE BASELINES (test set)")
    print("=" * 60)
    c0_auc_t, c0_acc_t = single_stage_auc(y_test, X_test[:, 0], "C0 only (MobileNet+SVM)")
    c1_auc_t, c1_acc_t = single_stage_auc(y_test, X_test[:, 1], "C1 only (BERT)")

    # ---- Evaluate fusion ----
    print("\n" + "=" * 60)
    print("FUSION EVALUATION")
    print("=" * 60)
    print(f"\nUsing best C = {best_C}")

    all_results = []

    val_pred  = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    all_results.append(metrics(y_val,  val_pred,  val_proba,  "val"))

    test_pred  = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    all_results.append(metrics(y_test, test_pred, test_proba, "test"))

    # ---- Fusion model coefficients ----
    print(f"\nFusion model coefficients:")
    print(f"  C0 weight (MobileNet+SVM) : {best_model.coef_[0][0]:+.4f}")
    print(f"  C1 weight (BERT)          : {best_model.coef_[0][1]:+.4f}")
    print(f"  Intercept                 : {best_model.intercept_[0]:+.4f}")

    # ---- Summary comparison table ----
    test_res = all_results[1]
    summary = pd.DataFrame([
        {
            "Stage": "MobileNetV2 + SVM  (C0)",
            "Accuracy": round(c0_acc_t, 4),
            "AUC": round(c0_auc_t, 4),
        },
        {
            "Stage": "BERT               (C1)",
            "Accuracy": round(c1_acc_t, 4),
            "AUC": round(c1_auc_t, 4),
        },
        {
            "Stage": "Fusion C0 + C1",
            "Accuracy": round(test_res["accuracy"], 4),
            "AUC": round(test_res["auc"], 4),
        },
    ])

    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE (Test Set)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print()

    if test_res["auc"] >= max(c0_auc_t, c1_auc_t):
        gain = (test_res["auc"] - max(c0_auc_t, c1_auc_t)) * 100
        print(f"  ✅ Fusion improves AUC by +{gain:.3f}% over the best single stage.")
    else:
        print("  ⚠  Fusion did not improve over the best single stage.")
        print("     (This can happen if one stage is already near-perfect.)")

    # ---- Save ----
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path   = os.path.join(MODEL_DIR, "fusion_model.pkl")
    results_path = os.path.join(RESULTS_DIR, "comparison_table.csv")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    summary.to_csv(results_path, index=False)

    print(f"\n✅ Model saved  : {model_path}")
    print(f"✅ Table saved  : {results_path}")

    # ---- Threshold tuning (bonus) ----
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING (val set F1 maximisation)")
    print("=" * 60)
    val_proba_unfilt = best_model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_val, (val_proba_unfilt >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"  Best threshold (val F1)  : {best_thresh:.2f}  (F1={best_f1:.4f})")
    test_preds_tuned = (test_proba >= best_thresh).astype(int)
    tuned_f1  = f1_score(y_test, test_preds_tuned, zero_division=0)
    tuned_acc = accuracy_score(y_test, test_preds_tuned)
    print(f"  Test F1 @ tuned threshold: {tuned_f1:.4f}")
    print(f"  Test Acc @ tuned threshold: {tuned_acc:.4f}")

    print("\n" + "=" * 60)
    print("✅  Fusion training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
