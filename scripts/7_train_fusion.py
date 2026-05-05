"""
Script 7: Stage 2 - Logistic Regression Fusion
===============================================
Combines C0 (MobileNetV2) + C1 (BERT)
to produce final prediction.

Input:  [C0, C1] per URL
Output: Binary prediction + confidence
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

OUTPUT_DIR  = "models/stage2"
RESULTS_DIR = "outputs/stage2"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# LOAD AND MERGE C0 + C1
# ==============================
def load_scores(split):
    """
    Load C0 and C1 scores and merge on label.
    C0 from ImageFolder (img_path, label, C0)
    C1 from URLDataset  (url, label, C1)
    """

    c0_df = pd.read_csv(f"outputs/stage0/{split}_with_C0.csv")
    c1_df = pd.read_csv(f"outputs/stage1/features/{split}_with_C1.csv")

    # C0 has img_path + label + C0
    # C1 has url + label + C1
    # Both are in same order (sorted by ImageFolder / CSV order)
    # Align by position and label

    print(f"\n{split.upper()}:")
    print(f"  C0 samples : {len(c0_df):,}")
    print(f"  C1 samples : {len(c1_df):,}")

    # Use the smaller size (in case of minor mismatch)
    n = min(len(c0_df), len(c1_df))

    c0_vals = c0_df['C0'].values[:n]
    c1_vals = c1_df['C1'].values[:n]

    # Use C1 labels (from CSV, more reliable)
    labels  = c1_df['label'].values[:n]

    print(f"  Using      : {n:,} samples")
    print(f"  Benign     : {(labels==0).sum():,}")
    print(f"  Malicious  : {(labels==1).sum():,}")

    # Feature matrix [C0, C1]
    X = np.column_stack([c0_vals, c1_vals])
    y = labels

    return X, y


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 2: Logistic Regression Fusion")
    print("=" * 60)

    # Load splits
    X_train, y_train = load_scores('train')
    X_val,   y_val   = load_scores('val')
    X_test,  y_test  = load_scores('test')

    print(f"\nFeature matrix shape:")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")

    # ==============================
    # TRAIN FUSION MODEL
    # ==============================
    print("\nTraining Logistic Regression fusion...")

    fusion = LogisticRegression(
        solver    = 'lbfgs',
        max_iter  = 1000,
        C         = 1.0,
        random_state = 42,
        verbose   = 1
    )

    fusion.fit(X_train, y_train)
    print("Fusion model trained")

    # ==============================
    # EVALUATE
    # ==============================
    def evaluate(X, y, split_name):
        preds = fusion.predict(X)
        proba = fusion.predict_proba(X)[:, 1]

        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f1   = f1_score(y, preds, zero_division=0)
        cm   = confusion_matrix(y, preds)

        print(f"\n{split_name.upper()} Results:")
        print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]:6,}  FP={cm[0,1]:6,}")
        print(f"    FN={cm[1,0]:6,}  TP={cm[1,1]:6,}")

        return acc, prec, rec, f1

    val_acc,  val_prec,  val_rec,  val_f1  = evaluate(X_val,  y_val,  'val')
    test_acc, test_prec, test_rec, test_f1 = evaluate(X_test, y_test, 'test')

    # ==============================
    # COMPARE INDIVIDUAL STAGES
    # ==============================
    print("\n" + "=" * 60)
    print("INDIVIDUAL STAGE COMPARISON (Test Set)")
    print("=" * 60)

    c0_preds    = (X_test[:, 0] > 0.5).astype(int)
    c1_preds    = (X_test[:, 1] > 0.5).astype(int)
    fused_preds = fusion.predict(X_test)

    print(f"  Stage 0 only (MobileNetV2) : "
          f"{accuracy_score(y_test, c0_preds)*100:.2f}%")
    print(f"  Stage 1 only (BERT)        : "
          f"{accuracy_score(y_test, c1_preds)*100:.2f}%")
    print(f"  Stage 2 Fusion (C0 + C1)   : "
          f"{accuracy_score(y_test, fused_preds)*100:.2f}%")
    print("\nFusion should outperform individual stages!")

    # Model coefficients
    print(f"\nFusion Model Coefficients:")
    print(f"  C0 weight (MobileNetV2) : {fusion.coef_[0][0]:.4f}")
    print(f"  C1 weight (BERT)        : {fusion.coef_[0][1]:.4f}")
    print(f"  Intercept               : {fusion.intercept_[0]:.4f}")

    # ==============================
    # SAVE MODEL
    # ==============================
    model_path = f"{OUTPUT_DIR}/fusion_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(fusion, f)

    print(f"\nFusion model saved: {model_path}")

    # Save results summary
    summary = {
        'stage':     ['MobileNetV2 only', 'BERT only', 'Fusion (C0+C1)'],
        'accuracy':  [
            accuracy_score(y_test, c0_preds),
            accuracy_score(y_test, c1_preds),
            test_acc
        ],
        'precision': [
            precision_score(y_test, c0_preds),
            precision_score(y_test, c1_preds),
            test_prec
        ],
        'recall': [
            recall_score(y_test, c0_preds),
            recall_score(y_test, c1_preds),
            test_rec
        ],
        'f1': [
            f1_score(y_test, c0_preds),
            f1_score(y_test, c1_preds),
            test_f1
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False)

    print(f"\nComparison table saved: {RESULTS_DIR}/comparison_table.csv")

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(summary_df.to_string(index=False))
