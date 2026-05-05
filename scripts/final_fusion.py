import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

OUTPUT_DIR = "models/stage2"
RESULTS_DIR = "outputs/stage2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

C0_PATH = "outputs/stage0/{split}_with_C0_fusion_v2.csv"
C1_PATH = "outputs/stage1/features/{split}_with_C1_fusion_v2.csv"

MODEL_OUT = f"{OUTPUT_DIR}/fusion_model_v2.pkl"
META_OUT = f"{OUTPUT_DIR}/fusion_meta_v2.json"
TABLE_OUT = f"{RESULTS_DIR}/comparison_table_fusion_v2.csv"


def load_split(split: str):
    c0 = pd.read_csv(C0_PATH.format(split=split))
    c1 = pd.read_csv(C1_PATH.format(split=split))

    req_c0 = {"sample_id", "url", "label", "C0"}
    req_c1 = {"sample_id", "url", "label", "C1"}

    if not req_c0.issubset(set(c0.columns)):
        raise ValueError(f"{split}: C0 file missing columns: {req_c0 - set(c0.columns)}")
    if not req_c1.issubset(set(c1.columns)):
        raise ValueError(f"{split}: C1 file missing columns: {req_c1 - set(c1.columns)}")

    merged = c0.merge(
        c1,
        on="sample_id",
        suffixes=("_c0", "_c1"),
        how="inner",
        validate="one_to_one"
    )

    url_mismatch = int((merged["url_c0"] != merged["url_c1"]).sum())
    label_mismatch = int((merged["label_c0"].astype(int) != merged["label_c1"].astype(int)).sum())
    if url_mismatch > 0 or label_mismatch > 0:
        raise ValueError(
            f"{split}: alignment check failed | url_mismatch={url_mismatch}, "
            f"label_mismatch={label_mismatch}"
        )

    merged = merged.dropna(subset=["C0", "C1", "label_c0"]).copy()

    X = merged[["C0", "C1"]].values.astype(np.float64)
    y = merged["label_c0"].astype(int).values

    print(f"\n{split.upper()}")
    print(f"  c0={len(c0):,} c1={len(c1):,} merged={len(merged):,}")
    print(f"  positives={int(y.sum()):,} negatives={int((y==0).sum()):,}")

    return X, y, merged


def best_threshold_by_f1(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }
    print(
        f"{name}: ACC={metrics['acc']:.4f} PREC={metrics['prec']:.4f} "
        f"REC={metrics['rec']:.4f} F1={metrics['f1']:.4f} "
        f"AUC={metrics['auc']:.4f} @thr={threshold:.2f}"
    )
    return metrics


def main():
    print("STAGE2 FUSION V2 (MERGE ON SAMPLE_ID)")
    print("=" * 70)

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")

    c_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best = None

    for c in c_grid:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=3000,
            C=c,
            random_state=42,
        )
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        thr, val_f1 = best_threshold_by_f1(y_val, val_prob)

        if best is None or val_f1 > best["val_f1"]:
            best = {
                "C": c,
                "thr": thr,
                "val_f1": val_f1,
                "model": model
            }

    fusion = best["model"]
    best_thr = best["thr"]

    coef_c0, coef_c1 = fusion.coef_[0]
    intercept = fusion.intercept_[0]

    print("\nBEST SETTINGS")
    print(f"  C={best['C']}")
    print(f"  threshold={best_thr:.2f}")
    print(f"  val_f1={best['val_f1']:.4f}")

    print("\nLEARNED FUSION")
    print(f"  logit = {intercept:.6f} + ({coef_c0:.6f} * C0) + ({coef_c1:.6f} * C1)")
    denom = abs(coef_c0) + abs(coef_c1) + 1e-12
    print(f"  rel_abs_weight: C0={abs(coef_c0)/denom:.4f}, C1={abs(coef_c1)/denom:.4f}")

    val_prob = fusion.predict_proba(X_val)[:, 1]
    test_prob = fusion.predict_proba(X_test)[:, 1]

    print("\nEVALUATION")
    val_m = evaluate("VAL ", y_val, val_prob, best_thr)
    test_m = evaluate("TEST", y_test, test_prob, best_thr)

    c0_prob = X_test[:, 0]
    c1_prob = X_test[:, 1]
    c0_pred = (c0_prob >= 0.5).astype(int)
    c1_pred = (c1_prob >= 0.5).astype(int)

    baseline = pd.DataFrame([
        {
            "model": "C0@0.5",
            "acc": accuracy_score(y_test, c0_pred),
            "prec": precision_score(y_test, c0_pred, zero_division=0),
            "rec": recall_score(y_test, c0_pred, zero_division=0),
            "f1": f1_score(y_test, c0_pred, zero_division=0),
            "auc": roc_auc_score(y_test, c0_prob),
        },
        {
            "model": "C1@0.5",
            "acc": accuracy_score(y_test, c1_pred),
            "prec": precision_score(y_test, c1_pred, zero_division=0),
            "rec": recall_score(y_test, c1_pred, zero_division=0),
            "f1": f1_score(y_test, c1_pred, zero_division=0),
            "auc": roc_auc_score(y_test, c1_prob),
        },
        {
            "model": "FusionV2",
            "acc": test_m["acc"],
            "prec": test_m["prec"],
            "rec": test_m["rec"],
            "f1": test_m["f1"],
            "auc": test_m["auc"],
        },
    ])

    print("\nBASELINES (TEST)")
    print(baseline.to_string(index=False))

    baseline.to_csv(TABLE_OUT, index=False)

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(fusion, f)

    with open(META_OUT, "w") as f:
        json.dump(
            {
                "best_C": best["C"],
                "best_threshold": best_thr,
                "coef_C0": float(coef_c0),
                "coef_C1": float(coef_c1),
                "intercept": float(intercept),
                "train_size": int(len(y_train)),
                "val_size": int(len(y_val)),
                "test_size": int(len(y_test)),
                "val_metrics": val_m,
                "test_metrics": test_m,
                "input_files": {
                    "c0": C0_PATH,
                    "c1": C1_PATH
                },
            },
            f,
            indent=2
        )

    print("\nSaved:")
    print(f"  {TABLE_OUT}")
    print(f"  {MODEL_OUT}")
    print(f"  {META_OUT}")


if __name__ == "__main__":
    main()
