# scripts/7_train_fusion_final.py
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

CANONICAL = "data/processed/{split}.csv"
C0_PATH   = "outputs/stage0/{split}_with_C0.csv"
C1_PATH   = "outputs/stage1/features/{split}_with_C1.csv"

MODEL_OUT = f"{OUTPUT_DIR}/fusion_model_final.pkl"
META_OUT  = f"{OUTPUT_DIR}/fusion_meta_final.json"
TABLE_OUT = f"{RESULTS_DIR}/comparison_table_final.csv"


def load_split(split: str):
    # 1. Canonical labels — single source of truth
    canon = pd.read_csv(CANONICAL.format(split=split))[["url", "label"]]
    canon["label"] = canon["label"].astype(int)
    # If a URL appears multiple times, keep the majority label
    canon = (
        canon.groupby("url")["label"]
             .agg(lambda x: int(x.mode()[0]))
             .reset_index()
    )

    # 2. C0 scores — keep url + C0 only, discard C0 file labels
    c0 = pd.read_csv(C0_PATH.format(split=split))[["url", "C0"]]
    c0["url"] = c0["url"].astype(str)
    c0 = c0.groupby("url", as_index=False)["C0"].mean()

    # 3. C1 scores — keep url + C1 only, labels come from canon
    c1 = pd.read_csv(C1_PATH.format(split=split))[["url", "C1"]]
    c1["url"] = c1["url"].astype(str)
    c1 = c1.groupby("url", as_index=False)["C1"].mean()

    # 4. Join: canonical ← C0 ← C1 (inner on both)
    merged = (
        canon
        .merge(c0, on="url", how="inner")
        .merge(c1, on="url", how="inner")
        .dropna(subset=["C0", "C1", "label"])
        .drop_duplicates(subset="url")
        .reset_index(drop=True)
    )

    X = merged[["C0", "C1"]].values.astype(np.float64)
    y = merged["label"].values.astype(int)

    print(f"\n{split.upper()}")
    print(f"  canon={len(canon):,}  c0_uniq={len(c0):,}  c1_uniq={len(c1):,}  merged={len(merged):,}")
    print(f"  positives={int(y.sum()):,}  negatives={int((y==0).sum()):,}")

    # Sanity: C1 AUC on merged set should be high
    from sklearn.metrics import roc_auc_score as auc
    print(f"  C0_AUC={auc(y, X[:,0]):.4f}  C1_AUC={auc(y, X[:,1]):.4f}")

    return X, y, merged


def best_threshold_by_f1(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def evaluate(name, y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    m = {
        "acc":  accuracy_score(y_true, pred),
        "prec": precision_score(y_true, pred, zero_division=0),
        "rec":  recall_score(y_true, pred, zero_division=0),
        "f1":   f1_score(y_true, pred, zero_division=0),
        "auc":  roc_auc_score(y_true, y_prob),
    }
    print(
        f"{name}: ACC={m['acc']:.4f} PREC={m['prec']:.4f} "
        f"REC={m['rec']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} @thr={threshold:.2f}"
    )
    return m


def main():
    print("STAGE2 FUSION FINAL (CANONICAL LABELS, URL JOIN)")
    print("=" * 70)

    X_train, y_train, _ = load_split("train")
    X_val,   y_val,   _ = load_split("val")
    X_test,  y_test,  _ = load_split("test")

    # Hyperparameter sweep on val F1
    best = None
    for c in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model = LogisticRegression(
            solver="lbfgs", max_iter=3000, C=c, random_state=42
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        thr, val_f1 = best_threshold_by_f1(y_val, val_prob)
        if best is None or val_f1 > best["val_f1"]:
            best = {"C": c, "thr": thr, "val_f1": val_f1, "model": model}

    fusion    = best["model"]
    best_thr  = best["thr"]
    coef_c0, coef_c1 = fusion.coef_[0]
    intercept = fusion.intercept_[0]

    print("\nBEST SETTINGS")
    print(f"  C={best['C']}  threshold={best_thr:.2f}  val_f1={best['val_f1']:.4f}")

    denom = abs(coef_c0) + abs(coef_c1) + 1e-12
    print("\nLEARNED FUSION WEIGHTS")
    print(f"  logit = {intercept:.4f} + ({coef_c0:.4f} * C0) + ({coef_c1:.4f} * C1)")
    print(f"  C0 relative weight: {abs(coef_c0)/denom:.4f}")
    print(f"  C1 relative weight: {abs(coef_c1)/denom:.4f}")

    val_prob  = fusion.predict_proba(X_val)[:, 1]
    test_prob = fusion.predict_proba(X_test)[:, 1]

    print("\nEVALUATION")
    val_m  = evaluate("VAL ", y_val,  val_prob,  best_thr)
    test_m = evaluate("TEST", y_test, test_prob, best_thr)

    # Baselines on same cleaned test set
    c0_pred = (X_test[:,0] >= 0.5).astype(int)
    c1_pred = (X_test[:,1] >= 0.5).astype(int)

    baseline = pd.DataFrame([
        {"model": "C0 (MobileNet+SVM)",
         "acc": accuracy_score(y_test, c0_pred),
         "prec": precision_score(y_test, c0_pred, zero_division=0),
         "rec": recall_score(y_test, c0_pred, zero_division=0),
         "f1": f1_score(y_test, c0_pred, zero_division=0),
         "auc": roc_auc_score(y_test, X_test[:,0])},
        {"model": "C1 (BERT)",
         "acc": accuracy_score(y_test, c1_pred),
         "prec": precision_score(y_test, c1_pred, zero_division=0),
         "rec": recall_score(y_test, c1_pred, zero_division=0),
         "f1": f1_score(y_test, c1_pred, zero_division=0),
         "auc": roc_auc_score(y_test, X_test[:,1])},
        {"model": "Fusion (Final)",
         "acc": test_m["acc"], "prec": test_m["prec"],
         "rec": test_m["rec"], "f1": test_m["f1"], "auc": test_m["auc"]},
    ])

    print("\nFINAL COMPARISON TABLE (TEST)")
    print(baseline.to_string(index=False))

    baseline.to_csv(TABLE_OUT, index=False)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(fusion, f)
    with open(META_OUT, "w") as f:
        json.dump({
            "best_C": best["C"],
            "best_threshold": best_thr,
            "coef_C0": float(coef_c0),
            "coef_C1": float(coef_c1),
            "intercept": float(intercept),
            "C0_relative_weight": float(abs(coef_c0)/denom),
            "C1_relative_weight": float(abs(coef_c1)/denom),
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "test_size": int(len(y_test)),
            "val_metrics": val_m,
            "test_metrics": test_m,
        }, f, indent=2)

    print("\nSaved:")
    print(f"  {TABLE_OUT}")
    print(f"  {MODEL_OUT}")
    print(f"  {META_OUT}")


if __name__ == "__main__":
    main()
