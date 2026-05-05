import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

OUTPUT_DIR = "models/stage2"
RESULTS_DIR = "outputs/stage2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_clean(split):
    c0 = pd.read_csv(f"outputs/stage0/{split}_with_C0.csv")[["url", "label", "C0"]].copy()
    c1 = pd.read_csv(f"outputs/stage1/features/{split}_with_C1.csv")[["url", "label", "C1"]].copy()

    c0["label"] = c0["label"].astype(int)
    c1["label"] = c1["label"].astype(int)

    # Collapse duplicates by (url,label) so merge is one-to-one
    c0u = c0.groupby(["url", "label"], as_index=False)["C0"].mean()
    c1u = c1.groupby(["url", "label"], as_index=False)["C1"].mean()

    merged = c0u.merge(c1u, on=["url", "label"], how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["C0", "C1", "label"]).copy()

    X = merged[["C0", "C1"]].values.astype(np.float64)
    y = merged["label"].values.astype(int)

    print(f"\n{split.upper()}")
    print(f"  stage0 raw={len(c0):,} uniq(url,label)={len(c0u):,}")
    print(f"  stage1 raw={len(c1):,} uniq(url,label)={len(c1u):,}")
    print(f"  merged={len(merged):,} pos={y.sum():,} neg={(y==0).sum():,}")
    return X, y, merged

def find_best_threshold(y_true, prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def evaluate(y, prob, thr, name):
    pred = (prob >= thr).astype(int)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, prob)
    print(f"{name}: ACC={acc:.4f} PREC={prec:.4f} REC={rec:.4f} F1={f1:.4f} AUC={auc:.4f} @thr={thr:.2f}")
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "auc":auc}

print("STAGE2 FUSION (SAFE MERGE ON URL+LABEL)")
print("="*70)

X_train, y_train, _ = load_clean("train")
X_val, y_val, _ = load_clean("val")
X_test, y_test, test_df = load_clean("test")

# Hyperparam sweep on validation F1
c_grid = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best = None

for c in c_grid:
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=c,
        random_state=42
    )
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    thr, val_f1 = find_best_threshold(y_val, val_prob)

    if best is None or val_f1 > best["val_f1"]:
        best = {"C": c, "thr": thr, "val_f1": val_f1, "model": model}

fusion = best["model"]
best_thr = best["thr"]

print("\nBEST SETTINGS")
print(f"  C={best['C']}")
print(f"  threshold={best_thr:.2f}")
print(f"  val_f1={best['val_f1']:.4f}")

coef_c0, coef_c1 = fusion.coef_[0]
intercept = fusion.intercept_[0]
print("\nLEARNED FUSION")
print(f"  logit = {intercept:.6f} + ({coef_c0:.6f} * C0) + ({coef_c1:.6f} * C1)")
print(f"  relative_abs_weight C0={abs(coef_c0)/(abs(coef_c0)+abs(coef_c1)+1e-12):.4f}, C1={abs(coef_c1)/(abs(coef_c0)+abs(coef_c1)+1e-12):.4f}")

val_prob = fusion.predict_proba(X_val)[:, 1]
test_prob = fusion.predict_proba(X_test)[:, 1]

print("\nEVALUATION")
val_metrics = evaluate(y_val, val_prob, best_thr, "VAL ")
test_metrics = evaluate(y_test, test_prob, best_thr, "TEST")

# Baselines on same cleaned test set
c0_auc = roc_auc_score(y_test, X_test[:,0])
c1_auc = roc_auc_score(y_test, X_test[:,1])
c0_pred = (X_test[:,0] >= 0.5).astype(int)
c1_pred = (X_test[:,1] >= 0.5).astype(int)

baseline = pd.DataFrame([
    {"model":"C0@0.5", "acc":accuracy_score(y_test,c0_pred), "f1":f1_score(y_test,c0_pred,zero_division=0), "auc":c0_auc},
    {"model":"C1@0.5", "acc":accuracy_score(y_test,c1_pred), "f1":f1_score(y_test,c1_pred,zero_division=0), "auc":c1_auc},
    {"model":"Fusion", "acc":test_metrics["acc"], "f1":test_metrics["f1"], "auc":test_metrics["auc"]},
])
print("\nBASELINES (same cleaned test set)")
print(baseline.to_string(index=False))

baseline.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False)

with open(f"{OUTPUT_DIR}/fusion_model.pkl", "wb") as f:
    pickle.dump(fusion, f)

with open(f"{OUTPUT_DIR}/fusion_meta.json", "w") as f:
    json.dump({
        "best_C": best["C"],
        "best_threshold": best_thr,
        "coef_C0": float(coef_c0),
        "coef_C1": float(coef_c1),
        "intercept": float(intercept),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
    }, f, indent=2)

print("\nSaved:")
print(f"  {RESULTS_DIR}/comparison_table.csv")
print(f"  {OUTPUT_DIR}/fusion_model.pkl")
print(f"  {OUTPUT_DIR}/fusion_meta.json")

