"""
7b_retrain_fusion_hho.py
Retrain the Stage 2 logistic fusion using the new HHO-based C0 scores.

Reads:
  outputs/stage0_v2/{split}_with_C0_hho_v2.csv  -- new C0 scores (HHO SVM)
  outputs/stage1/features/{split}_with_C1.csv   -- BERT C1 scores (unchanged)

Outputs:
  models/stage2/fusion_meta_hho.json            -- new fusion weights
  outputs/stage2/fusion_hho_{split}_results.csv -- per-split predictions
"""

import numpy as np
import pandas as pd
import json, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

BASE      = os.path.expanduser("~/QuishGuard")
C0_DIR    = os.path.join(BASE, "outputs/stage0_v2")
C1_DIR    = os.path.join(BASE, "outputs/stage1/features")
OUT_MODEL = os.path.join(BASE, "models/stage2")
OUT_PRED  = os.path.join(BASE, "outputs/stage2")
os.makedirs(OUT_MODEL, exist_ok=True)
os.makedirs(OUT_PRED,  exist_ok=True)

# ── load and merge C0 + C1 for one split ──────────────────────────────────
def load_split(split):
    c0 = pd.read_csv(os.path.join(C0_DIR, f"{split}_with_C0_hho_v2.csv"))
    c1 = pd.read_csv(os.path.join(C1_DIR, f"{split}_with_C1.csv"))

    # normalise join key (url)
    c0["url"] = c0["url"].astype(str).str.strip()
    c1["url"] = c1["url"].astype(str).str.strip()

    # C1 CSV uses column name 'C1' (not 'C1_score')
    c1 = c1.rename(columns={"C1": "C1_score"})

    # deduplicate by URL before merging to avoid many-to-many join explosion
    # (the dataset itself has ~4K duplicate URLs; keep first occurrence)
    c0 = c0.drop_duplicates(subset="url", keep="first")
    c1 = c1.drop_duplicates(subset="url", keep="first")

    merged = pd.merge(c0[["url", "label", "C0_score"]],
                      c1[["url", "C1_score"]],
                      on="url", how="inner").reset_index(drop=True)

    print(f"  [{split}] C0={len(c0):,}  C1={len(c1):,}  merged={len(merged):,}")
    return merged

print("Loading splits …")
tr  = load_split("train")
val = load_split("val")
te  = load_split("test")

# feature matrix: [C0_score, C1_score]
def XY(df):
    return df[["C0_score", "C1_score"]].values, df["label"].values

X_tr,  y_tr  = XY(tr)
X_val, y_val = XY(val)
X_te,  y_te  = XY(te)

# ── train logistic regression on train+val ────────────────────────────────
X_tv = np.vstack([X_tr, X_val])
y_tv = np.concatenate([y_tr, y_val])

print("\nTraining logistic fusion (L2, train+val) …")
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tv, y_tv)

coef_C0 = float(lr.coef_[0][0])
coef_C1 = float(lr.coef_[0][1])
intercept = float(lr.intercept_[0])
print(f"  coef_C0={coef_C0:.4f}  coef_C1={coef_C1:.4f}  intercept={intercept:.4f}")

# ── find best threshold on val ────────────────────────────────────────────
val_proba = lr.predict_proba(X_val)[:, 1]
best_t, best_acc = 0.5, 0.0
for t in np.arange(0.3, 0.95, 0.01):
    acc = accuracy_score(y_val, (val_proba >= t).astype(int))
    if acc > best_acc:
        best_acc, best_t = acc, t
print(f"  Best threshold on val: {best_t:.2f}  val acc={best_acc:.4f}")

# ── evaluate ──────────────────────────────────────────────────────────────
def evaluate(X, y, split):
    proba = lr.predict_proba(X)[:, 1]
    preds = (proba >= best_t).astype(int)
    acc   = accuracy_score(y, preds)
    auc   = roc_auc_score(y, proba)
    cm    = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"  [{split}] acc={acc:.4f}  auc={auc:.4f}  "
          f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    return acc, auc, proba, preds

print("\nEvaluating …")
tr_acc,  tr_auc,  tr_prob,  tr_pred  = evaluate(X_tr,  y_tr,  "train")
val_acc, val_auc, val_prob, val_pred = evaluate(X_val, y_val, "val  ")
te_acc,  te_auc,  te_prob,  te_pred  = evaluate(X_te,  y_te,  "test ")

# ── load old fusion for comparison ────────────────────────────────────────
old_path = os.path.join(OUT_MODEL, "fusion_meta_final.json")
if os.path.exists(old_path):
    with open(old_path) as f:
        old = json.load(f)
    print(f"\nOld fusion  test acc={old.get('test_acc','?')}  auc={old.get('test_auc','?')}")
print(f"New fusion  test acc={te_acc:.4f}  auc={te_auc:.4f}")

# ── save fusion meta ──────────────────────────────────────────────────────
meta = {
    "coef_C0":    coef_C0,
    "coef_C1":    coef_C1,
    "intercept":  intercept,
    "threshold":  best_t,
    "c0_model":   "svm_quadratic_hho_v2 (313 HHO features)",
    "c0_test_acc":0.9398,
    "train_acc":  round(tr_acc,  4),
    "train_auc":  round(tr_auc,  4),
    "val_acc":    round(val_acc, 4),
    "val_auc":    round(val_auc, 4),
    "test_acc":   round(te_acc,  4),
    "test_auc":   round(te_auc,  4),
}
out_json = os.path.join(OUT_MODEL, "fusion_meta_hho.json")
with open(out_json, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved: {out_json}")

# ── save per-split prediction CSVs ────────────────────────────────────────
for split, df, prob, pred in [("train", tr,  tr_prob,  tr_pred),
                               ("val",   val, val_prob, val_pred),
                               ("test",  te,  te_prob,  te_pred)]:
    out = df.copy()
    out["fusion_score"] = prob
    out["fusion_pred"]  = pred
    p = os.path.join(OUT_PRED, f"fusion_hho_{split}_results.csv")
    out.to_csv(p, index=False)
    print(f"  Saved {split} predictions: {p}")

print(f"\n{'='*50}")
print(f"Fusion test acc : {te_acc:.4f}")
print(f"Fusion test AUC : {te_auc:.4f}")
print(f"Threshold used  : {best_t:.2f}")
print(f"coef_C0={coef_C0:.4f}  coef_C1={coef_C1:.4f}")
print(f"{'='*50}")
