"""
4d_train_svm_hho.py
Train quadratic-kernel SVM on HHO-selected features from 4c.

Inputs:
  outputs/stage0_v2/hho_reduced_{train,val,test}.npy
  outputs/stage0_v2/hho_labels_{train,val,test}.npy
  outputs/stage0_v2/hho_selected_indices.npy
  outputs/stage0_v2/hho_meta.json

Outputs:
  models/stage0/svm_quadratic_hho.pkl          -- trained SVM
  outputs/stage0_v2/train_with_C0_hho.csv      -- C0 scores for all splits
  outputs/stage0_v2/val_with_C0_hho.csv
  outputs/stage0_v2/test_with_C0_hho.csv
  outputs/stage0_v2/svm_hho_results.json       -- accuracy metrics
"""

import numpy as np
import pandas as pd
import json, os, pickle, time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

BASE     = os.path.expanduser("~/QuishGuard")
FEAT_DIR = os.path.join(BASE, "outputs/stage0_v2")
MODEL_DIR= os.path.join(BASE, "models/stage0")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── load HHO v2 outputs ─────────────────────────────────────────────────────
print("Loading HHO v2-reduced features …")
X_tr  = np.load(os.path.join(FEAT_DIR, "hho_v2_reduced_train.npy"))
y_tr  = np.load(os.path.join(FEAT_DIR, "hho_v2_labels_train.npy"))
X_val = np.load(os.path.join(FEAT_DIR, "hho_v2_reduced_val.npy"))
y_val = np.load(os.path.join(FEAT_DIR, "hho_v2_labels_val.npy"))
X_te  = np.load(os.path.join(FEAT_DIR, "hho_v2_reduced_test.npy"))
y_te  = np.load(os.path.join(FEAT_DIR, "hho_v2_labels_test.npy"))

with open(os.path.join(FEAT_DIR, "hho_v2_meta.json")) as f:
    hho_meta = json.load(f)

n_feat = hho_meta["n_selected"]
print(f"  Features selected by HHO: {n_feat}")
print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

# concatenate train+val for final model training (standard practice)
X_trainval = np.vstack([X_tr, X_val])
y_trainval = np.concatenate([y_tr, y_val])

# ── grid search on train only first (to pick C) ────────────────────────────
print("\nGrid search for SVM C parameter (train set, quadratic kernel) …")
t0 = time.time()

# polynomial kernel of degree 2 = quadratic SVM (as in paper)
param_grid = {"C": [0.1, 0.5, 1.0, 5.0, 10.0]}
base_svm = SVC(
    kernel="poly",
    degree=2,
    coef0=1,
    gamma="scale",
    probability=True,   # needed for decision scores → C0
    random_state=42
)

gs = GridSearchCV(
    base_svm,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)
gs.fit(X_tr, y_tr)

best_C   = gs.best_params_["C"]
best_cv  = gs.best_score_
print(f"\nBest C={best_C}  CV acc={best_cv:.4f}  ({(time.time()-t0)/60:.1f} min)")

# ── re-train on train+val with best C ──────────────────────────────────────
print(f"\nRetraining on train+val (n={len(X_trainval):,}) with C={best_C} …")
t1 = time.time()
final_svm = SVC(
    kernel="poly",
    degree=2,
    coef0=1,
    gamma="scale",
    C=best_C,
    probability=True,
    random_state=42
)
final_svm.fit(X_trainval, y_trainval)
print(f"  Done in {(time.time()-t1)/60:.1f} min")

# ── evaluate ───────────────────────────────────────────────────────────────
def evaluate(svm, X, y, split_name):
    preds  = svm.predict(X)
    proba  = svm.predict_proba(X)[:, 1]   # P(malicious)
    acc    = accuracy_score(y, preds)
    auc    = roc_auc_score(y, proba)
    print(f"  [{split_name}] acc={acc:.4f}  auc={auc:.4f}")
    return preds, proba, acc, auc

print("\nEvaluating …")
tr_preds, tr_proba, tr_acc, tr_auc = evaluate(final_svm, X_tr,  y_tr,  "train")
va_preds, va_proba, va_acc, va_auc = evaluate(final_svm, X_val, y_val, "val  ")
te_preds, te_proba, te_acc, te_auc = evaluate(final_svm, X_te,  y_te,  "test ")

# ── save model ────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "svm_quadratic_hho_v2.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_svm, f)
print(f"\nSaved model: {model_path}")

# ── save C0 score CSVs (feeds into fusion retraining) ─────────────────────
# Load original manifests to get url+label for C0 csv format.
# The hho_labels files are aligned with the inner-join order from 4c.
# We need to match back to the manifest to reconstruct the CSV.
# Simple approach: just write index + label + C0 score, then use url from manifest.

import hashlib

def save_c0_csv(split, proba, labels):
    mob_mf  = pd.read_csv(os.path.join(FEAT_DIR, f"{split}_manifest.csv"))
    shuf_mf = pd.read_csv(os.path.join(FEAT_DIR, f"{split}_shufflenet_manifest.csv"))

    mob_mf["url_hash"] = mob_mf["url"].apply(
        lambda u: hashlib.md5(u.encode("utf-8")).hexdigest()
    )
    mob_mf  = mob_mf.rename(columns={"npy_index": "mob_idx"})
    shuf_mf = shuf_mf.rename(columns={"npy_index": "shuf_idx"})

    merged = pd.merge(
        mob_mf [["url", "url_hash", "label", "mob_idx"]],
        shuf_mf[["url",             "shuf_idx"]].rename(columns={"url": "url_hash"}),
        on="url_hash", how="inner"
    ).reset_index(drop=True)

    assert len(merged) == len(proba), \
        f"[{split}] Mismatch: merged={len(merged)}, proba={len(proba)}"

    merged["C0_score"] = proba
    merged["C0_pred"]  = (proba >= 0.5).astype(int)

    out_path = os.path.join(FEAT_DIR, f"{split}_with_C0_hho_v2.csv")
    merged[["url", "label", "C0_score", "C0_pred"]].to_csv(out_path, index=False)
    print(f"  Saved {split} C0 CSV: {out_path}  ({len(merged):,} rows)")

print("\nSaving C0 score CSVs …")
save_c0_csv("train", tr_proba, y_tr)
save_c0_csv("val",   va_proba, y_val)
save_c0_csv("test",  te_proba, y_te)

# ── save results JSON ─────────────────────────────────────────────────────
results = {
    "n_features_hho":  n_feat,
    "best_C":          best_C,
    "cv_acc":          round(best_cv, 4),
    "train_acc":       round(tr_acc, 4),
    "train_auc":       round(tr_auc, 4),
    "val_acc":         round(va_acc, 4),
    "val_auc":         round(va_auc, 4),
    "test_acc":        round(te_acc, 4),
    "test_auc":        round(te_auc, 4),
    "paper_target_test_acc": 0.9589
}
out_json = os.path.join(FEAT_DIR, "svm_hho_v2_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {out_json}")
print(f"\n{'='*50}")
print(f"Test acc:  {te_acc:.4f}  (paper target: 0.9589)")
print(f"Test AUC:  {te_auc:.4f}")
if te_acc >= 0.9589:
    print("✅ Meets or exceeds paper benchmark!")
else:
    delta = 0.9589 - te_acc
    print(f"⚠️  {delta:.4f} below paper benchmark — consider re-running HHO")
    print("   with V-shaped transfer function targeting ~200-300 features.")
print(f"{'='*50}")
