"""
4e_svm_full_features.py
Sanity check: train quadratic SVM on full 2000-d (MobileNet+ShuffleNet)
features WITHOUT HHO selection, to determine if HHO/feature selection
is the bottleneck or if 93.98% is the data ceiling.

Uses a 100K subsample of train for speed (full 240K would take hours).
"""

import numpy as np
import pandas as pd
import hashlib, os, json, time, pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

BASE     = os.path.expanduser("~/QuishGuard")
FEAT_DIR = os.path.join(BASE, "outputs/stage0_v2")
MODEL_DIR= os.path.join(BASE, "models/stage0")

rng = np.random.default_rng(42)

# ── load & align (same as 4c) ──────────────────────────────────────────────
def load_split(split):
    mob_feat  = np.load(os.path.join(FEAT_DIR, f"{split}_features_1000d.npy"),           mmap_mode="r")
    mob_mf    = pd.read_csv(os.path.join(FEAT_DIR, f"{split}_manifest.csv"))
    shuf_feat = np.load(os.path.join(FEAT_DIR, f"{split}_shufflenet_features_1000d.npy"), mmap_mode="r")
    shuf_mf   = pd.read_csv(os.path.join(FEAT_DIR, f"{split}_shufflenet_manifest.csv"))

    mob_mf["url_hash"] = mob_mf["url"].apply(
        lambda u: hashlib.md5(u.encode("utf-8")).hexdigest()
    )
    mob_mf  = mob_mf.rename(columns={"npy_index": "mob_idx"})
    shuf_mf = shuf_mf.rename(columns={"npy_index": "shuf_idx"})

    merged = pd.merge(
        mob_mf [["url_hash", "label", "mob_idx"]],
        shuf_mf[["url", "shuf_idx"]].rename(columns={"url": "url_hash"}),
        on="url_hash", how="inner"
    ).reset_index(drop=True)

    print(f"  [{split}] inner-join={len(merged):,}")
    X = np.hstack([mob_feat[merged["mob_idx"].values],
                   shuf_feat[merged["shuf_idx"].values]]).astype(np.float32)
    y = merged["label"].values.astype(np.int32)
    return X, y

t0 = time.time()

print("Loading data …")
X_tr,  y_tr  = load_split("train")
X_val, y_val = load_split("val")
X_te,  y_te  = load_split("test")

print("Scaling …")
scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)
X_te  = scaler.transform(X_te)

# ── 1. LinearSVC on full 2000-d (fast baseline) ───────────────────────────
print("\n[1] LinearSVC on full 2000-d features …")
lsvc = LinearSVC(C=1.0, max_iter=3000, random_state=42)
lsvc.fit(X_tr, y_tr)
lsvc_val = accuracy_score(y_val, lsvc.predict(X_val))
lsvc_te  = accuracy_score(y_te,  lsvc.predict(X_te))
print(f"  val acc={lsvc_val:.4f}  test acc={lsvc_te:.4f}  ({(time.time()-t0)/60:.1f}min)")

# ── 2. Quadratic SVM on 50K subsample (speed test) ────────────────────────
print("\n[2] Quadratic SVM (C=10) on 50K train subsample …")
sub_idx = rng.choice(len(X_tr), 50_000, replace=False)
X_sub   = X_tr[sub_idx]
y_sub   = y_tr[sub_idx]

t1 = time.time()
qsvm_sub = SVC(kernel="poly", degree=2, coef0=1, gamma="scale",
               C=10.0, probability=False, random_state=42)
qsvm_sub.fit(X_sub, y_sub)
sub_val = accuracy_score(y_val, qsvm_sub.predict(X_val))
sub_te  = accuracy_score(y_te,  qsvm_sub.predict(X_te))
print(f"  val acc={sub_val:.4f}  test acc={sub_te:.4f}  ({(time.time()-t1)/60:.1f}min)")

# ── 3. Quadratic SVM on HHO v2 features (313-d) — already done but confirm ─
print("\n[3] Quadratic SVM (C=10) on HHO v2 313-d features (train only, no val) …")
X_hho_tr = np.load(os.path.join(FEAT_DIR, "hho_v2_reduced_train.npy"))
y_hho_tr = np.load(os.path.join(FEAT_DIR, "hho_v2_labels_train.npy"))
X_hho_val= np.load(os.path.join(FEAT_DIR, "hho_v2_reduced_val.npy"))
y_hho_val= np.load(os.path.join(FEAT_DIR, "hho_v2_labels_val.npy"))
X_hho_te = np.load(os.path.join(FEAT_DIR, "hho_v2_reduced_test.npy"))
y_hho_te = np.load(os.path.join(FEAT_DIR, "hho_v2_labels_test.npy"))

sub2_idx = rng.choice(len(X_hho_tr), 50_000, replace=False)
t2 = time.time()
qsvm_hho = SVC(kernel="poly", degree=2, coef0=1, gamma="scale",
               C=10.0, probability=False, random_state=42)
qsvm_hho.fit(X_hho_tr[sub2_idx], y_hho_tr[sub2_idx])
hho_val = accuracy_score(y_hho_val, qsvm_hho.predict(X_hho_val))
hho_te  = accuracy_score(y_hho_te,  qsvm_hho.predict(X_hho_te))
print(f"  val acc={hho_val:.4f}  test acc={hho_te:.4f}  ({(time.time()-t2)/60:.1f}min)")

# ── summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'Model':<35} {'Val':>8} {'Test':>8}")
print(f"{'-'*55}")
print(f"{'LinearSVC — 2000-d full':<35} {lsvc_val:>8.4f} {lsvc_te:>8.4f}")
print(f"{'QuadSVM(50K) — 2000-d full':<35} {sub_val:>8.4f} {sub_te:>8.4f}")
print(f"{'QuadSVM(50K) — HHO v2 313-d':<35} {hho_val:>8.4f} {hho_te:>8.4f}")
print(f"{'Paper target C0':<35} {'':>8} {'0.9589':>8}")
print(f"{'='*55}")
print(f"Total: {(time.time()-t0)/60:.1f}min")
