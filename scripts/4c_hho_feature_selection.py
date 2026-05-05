"""
4c_hho_feature_selection.py
Harris Hawks Optimization (HHO) feature selection on 2000-d combined
MobileNet + ShuffleNet features.

Paper target: ~253 features selected from 2000-d combined space.
Fitness = LinearSVC 5-fold CV error on 50K subsample of train set.

Outputs:
  outputs/stage0_v2/hho_selected_indices.npy   -- selected feature indices
  outputs/stage0_v2/hho_reduced_{split}.npy    -- reduced features per split
  outputs/stage0_v2/hho_labels_{split}.npy     -- labels per split
  outputs/stage0_v2/hho_meta.json              -- meta info
"""

import numpy as np
import pandas as pd
import json, os, time, math, warnings, hashlib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/QuishGuard")
FEAT_DIR = os.path.join(BASE, "outputs/stage0_v2")
OUT_DIR  = FEAT_DIR                      # save alongside other stage-0 outputs

# ── HHO hyper-parameters (from paper) ─────────────────────────────────────
N_HAWKS   = 30
MAX_ITER  = 150
SUBSAMPLE = 50_000      # train subsample for fast fitness eval
SEED      = 42
rng       = np.random.default_rng(SEED)

# ── helper: load & align MobileNet + ShuffleNet for one split ──────────────
def load_split(split):
    """
    Returns (X, y) where X.shape = (N, 2000), y.shape = (N,)
    Alignment is done via inner-join on URL so 240K vs 235K mismatch is safe.
    """
    mob_feat = np.load(os.path.join(FEAT_DIR, f"{split}_features_1000d.npy"),
                       mmap_mode="r")
    mob_mf   = pd.read_csv(os.path.join(FEAT_DIR, f"{split}_manifest.csv"))

    shuf_feat = np.load(os.path.join(FEAT_DIR, f"{split}_shufflenet_features_1000d.npy"),
                        mmap_mode="r")
    shuf_mf   = pd.read_csv(os.path.join(FEAT_DIR, f"{split}_shufflenet_manifest.csv"))

    # rename npy_index columns before merge
    mob_mf  = mob_mf.rename(columns={"npy_index": "mob_idx"})
    shuf_mf = shuf_mf.rename(columns={"npy_index": "shuf_idx"})

    # MobileNet manifest stores actual URLs; ShuffleNet manifest stores MD5 hashes.
    # Compute md5(url) for MobileNet rows so the join keys match.
    mob_mf["url_hash"] = mob_mf["url"].apply(
        lambda u: hashlib.md5(u.encode("utf-8")).hexdigest()
    )

    merged = pd.merge(
        mob_mf [["url_hash", "label", "mob_idx"]],
        shuf_mf[["url",               "shuf_idx"]].rename(columns={"url": "url_hash"}),
        on="url_hash", how="inner"
    ).reset_index(drop=True)

    print(f"  [{split}] MobileNet={len(mob_mf):,}  ShuffleNet={len(shuf_mf):,}"
          f"  inner-join={len(merged):,}")

    mob_rows  = mob_feat [merged["mob_idx"].values]   # (N,1000)
    shuf_rows = shuf_feat[merged["shuf_idx"].values]  # (N,1000)
    X = np.hstack([mob_rows, shuf_rows]).astype(np.float32)  # (N,2000)
    y = merged["label"].values.astype(np.int32)
    return X, y


# ── fitness function ───────────────────────────────────────────────────────
def fitness(position_binary, X_tr_sub, y_tr_sub, X_val, y_val):
    """
    Binary position → subset of 2000 features.
    Returns error rate on val set (lower = better).
    """
    sel = np.where(position_binary)[0]
    if len(sel) == 0:
        return 1.0
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=SEED)
    clf.fit(X_tr_sub[:, sel], y_tr_sub)
    err = 1.0 - (clf.predict(X_val[:, sel]) == y_val).mean()
    return float(err)


# ── sigmoid binarization ───────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def binarize(x, rng):
    return (rng.random(x.shape) < sigmoid(x)).astype(np.float64)


# ── Levy flight (Cauchy distribution as paper uses for simplicity) ──────────
def levy(D, rng):
    beta = 1.5
    sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
             (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = rng.standard_normal(D) * sigma
    v = rng.standard_normal(D)
    return u / (np.abs(v)**(1/beta))


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    D  = 2000  # feature dimension

    # load data
    print("Loading train split …")
    X_tr, y_tr = load_split("train")
    print("Loading val split …")
    X_val, y_val = load_split("val")
    print("Loading test split …")
    X_te, y_te  = load_split("test")

    # scale
    print("Scaling features …")
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # fixed subsample of train for fitness (speed)
    sub_idx = rng.choice(len(X_tr), min(SUBSAMPLE, len(X_tr)), replace=False)
    X_sub = X_tr[sub_idx]
    y_sub = y_tr[sub_idx]

    print(f"\nRunning HHO: {N_HAWKS} hawks, {MAX_ITER} iterations, D={D}")
    print(f"  Train subsample for fitness: {len(X_sub):,}")
    print(f"  Val set: {len(X_val):,}\n")

    # ── initialise hawk positions (continuous) ─────────────────────────────
    positions  = rng.uniform(-2, 2, (N_HAWKS, D))   # continuous
    bin_pos    = np.array([binarize(positions[i], rng) for i in range(N_HAWKS)])

    # evaluate initial fitness
    fit_vals = np.array([fitness(bin_pos[i], X_sub, y_sub, X_val, y_val)
                         for i in range(N_HAWKS)])

    best_idx   = int(np.argmin(fit_vals))
    rabbit_pos = positions[best_idx].copy()
    rabbit_bin = bin_pos[best_idx].copy()
    rabbit_fit = fit_vals[best_idx]

    history = []

    for t in range(MAX_ITER):
        t_start = time.time()

        # escaping energy
        E0  = 2.0 * rng.random() - 1.0          # random in [-1,1]
        E   = 2.0 * E0 * (1.0 - t / MAX_ITER)   # decreasing
        abE = abs(E)

        for i in range(N_HAWKS):
            # ── exploration ─────────────────────────────────────────────
            if abE >= 1.0:
                if rng.random() >= 0.5:
                    rand_hawk = positions[rng.integers(N_HAWKS)]
                    Xm = X_tr[rng.choice(len(X_tr), 1)].ravel()[:D]  # not actually used in standard impl
                    # standard: X(t+1) = X_rand - r1*|X_rand - 2*r2*X_rabbit|
                    r1, r2 = rng.random(), rng.random()
                    positions[i] = rand_hawk - r1 * np.abs(rand_hawk - 2*r2*rabbit_pos)
                else:
                    r3, r4 = rng.random(), rng.random()
                    # jump toward rabbit with random offset
                    Xm = rng.uniform(positions.min(axis=0), positions.max(axis=0))
                    positions[i] = (rabbit_pos - positions.mean(axis=0)
                                    - r3 * (rng.uniform(-2,2,D)) * r4)

            # ── exploitation ────────────────────────────────────────────
            else:
                J   = 2.0 * (1.0 - rng.random())
                r5  = rng.random()
                Lev = levy(D, rng)

                if r5 >= 0.5 and abE >= 0.5:
                    # soft besiege
                    delta = rabbit_pos - positions[i]
                    positions[i] = delta - E * np.abs(J * rabbit_pos - positions[i])

                elif r5 >= 0.5 and abE < 0.5:
                    # hard besiege
                    positions[i] = rabbit_pos - E * np.abs(rabbit_pos - positions[i])

                elif r5 < 0.5 and abE >= 0.5:
                    # soft besiege + Levy dive
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - positions[i])
                    Z = Y + rng.standard_normal(D) * Lev
                    y_bin = binarize(Y, rng)
                    z_bin = binarize(Z, rng)
                    f_Y = fitness(y_bin, X_sub, y_sub, X_val, y_val)
                    f_Z = fitness(z_bin, X_sub, y_sub, X_val, y_val)
                    if f_Y < fit_vals[i]:
                        positions[i] = Y; bin_pos[i] = y_bin; fit_vals[i] = f_Y
                    if f_Z < fit_vals[i]:
                        positions[i] = Z; bin_pos[i] = z_bin; fit_vals[i] = f_Z
                    continue   # skip the normal binarize/eval below

                else:
                    # hard besiege + Levy dive
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - positions.mean(axis=0))
                    Z = Y + rng.standard_normal(D) * Lev
                    y_bin = binarize(Y, rng)
                    z_bin = binarize(Z, rng)
                    f_Y = fitness(y_bin, X_sub, y_sub, X_val, y_val)
                    f_Z = fitness(z_bin, X_sub, y_sub, X_val, y_val)
                    if f_Y < fit_vals[i]:
                        positions[i] = Y; bin_pos[i] = y_bin; fit_vals[i] = f_Y
                    if f_Z < fit_vals[i]:
                        positions[i] = Z; bin_pos[i] = z_bin; fit_vals[i] = f_Z
                    continue

            # binarize and evaluate (for exploration / soft-hard besiege)
            bin_pos[i] = binarize(positions[i], rng)
            fit_vals[i] = fitness(bin_pos[i], X_sub, y_sub, X_val, y_val)

        # update rabbit
        best_i = int(np.argmin(fit_vals))
        if fit_vals[best_i] < rabbit_fit:
            rabbit_pos = positions[best_i].copy()
            rabbit_bin = bin_pos[best_i].copy()
            rabbit_fit = fit_vals[best_i]

        n_feat = int(rabbit_bin.sum())
        elapsed = (time.time() - t0) / 60
        eta     = elapsed / (t+1) * (MAX_ITER - t - 1)
        history.append({"iter": t+1, "err": rabbit_fit, "n_features": n_feat})
        print(f"  [{t+1:3d}/{MAX_ITER}] err={rabbit_fit:.4f}  "
              f"features={n_feat:4d}  elapsed={elapsed:.1f}m  eta={eta:.1f}m",
              flush=True)

    # ── save outputs ───────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    sel_indices = np.where(rabbit_bin)[0]
    print(f"\nSelected {len(sel_indices)} features out of {D}")
    np.save(os.path.join(OUT_DIR, "hho_selected_indices.npy"), sel_indices)

    for split, X, y in [("train", X_tr, y_tr),
                         ("val",   X_val, y_val),
                         ("test",  X_te,  y_te)]:
        np.save(os.path.join(OUT_DIR, f"hho_reduced_{split}.npy"),  X[:, sel_indices])
        np.save(os.path.join(OUT_DIR, f"hho_labels_{split}.npy"),   y)
        print(f"  Saved {split}: shape={X[:, sel_indices].shape}")

    meta = {
        "n_selected":   int(len(sel_indices)),
        "best_val_err": float(rabbit_fit),
        "best_val_acc": float(1.0 - rabbit_fit),
        "n_hawks":      N_HAWKS,
        "max_iter":     MAX_ITER,
        "subsample":    SUBSAMPLE,
        "total_min":    round((time.time() - t0) / 60, 1),
        "history":      history
    }
    with open(os.path.join(OUT_DIR, "hho_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Best val acc = {1-rabbit_fit:.4f}  |  "
          f"Total time = {meta['total_min']:.1f} min")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
