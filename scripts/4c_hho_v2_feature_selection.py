"""
4c_hho_v2_feature_selection.py
HHO feature selection — V-shaped transfer function (V3: |x/√(1+x²)|)
with soft penalty for feature counts outside [100, 500].

Targets ~200-300 features from 2000-d combined MobileNet+ShuffleNet space.
Paper benchmark: ~253 features, test acc 95.89%.

Same outputs as 4c but in separate files:
  outputs/stage0_v2/hho_v2_selected_indices.npy
  outputs/stage0_v2/hho_v2_reduced_{split}.npy
  outputs/stage0_v2/hho_v2_labels_{split}.npy
  outputs/stage0_v2/hho_v2_meta.json
"""

import numpy as np
import pandas as pd
import json, os, time, math, warnings, hashlib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.expanduser("~/QuishGuard")
FEAT_DIR = os.path.join(BASE, "outputs/stage0_v2")
OUT_DIR  = FEAT_DIR

# ── HHO hyper-parameters ───────────────────────────────────────────────────
N_HAWKS      = 30
MAX_ITER     = 150
SUBSAMPLE    = 50_000
SEED         = 42
# Feature count soft-penalty window (paper targets ~253)
MIN_FEAT     = 100
MAX_FEAT     = 500
PENALTY_W    = 0.15   # weight: 0.15 * fraction-outside-window added to error
rng          = np.random.default_rng(SEED)
D            = 2000

# ── data loading ──────────────────────────────────────────────────────────
def load_split(split):
    mob_feat  = np.load(os.path.join(FEAT_DIR, f"{split}_features_1000d.npy"),          mmap_mode="r")
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
        shuf_mf[["url",               "shuf_idx"]].rename(columns={"url": "url_hash"}),
        on="url_hash", how="inner"
    ).reset_index(drop=True)

    print(f"  [{split}] MobileNet={len(mob_mf):,}  ShuffleNet={len(shuf_mf):,}"
          f"  inner-join={len(merged):,}")

    mob_rows  = mob_feat [merged["mob_idx"].values]
    shuf_rows = shuf_feat[merged["shuf_idx"].values]
    X = np.hstack([mob_rows, shuf_rows]).astype(np.float32)
    y = merged["label"].values.astype(np.int32)
    return X, y


# ── V-shaped transfer function (V3) ───────────────────────────────────────
def v_transfer(x):
    """V3: |x / sqrt(1 + x²)|  — maps any real → [0, 1)"""
    return np.abs(x / np.sqrt(1.0 + x * x))

def binarize_v(x, current_bin, rng):
    """
    Flip each bit with probability v_transfer(x[i]).
    Large |x| → high flip probability → explores both 0→1 and 1→0.
    Maintains higher average feature density than sigmoid binarization.
    """
    prob = v_transfer(x)
    flip = rng.random(x.shape) < prob
    return np.where(flip, 1 - current_bin, current_bin).astype(np.float64)


# ── fitness ────────────────────────────────────────────────────────────────
def fitness(position_binary, X_sub, y_sub, X_val, y_val):
    sel = np.where(position_binary)[0]
    n   = len(sel)
    if n == 0:
        return 1.0

    clf = LinearSVC(C=1.0, max_iter=2000, random_state=SEED)
    clf.fit(X_sub[:, sel], y_sub)
    err = 1.0 - (clf.predict(X_val[:, sel]) == y_val).mean()

    # soft penalty keeps feature count inside [MIN_FEAT, MAX_FEAT]
    if n < MIN_FEAT:
        penalty = PENALTY_W * (MIN_FEAT - n) / MIN_FEAT
    elif n > MAX_FEAT:
        penalty = PENALTY_W * (n - MAX_FEAT) / D
    else:
        penalty = 0.0

    return float(err) + penalty


# ── Levy flight ────────────────────────────────────────────────────────────
def levy(D, rng):
    beta  = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(D) * sigma
    v = rng.standard_normal(D)
    return u / (np.abs(v) ** (1 / beta))


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("Loading train split …")
    X_tr,  y_tr  = load_split("train")
    print("Loading val split …")
    X_val, y_val = load_split("val")
    print("Loading test split …")
    X_te,  y_te  = load_split("test")

    print("Scaling features …")
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    sub_idx = rng.choice(len(X_tr), min(SUBSAMPLE, len(X_tr)), replace=False)
    X_sub   = X_tr[sub_idx]
    y_sub   = y_tr[sub_idx]

    print(f"\nRunning HHO v2 (V-shaped): {N_HAWKS} hawks, {MAX_ITER} iters, D={D}")
    print(f"  Feature penalty window: [{MIN_FEAT}, {MAX_FEAT}]  weight={PENALTY_W}")
    print(f"  Train subsample: {len(X_sub):,}  |  Val: {len(X_val):,}\n")

    # ── initialise positions with ~15% density ────────────────────────────
    # Start binary positions with ~15% of bits = 1 (close to paper's 253/2000)
    init_density = 0.15
    positions = rng.uniform(-2, 2, (N_HAWKS, D))
    bin_pos   = (rng.random((N_HAWKS, D)) < init_density).astype(np.float64)

    fit_vals = np.array([fitness(bin_pos[i], X_sub, y_sub, X_val, y_val)
                         for i in range(N_HAWKS)])

    best_idx    = int(np.argmin(fit_vals))
    rabbit_pos  = positions[best_idx].copy()
    rabbit_bin  = bin_pos[best_idx].copy()
    rabbit_fit  = fit_vals[best_idx]

    history = []

    for t in range(MAX_ITER):
        E0  = 2.0 * rng.random() - 1.0
        E   = 2.0 * E0 * (1.0 - t / MAX_ITER)
        abE = abs(E)

        for i in range(N_HAWKS):
            if abE >= 1.0:
                # exploration
                if rng.random() >= 0.5:
                    rand_hawk = positions[rng.integers(N_HAWKS)]
                    r1, r2    = rng.random(), rng.random()
                    positions[i] = rand_hawk - r1 * np.abs(rand_hawk - 2 * r2 * rabbit_pos)
                else:
                    r3, r4 = rng.random(), rng.random()
                    positions[i] = (rabbit_pos - positions.mean(axis=0)
                                    - r3 * rng.uniform(-2, 2, D) * r4)
                bin_pos[i]  = binarize_v(positions[i], bin_pos[i], rng)
                fit_vals[i] = fitness(bin_pos[i], X_sub, y_sub, X_val, y_val)

            else:
                J   = 2.0 * (1.0 - rng.random())
                r5  = rng.random()
                Lev = levy(D, rng)

                if r5 >= 0.5 and abE >= 0.5:
                    # soft besiege
                    new_pos     = rabbit_pos - E * np.abs(J * rabbit_pos - positions[i])
                    new_bin     = binarize_v(new_pos, bin_pos[i], rng)
                    new_fit     = fitness(new_bin, X_sub, y_sub, X_val, y_val)
                    positions[i] = new_pos; bin_pos[i] = new_bin; fit_vals[i] = new_fit

                elif r5 >= 0.5 and abE < 0.5:
                    # hard besiege
                    new_pos     = rabbit_pos - E * np.abs(rabbit_pos - positions[i])
                    new_bin     = binarize_v(new_pos, bin_pos[i], rng)
                    new_fit     = fitness(new_bin, X_sub, y_sub, X_val, y_val)
                    positions[i] = new_pos; bin_pos[i] = new_bin; fit_vals[i] = new_fit

                elif r5 < 0.5 and abE >= 0.5:
                    # soft besiege + Levy
                    Y    = rabbit_pos - E * np.abs(J * rabbit_pos - positions[i])
                    Z    = Y + rng.standard_normal(D) * Lev
                    yb   = binarize_v(Y, bin_pos[i], rng)
                    zb   = binarize_v(Z, bin_pos[i], rng)
                    f_Y  = fitness(yb, X_sub, y_sub, X_val, y_val)
                    f_Z  = fitness(zb, X_sub, y_sub, X_val, y_val)
                    if f_Y < fit_vals[i]:
                        positions[i] = Y; bin_pos[i] = yb; fit_vals[i] = f_Y
                    if f_Z < fit_vals[i]:
                        positions[i] = Z; bin_pos[i] = zb; fit_vals[i] = f_Z

                else:
                    # hard besiege + Levy
                    Y    = rabbit_pos - E * np.abs(J * rabbit_pos - positions.mean(axis=0))
                    Z    = Y + rng.standard_normal(D) * Lev
                    yb   = binarize_v(Y, bin_pos[i], rng)
                    zb   = binarize_v(Z, bin_pos[i], rng)
                    f_Y  = fitness(yb, X_sub, y_sub, X_val, y_val)
                    f_Z  = fitness(zb, X_sub, y_sub, X_val, y_val)
                    if f_Y < fit_vals[i]:
                        positions[i] = Y; bin_pos[i] = yb; fit_vals[i] = f_Y
                    if f_Z < fit_vals[i]:
                        positions[i] = Z; bin_pos[i] = zb; fit_vals[i] = f_Z

        # update rabbit
        best_i = int(np.argmin(fit_vals))
        if fit_vals[best_i] < rabbit_fit:
            rabbit_pos = positions[best_i].copy()
            rabbit_bin = bin_pos[best_i].copy()
            rabbit_fit = fit_vals[best_i]

        n_feat  = int(rabbit_bin.sum())
        elapsed = (time.time() - t0) / 60
        eta     = elapsed / (t + 1) * (MAX_ITER - t - 1)
        history.append({"iter": t + 1, "err": rabbit_fit, "n_features": n_feat})
        print(f"  [{t+1:3d}/{MAX_ITER}] err={rabbit_fit:.4f}  "
              f"features={n_feat:4d}  elapsed={elapsed:.1f}m  eta={eta:.1f}m",
              flush=True)

    # ── save outputs ───────────────────────────────────────────────────────
    sel_indices = np.where(rabbit_bin)[0]
    print(f"\nSelected {len(sel_indices)} features out of {D}")

    np.save(os.path.join(OUT_DIR, "hho_v2_selected_indices.npy"), sel_indices)

    for split, X, y in [("train", X_tr, y_tr),
                         ("val",   X_val, y_val),
                         ("test",  X_te,  y_te)]:
        np.save(os.path.join(OUT_DIR, f"hho_v2_reduced_{split}.npy"),  X[:, sel_indices])
        np.save(os.path.join(OUT_DIR, f"hho_v2_labels_{split}.npy"),   y)
        print(f"  Saved {split}: shape={X[:, sel_indices].shape}")

    meta = {
        "n_selected":    int(len(sel_indices)),
        "best_val_fit":  float(rabbit_fit),
        "best_val_acc":  float(1.0 - rabbit_fit),  # approx (includes penalty)
        "n_hawks":       N_HAWKS,
        "max_iter":      MAX_ITER,
        "subsample":     SUBSAMPLE,
        "transfer_fn":   "V3",
        "penalty_window":[MIN_FEAT, MAX_FEAT],
        "penalty_weight":PENALTY_W,
        "total_min":     round((time.time() - t0) / 60, 1),
        "history":       history
    }
    with open(os.path.join(OUT_DIR, "hho_v2_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Features={len(sel_indices)}  Total time={meta['total_min']:.1f} min")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
