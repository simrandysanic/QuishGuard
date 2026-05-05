"""
7c_retrain_fusion_hho_clean.py

Retrain the Stage 2 logistic fusion using the HHO-based Stage 0 scores,
but with a clean paper-safe protocol:

1. Train on train only
2. Select threshold on val only
3. Evaluate once on test

The script is intentionally flexible about the QuishGuard root because the
local experiment workspace sometimes stores artifacts under ~/QuishGuard and
sometimes under ~/QuishGuard/scripts.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


@dataclass(frozen=True)
class ResolvedPaths:
    experiment_root: str
    c0_dir: str
    c1_dir: str
    model_dir: str
    pred_dir: str


def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def first_existing(paths: Iterable[str], label: str) -> str:
    candidates = [expand(path) for path in paths]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve {label}. Checked: {candidates}"
    )


def resolve_paths() -> ResolvedPaths:
    env_root = os.environ.get("QUISHGUARD_BASE")
    root_candidates = [
        env_root,
        "~/QuishGuard",
        "~/QuishGuard/scripts",
    ]
    root_candidates = [candidate for candidate in root_candidates if candidate]

    experiment_root = first_existing(root_candidates, "QuishGuard base directory")

    c0_dir = first_existing(
        [
            os.path.join(experiment_root, "outputs", "stage0_v2"),
            os.path.join(experiment_root, "scripts", "outputs", "stage0_v2"),
        ],
        "Stage 0 HHO score directory",
    )
    c1_dir = first_existing(
        [
            os.path.join(experiment_root, "outputs", "stage1", "features"),
            os.path.join(experiment_root, "scripts", "outputs", "stage1", "features"),
        ],
        "Stage 1 score directory",
    )

    model_dir = expand(os.path.join(experiment_root, "outputs", "stage2_models_clean"))
    pred_dir = expand(os.path.join(experiment_root, "outputs", "stage2_clean"))
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    return ResolvedPaths(
        experiment_root=expand(experiment_root),
        c0_dir=c0_dir,
        c1_dir=c1_dir,
        model_dir=model_dir,
        pred_dir=pred_dir,
    )


def load_split(paths: ResolvedPaths, split: str) -> pd.DataFrame:
    c0_path = os.path.join(paths.c0_dir, f"{split}_with_C0_hho_v2.csv")
    c1_path = os.path.join(paths.c1_dir, f"{split}_with_C1.csv")

    c0 = pd.read_csv(c0_path)
    c1 = pd.read_csv(c1_path)

    c0["url"] = c0["url"].astype(str).str.strip()
    c1["url"] = c1["url"].astype(str).str.strip()

    if "C1" in c1.columns and "C1_score" not in c1.columns:
        c1 = c1.rename(columns={"C1": "C1_score"})

    if "C0" in c0.columns and "C0_score" not in c0.columns:
        c0 = c0.rename(columns={"C0": "C0_score"})

    required_c0 = {"url", "label", "C0_score"}
    required_c1 = {"url", "C1_score"}
    if not required_c0.issubset(c0.columns):
        raise ValueError(f"{split}: C0 file missing columns {required_c0 - set(c0.columns)}")
    if not required_c1.issubset(c1.columns):
        raise ValueError(f"{split}: C1 file missing columns {required_c1 - set(c1.columns)}")

    c0 = c0.drop_duplicates(subset="url", keep="first")
    c1 = c1.drop_duplicates(subset="url", keep="first")

    merged = pd.merge(
        c0[["url", "label", "C0_score"]],
        c1[["url", "C1_score"]],
        on="url",
        how="inner",
    ).reset_index(drop=True)

    print(
        f"  [{split}] C0={len(c0):,} C1={len(c1):,} merged={len(merged):,}"
    )
    return merged


def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame[["C0_score", "C1_score"]].values, frame["label"].values


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.arange(0.30, 0.95, 0.01):
        preds = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def evaluate(
    model: LogisticRegression,
    threshold: float,
    features: np.ndarray,
    labels: np.ndarray,
    split: str,
) -> tuple[float, float, np.ndarray, np.ndarray, dict[str, int]]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    stats = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    print(
        f"  [{split}] acc={accuracy:.4f} auc={auc:.4f} "
        f"TP={tp} TN={tn} FP={fp} FN={fn}"
    )
    return accuracy, auc, probabilities, predictions, stats


def main() -> None:
    paths = resolve_paths()

    print("=" * 60)
    print("CLEAN HHO FUSION RETRAINING")
    print("=" * 60)
    print(f"Experiment root : {paths.experiment_root}")
    print(f"C0 dir          : {paths.c0_dir}")
    print(f"C1 dir          : {paths.c1_dir}")
    print(f"Model output    : {paths.model_dir}")
    print(f"Prediction out  : {paths.pred_dir}")

    print("\nLoading splits...")
    train_frame = load_split(paths, "train")
    val_frame = load_split(paths, "val")
    test_frame = load_split(paths, "test")

    x_train, y_train = xy(train_frame)
    x_val, y_val = xy(val_frame)
    x_test, y_test = xy(test_frame)

    print("\nTraining logistic fusion on train only...")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    coef_c0 = float(model.coef_[0][0])
    coef_c1 = float(model.coef_[0][1])
    intercept = float(model.intercept_[0])
    print(
        f"  coef_C0={coef_c0:.4f} coef_C1={coef_c1:.4f} intercept={intercept:.4f}"
    )

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold, val_best_acc = select_threshold(y_val, val_prob)
    print(f"  Best validation threshold={threshold:.2f} val_acc={val_best_acc:.4f}")

    print("\nEvaluating clean fusion...")
    train_acc, train_auc, train_prob, train_pred, train_stats = evaluate(
        model, threshold, x_train, y_train, "train"
    )
    val_acc, val_auc, _, val_pred, val_stats = evaluate(
        model, threshold, x_val, y_val, "val"
    )
    test_acc, test_auc, test_prob, test_pred, test_stats = evaluate(
        model, threshold, x_test, y_test, "test"
    )

    meta = {
        "protocol": "train_only_fit__val_only_threshold__single_test_eval",
        "experiment_root": paths.experiment_root,
        "coef_C0": coef_c0,
        "coef_C1": coef_c1,
        "intercept": intercept,
        "threshold": threshold,
        "c0_model": "svm_quadratic_hho_v2 (313 HHO features)",
        "train_acc": round(train_acc, 4),
        "train_auc": round(train_auc, 4),
        "val_acc": round(val_acc, 4),
        "val_auc": round(val_auc, 4),
        "test_acc": round(test_acc, 4),
        "test_auc": round(test_auc, 4),
        "train_confusion": train_stats,
        "val_confusion": val_stats,
        "test_confusion": test_stats,
    }

    meta_path = os.path.join(paths.model_dir, "fusion_meta_hho_clean.json")
    model_json_path = os.path.join(paths.model_dir, "fusion_model_hho_clean.json")
    model_pickle_path = os.path.join(paths.model_dir, "fusion_model_hho_clean.pkl")

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    with open(model_json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "coef_C0": coef_c0,
                "coef_C1": coef_c1,
                "intercept": intercept,
                "threshold": threshold,
            },
            handle,
            indent=2,
        )
    with open(model_pickle_path, "wb") as handle:
        pickle.dump(model, handle)

    for split, frame, probabilities, predictions in [
        ("train", train_frame, train_prob, train_pred),
        ("val", val_frame, val_prob, val_pred),
        ("test", test_frame, test_prob, test_pred),
    ]:
        output = frame.copy()
        output["fusion_score"] = probabilities
        output["fusion_pred"] = predictions
        output_path = os.path.join(paths.pred_dir, f"fusion_hho_clean_{split}_results.csv")
        output.to_csv(output_path, index=False)
        print(f"  Saved {split} predictions: {output_path}")

    print("\n" + "=" * 60)
    print(f"Clean fusion test acc : {test_acc:.4f}")
    print(f"Clean fusion test AUC : {test_auc:.4f}")
    print(f"Threshold used        : {threshold:.2f}")
    print(f"coef_C0={coef_c0:.4f} coef_C1={coef_c1:.4f}")
    print(f"Saved model pickle    : {model_pickle_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()