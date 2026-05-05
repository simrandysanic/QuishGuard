import os
import pickle
import numpy as np
import pandas as pd

os.makedirs("outputs/stage0", exist_ok=True)

SVM_PATH = "models/stage0/svm_quadratic.pkl"
TEMPLATE_PATH = "outputs/stage0/{split}_with_C0.csv"  # used only for url/order reference
FEATURES_PATH = "outputs/stage0/{split}_features_1000d.npy"
LABELS_PATH = "outputs/stage0/{split}_labels.npy"
OUT_PATH = "outputs/stage0/{split}_with_C0_fusion_v2.csv"

print("=" * 70)
print("REGENERATE C0 (ALIGNED, WITH SAMPLE_ID)")
print("=" * 70)

with open(SVM_PATH, "rb") as f:
    svm = pickle.load(f)
print("Loaded SVM:", SVM_PATH)

for split in ["train", "val", "test"]:
    print(f"\n--- {split.upper()} ---")

    feats = np.load(FEATURES_PATH.format(split=split))
    labels_npy = np.load(LABELS_PATH.format(split=split)).astype(int)

    template = pd.read_csv(TEMPLATE_PATH.format(split=split))
    if "url" not in template.columns:
        raise ValueError(f"{split}: template missing url column")
    if len(template) != len(labels_npy):
        raise ValueError(
            f"{split}: row mismatch template={len(template)} vs labels_npy={len(labels_npy)}"
        )

    proba = svm.predict_proba(feats)
    c0 = proba[:, 1]

    out = pd.DataFrame({
        "sample_id": [f"{split}_{i}" for i in range(len(labels_npy))],
        "url": template["url"].astype(str).values,
        "label": labels_npy,
        "C0": c0
    })

    # Optional drift check against old label column in template (if present)
    if "label" in template.columns:
        mismatch = int((template["label"].astype(int).values != labels_npy).sum())
        print(f"Label mismatch vs old template label: {mismatch}")

    path = OUT_PATH.format(split=split)
    out.to_csv(path, index=False)

    print(f"Saved: {path}")
    print(f"Rows: {len(out):,} | Pos: {int(out['label'].sum()):,} | Neg: {int((out['label']==0).sum()):,}")
    print(f"C0 mean={out['C0'].mean():.6f} std={out['C0'].std():.6f}")

print("\nDone: C0 fusion v2 files created.")
