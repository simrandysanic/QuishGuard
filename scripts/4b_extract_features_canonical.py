"""
Canonical Feature Extraction (Script B)
========================================
Extracts 1000-d MobileNetV2 features in CANONICAL CSV ORDER.

WHY THIS SCRIPT EXISTS:
  The original 3_train_mobilenet_features.py used ImageFolder, which
  walks subdirectories alphabetically – so the order of rows in the saved
  .npy files matched the folder walk, NOT the canonical CSV row order.
  This made URL-based alignment impossible.

WHAT THIS SCRIPT DOES:
  1. Loads the *already-trained* model (no retraining).
  2. Iterates over each split's manifest CSV (produced by
     2_generate_qr_codes_v2.py) row by row in canonical order.
  3. Extracts the 1000-d features for each image.
  4. Saves features as  outputs/stage0_v2/{split}_features_1000d.npy
     where row i corresponds exactly to manifest row i.
  5. Saves a companion manifest  outputs/stage0_v2/{split}_manifest.csv
     with columns: url, label, npy_index  (npy_index == CSV row index).

RESULT:
  For every split, features[i] ↔ manifest.url[i] ↔ manifest.label[i]
  with zero ambiguity.

Following: Alaca & Çelik (2023) – MobileNetV2 architecture unchanged.
"""

import os
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
IMG_SIZE       = 224
BATCH_SIZE     = 256   # batch for fast GPU extraction (no gradients)
QR_IMAGES_DIR  = "data/qr_images_v2"
MODEL_PATH     = "models/stage0/mobilenet_features.pth"
INPUT_DIR      = "data/qr_images_v2"      # where manifests live
OUTPUT_DIR     = "outputs/stage0_v2"
SPLITS         = ["train", "val", "test"]
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_TO_DIR   = {0: "benign", 1: "malicious"}
NUM_WORKERS    = 4          # DataLoader parallel workers


# ============================================================
# DATASET  – canonical-order QR image loader
# ============================================================

class CanonicalQRDataset(Dataset):
    """
    Loads QR images in the exact order of the manifest CSV.
    Missing images are returned as zero tensors (flagged in mask).
    """
    def __init__(self, manifest_df: pd.DataFrame, base_dir: str,
                 split: str, transform):
        self.manifest  = manifest_df.reset_index(drop=True)
        self.base_dir  = base_dir
        self.split     = split
        self.transform = transform
        self.label_to_dir = LABEL_TO_DIR

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row      = self.manifest.iloc[idx]
        url      = str(row["url"]).strip()
        label    = int(row["label"])
        dir_name = self.label_to_dir[label]
        filename = hashlib.md5(url.encode("utf-8")).hexdigest() + ".png"
        img_path = os.path.join(self.base_dir, self.split, dir_name, filename)

        if not os.path.exists(img_path):
            # Return zero tensor + flag
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label, idx, False

        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
            return tensor, label, idx, True
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label, idx, False


# ============================================================
# MODEL  (exact same architecture as 3_train_mobilenet_features.py)
# ============================================================

class MobileNetV2FeatureExtractor(nn.Module):
    """
    MobileNetV2 for 1000-d feature extraction.
    Architecture: backbone (1280-d) → Linear(1280,1000) → ReLU → Linear(1000,2)
    Features = 1000-d output after ReLU  (paper: 'Logits layer').
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False)   # weights loaded below
        self.features      = mobilenet.features
        self.avgpool       = nn.AdaptiveAvgPool2d(1)
        self.feature_layer = nn.Linear(1280, 1000)
        self.relu          = nn.ReLU(inplace=True)
        self.classifier    = nn.Linear(1000, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1000-d feature vector (no classifier)."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_layer(x)
        x = self.relu(x)
        return x


# ============================================================
# TRANSFORMS  (identical to training)
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225],    # ImageNet std
    ),
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path: str, device: str) -> MobileNetV2FeatureExtractor:
    print(f"Loading model: {model_path}")
    assert os.path.exists(model_path), (
        f"Model not found at {model_path}\n"
        "Run 3_train_mobilenet_features.py first (only needed once – already done)."
    )
    model = MobileNetV2FeatureExtractor(num_classes=2)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  ✅ Model loaded (device={device})")
    return model


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ".png"


def extract_split(split: str, model: MobileNetV2FeatureExtractor, device: str):
    manifest_path = os.path.join(INPUT_DIR, f"{split}_manifest.csv")
    assert os.path.exists(manifest_path), (
        f"Manifest not found: {manifest_path}\n"
        "Run 2_generate_qr_codes_v2.py first."
    )

    manifest_df = pd.read_csv(manifest_path)
    n = len(manifest_df)
    print(f"\n{split.upper()}: {n:,} samples")

    # Build Dataset + DataLoader (batched, canonical order, no shuffle)
    dataset = CanonicalQRDataset(manifest_df, INPUT_DIR, split, transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

    all_features = np.zeros((n, 1000), dtype=np.float32)
    failed_idxs  = []

    with torch.no_grad():
        for tensors, labels, idxs, valids in tqdm(loader, desc=f"  Extracting {split}"):
            tensors = tensors.to(device)
            feats   = model.extract_features(tensors)           # (B, 1000)
            feats_np = feats.cpu().numpy()

            for b_i, (canonical_idx, valid) in enumerate(zip(idxs.numpy(),
                                                              valids.numpy())):
                if valid:
                    all_features[canonical_idx] = feats_np[b_i]
                else:
                    failed_idxs.append(canonical_idx)

    if failed_idxs:
        failed_idxs = sorted(set(failed_idxs))
        print(f"  ⚠ {len(failed_idxs)} images failed/missing – removing from output.")
        manifest_df = manifest_df.drop(index=failed_idxs).reset_index(drop=True)
        all_features = np.delete(all_features, failed_idxs, axis=0)

    manifest_df["npy_index"] = range(len(manifest_df))

    # ------ Save ------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feat_path    = os.path.join(OUTPUT_DIR, f"{split}_features_1000d.npy")
    manifest_out = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")

    np.save(feat_path, all_features)
    manifest_df[["url", "label", "npy_index"]].to_csv(manifest_out, index=False)

    print(f"  Features saved  : {feat_path}  {all_features.shape}")
    print(f"  Manifest saved  : {manifest_out}  ({len(manifest_df):,} rows)")
    print(f"  Label dist.     : {manifest_df['label'].value_counts().to_dict()}")

    # Sanity: first row
    first = manifest_df.iloc[0]
    print(f"  Row 0  →  url={str(first['url'])[:60]!r}  "
          f"label={first['label']}  npy[0][:5]={all_features[0, :5].round(4)}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CANONICAL FEATURE EXTRACTION (Script B)")
    print("Using: existing models/stage0/mobilenet_features.pth")
    print("Order: canonical CSV row order (NOT ImageFolder walk)")
    print("=" * 60)

    model = load_model(MODEL_PATH, DEVICE)

    for split in SPLITS:
        extract_split(split, model, DEVICE)

    print("\n" + "=" * 60)
    print("✅  Feature extraction complete.")
    print(f"Output dir : {OUTPUT_DIR}/")
    print("NEXT STEP  : Run 5b_extract_C0_canonical.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
