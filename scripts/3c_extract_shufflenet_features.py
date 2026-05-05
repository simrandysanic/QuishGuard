"""
ShuffleNetV2 Feature Extraction Only (Script 3c)
Loads the already-trained model from models/stage0/shufflenet_features.pth
and extracts features for all 3 splits.

Run this after 3b_train_shufflenet_features.py completes (or crashes after training).
Fast: ~3-5 min per split on GPU.

  conda activate qr_env && cd ~/QuishGuard
  python3 scripts/3c_extract_shufflenet_features.py
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

IMG_SIZE   = 224
BATCH_SIZE = 64   # larger batch fine for inference-only

TRAIN_DIR  = "data/qr_images_v2/train"
VAL_DIR    = "data/qr_images_v2/val"
TEST_DIR   = "data/qr_images_v2/test"

MODEL_SAVE = "models/stage0/shufflenet_features.pth"
OUTPUT_DIR = "outputs/stage0_v2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

assert os.path.exists(MODEL_SAVE), f"Model not found: {MODEL_SAVE}\nRun 3b first."
print(f"Device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class CanonicalQRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []
        class_to_label = {"benign": 0, "malicious": 1}
        for cls in sorted(os.listdir(root_dir)):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            label = class_to_label[cls]
            for fname in sorted(os.listdir(cls_path)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                self.samples.append((
                    os.path.join(cls_path, fname),
                    label,
                    os.path.splitext(fname)[0]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, stem = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, stem


class ShuffleNetV2FeatureExtractor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.shufflenet_v2_x1_0(pretrained=False)
        self.backbone = nn.Sequential(
            base.conv1, base.maxpool,
            base.stage2, base.stage3, base.stage4, base.conv5,
        )
        self.avgpool       = nn.AdaptiveAvgPool2d(1)
        self.feature_layer = nn.Linear(1024, 1000)
        self.relu          = nn.ReLU(inplace=True)
        self.classifier    = nn.Linear(1000, num_classes)

    def extract_features(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.relu(self.feature_layer(x))


model = ShuffleNetV2FeatureExtractor(num_classes=2)
model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Model loaded from {MODEL_SAVE}")


def extract_and_save(split, root_dir):
    ds     = CanonicalQRDataset(root_dir, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=(DEVICE == "cuda"))
    print(f"\n{split.upper()}: {len(ds):,} samples")

    all_features, all_labels, all_stems = [], [], []
    with torch.no_grad():
        for imgs, labels, stems in tqdm(loader, desc=f"Extract {split}"):
            imgs   = imgs.to(DEVICE)
            feats  = model.extract_features(imgs)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_stems.extend(list(stems))

    features = np.vstack(all_features)
    labels   = np.array(all_labels, dtype=int)

    feat_path = os.path.join(OUTPUT_DIR, f"{split}_shufflenet_features_1000d.npy")
    np.save(feat_path, features)

    manifest = pd.DataFrame({
        "url":       all_stems,
        "label":     labels,
        "npy_index": np.arange(len(labels)),
    })
    manifest_path = os.path.join(OUTPUT_DIR, f"{split}_shufflenet_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"  Saved features: {feat_path}  shape={features.shape}")
    print(f"  Saved manifest: {manifest_path}")

    # Alignment check against MobileNet manifest (fixed: explicit length guard)
    mob_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
    if os.path.exists(mob_path):
        mob = pd.read_csv(mob_path)
        if len(mob) != len(manifest):
            print(f"  WARNING: length mismatch vs MobileNet manifest "
                  f"({len(mob)} vs {len(manifest)}) — check image sets")
        else:
            mob_urls  = mob["url"].astype(str).values
            shuf_urls = manifest["url"].astype(str).values
            url_match   = np.mean(mob_urls == shuf_urls) * 100
            label_match = np.mean(mob["label"].values == labels) * 100
            print(f"  URL-order alignment with MobileNet  = {url_match:.2f}%")
            print(f"  Label alignment with MobileNet      = {label_match:.2f}%")
            if url_match < 100.0:
                print(f"  WARNING: ordering mismatch — do NOT run HHO until resolved")
            else:
                print(f"  Alignment OK")
    else:
        print(f"  MobileNet manifest not found — skipping alignment check")


for split, path in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
    extract_and_save(split, path)

print("\nAll splits extracted successfully.")
print("Next step: run scripts/4c_hho_feature_selection.py")
