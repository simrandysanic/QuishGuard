"""
ShuffleNetV2 Feature Extractor Training (Script 3b)
Following: Alaca & Celik (2023) - exact replication of Stage 0 CNN pipeline

Mirrors 3_train_mobilenet_features.py but for ShuffleNetV2_x1_0.
Paper reference: "node 202 layer" -> equivalent to conv5 output (1024-d)
-> projected to 1000-d feature vector (matches MobileNetV2 feature dim)

Outputs:
  models/stage0/shufflenet_features.pth
  outputs/stage0_v2/{split}_shufflenet_features_1000d.npy
  outputs/stage0_v2/{split}_shufflenet_manifest.csv

Run under tmux - takes 4-12 hrs on GPU:
  tmux new -s shufflenet
  conda activate qr_env && cd ~/QuishGuard
  python3 scripts/3b_train_shufflenet_features.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# ==============================
# CONFIG - Paper aligned (same as MobileNet script)
# ==============================
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 16
LR         = 1e-4

TRAIN_DIR  = "data/qr_images_v2/train"
VAL_DIR    = "data/qr_images_v2/val"
TEST_DIR   = "data/qr_images_v2/test"

MODEL_SAVE = "models/stage0/shufflenet_features.pth"
OUTPUT_DIR = "outputs/stage0_v2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ==============================
# TRANSFORMS - ImageNet pretrained
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# CANONICAL DATASET
# Loads images in sorted class/filename order - reproducible.
# Stems (MD5 hashes) used as URL keys for manifest alignment.
# class_to_label: benign=0, malicious=1
# ==============================
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


def make_loader(root_dir, shuffle=False):
    ds = CanonicalQRDataset(root_dir, transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=4, pin_memory=(DEVICE == "cuda"))


train_loader = make_loader(TRAIN_DIR, shuffle=True)
val_loader   = make_loader(VAL_DIR,   shuffle=False)
test_loader  = make_loader(TEST_DIR,  shuffle=False)

print(f"Train: {len(train_loader.dataset):,} | "
      f"Val: {len(val_loader.dataset):,} | "
      f"Test: {len(test_loader.dataset):,}")

# ==============================
# MODEL
# ShuffleNetV2_x1_0: conv5 -> avgpool -> 1024-d -> Linear(1024,1000) -> 1000-d
# Paper: "node 202 layer" produces 1000 features, same dim as MobileNetV2
# ==============================
class ShuffleNetV2FeatureExtractor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.shufflenet_v2_x1_0(pretrained=True)
        # Keep backbone up to and including conv5
        self.backbone = nn.Sequential(
            base.conv1,
            base.maxpool,
            base.stage2,
            base.stage3,
            base.stage4,
            base.conv5,
        )
        self.avgpool       = nn.AdaptiveAvgPool2d(1)
        self.feature_layer = nn.Linear(1024, 1000)
        self.relu          = nn.ReLU(inplace=True)
        self.classifier    = nn.Linear(1000, num_classes)

    def forward(self, x):
        x        = self.backbone(x)
        x        = self.avgpool(x)
        x        = torch.flatten(x, 1)
        features = self.relu(self.feature_layer(x))
        logits   = self.classifier(features)
        return logits, features

    def extract_features(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.relu(self.feature_layer(x))


model = ShuffleNetV2FeatureExtractor(num_classes=2).to(DEVICE)

# ==============================
# LOSS & OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# TRAIN
# ==============================
def train_one_epoch(loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Val  "):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


print("\n" + "="*60)
print("TRAINING ShuffleNetV2_x1_0  (16 epochs, batch=32)")
print("="*60)

best_val_acc = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-"*40)
    tr_loss, tr_acc = train_one_epoch(train_loader)
    vl_loss, vl_acc = evaluate(val_loader)
    print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
    print(f"  Val    loss={vl_loss:.4f}  acc={vl_acc:.4f}")
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"  Best model saved  (val_acc={vl_acc:.4f})")

print(f"\nTraining complete.  Best val acc: {best_val_acc:.4f}")

# ==============================
# FEATURE EXTRACTION
# ==============================
print("\n" + "="*60)
print("EXTRACTING ShuffleNetV2 FEATURES")
print("="*60)

model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
model.eval()


def extract_and_save(loader, split):
    all_features, all_labels, all_stems = [], [], []
    with torch.no_grad():
        for imgs, labels, stems in tqdm(loader, desc=f"Extract {split}"):
            imgs = imgs.to(DEVICE)
            feats = model.extract_features(imgs)
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

    # Alignment check against existing MobileNet manifest
    mob_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
    if os.path.exists(mob_path):
        mob = pd.read_csv(mob_path)
        url_match   = (mob["url"].values == manifest["url"].values).mean() * 100
        label_match = (mob["label"].values == labels).mean() * 100
        print(f"  {split}: URL-order alignment with MobileNet = {url_match:.2f}%")
        print(f"  {split}: Label alignment with MobileNet     = {label_match:.2f}%")
        if url_match < 100.0:
            print(f"  WARNING: ordering mismatch - check CanonicalQRDataset before running HHO")
    else:
        print(f"  {split}: MobileNet manifest not found - skipping alignment check")

    print(f"  Saved {feat_path}  shape={features.shape}")
    print(f"  Saved {manifest_path}")


for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
    extract_and_save(loader, split)
    print()

print("ShuffleNetV2 training + feature extraction complete.")
print("Next step: run scripts/4c_hho_feature_selection.py")
