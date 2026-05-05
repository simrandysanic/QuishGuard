"""
MobileNetV2 Feature Extractor Training
Following: Alaca & Çelik (2023)

Key differences from before:
1. Extract features, don't do end-to-end classification
2. Use SVM on features (paper's approach)
3. Proper ImageNet normalization for pretrained model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# ==============================
# CONFIG - Paper aligned
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 16
LR = 1e-4

TRAIN_DIR = "data/qr_images/train"
VAL_DIR   = "data/qr_images/val"
TEST_DIR  = "data/qr_images/test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# TRANSFORMS - ImageNet pretrained
# CRITICAL: Use ImageNet mean/std
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

# ==============================
# DATASETS
# ==============================
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_data   = datasets.ImageFolder(VAL_DIR,   transform=transform)
test_data  = datasets.ImageFolder(TEST_DIR,  transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
print(f"Classes: {train_data.class_to_idx}")

# ==============================
# MODEL - Feature Extractor
# Following paper: extract from penultimate layer
# ==============================
class MobileNetV2FeatureExtractor(nn.Module):
    """
    MobileNetV2 for feature extraction
    Following: Alaca & Çelik (2023)
    
    Architecture:
    MobileNetV2 backbone (1280 features)
    → Linear(1280, 1000) [paper: 1000 deep features]
    → Linear(1000, 2)    [binary classification for training]
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Keep feature extractor
        self.features = mobilenet.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        
        # Paper: 1000 deep features from Logits layer
        self.feature_layer = nn.Linear(1280, 1000)
        self.relu = nn.ReLU(inplace=True)
        
        # Classification head (for training only)
        self.classifier = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 1000-d features (this is what we extract later)
        features = self.feature_layer(x)
        features = self.relu(features)
        
        # Classification (for training)
        logits = self.classifier(features)
        
        return logits, features
    
    def extract_features(self, x):
        """Extract 1000-d features without classification"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.feature_layer(x)
        features = self.relu(features)
        return features

model = MobileNetV2FeatureExtractor(num_classes=2).to(DEVICE)

# ==============================
# LOSS & OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# TRAIN FUNCTION
# ==============================
def train_one_epoch(loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Training"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(imgs)  # ignore features during training
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

# ==============================
# VALIDATION
# ==============================
def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _ = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# ==============================
# TRAIN LOOP
# ==============================
best_val_acc = 0

for epoch in range(EPOCHS):
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("="*50)
    
    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/stage0/mobilenet_features.pth")
        print(f"✅ Best model saved (Val Acc: {val_acc:.4f})")

print(f"\nBest Val Acc: {best_val_acc:.4f}")

# ==============================
# EXTRACT FEATURES FOR SVM
# ==============================
print("\n" + "="*50)
print("EXTRACTING FEATURES FOR SVM")
print("="*50)

model.load_state_dict(torch.load("models/stage0/mobilenet_features.pth"))
model.eval()

def extract_all_features(loader, split_name):
    """Extract 1000-d features for all images"""
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Extracting {split_name}"):
            imgs = imgs.to(DEVICE)
            features = model.extract_features(imgs)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels

train_features, train_labels = extract_all_features(train_loader, "train")
val_features, val_labels = extract_all_features(val_loader, "val")
test_features, test_labels = extract_all_features(test_loader, "test")

# Save features
np.save("outputs/stage0/train_features_1000d.npy", train_features)
np.save("outputs/stage0/train_labels.npy", train_labels)
np.save("outputs/stage0/val_features_1000d.npy", val_features)
np.save("outputs/stage0/val_labels.npy", val_labels)
np.save("outputs/stage0/test_features_1000d.npy", test_features)
np.save("outputs/stage0/test_labels.npy", test_labels)

print(f"\nSaved features:")
print(f"  Train: {train_features.shape}")
print(f"  Val:   {val_features.shape}")
print(f"  Test:  {test_features.shape}")
