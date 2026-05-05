import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==============================
# CONFIG  (UNCHANGED)
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4

TRAIN_DIR = "data/qr_images/train"
VAL_DIR   = "data/qr_images/val"
TEST_DIR  = "data/qr_images/test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# TRANSFORMS (UNCHANGED)
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
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

# ==============================
# MODEL (MobileNetV2)
# ==============================
model = models.mobilenet_v2(pretrained=True)

model.classifier[1] = nn.Linear(model.last_channel, 1)

model = model.to(DEVICE)

# ==============================
# LOSS & OPTIMIZER (UNCHANGED)
# ==============================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# TRAIN FUNCTION
# ==============================
def train_one_epoch(loader):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ==============================
# VALIDATION
# ==============================
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds = torch.sigmoid(outputs) > 0.5

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item()

    acc = correct / total
    return loss_total / len(loader), acc

# ==============================
# TRAIN LOOP
# ==============================
best_val_acc = 0

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "mobilenet_best.pth")
        print("✅ Best model saved")

# ==============================
# TEST EVALUATION
# ==============================
print("\nLoading best model for testing...")
model.load_state_dict(torch.load("mobilenet_best.pth"))

test_loss, test_acc = evaluate(test_loader)

print("\n==============================")
print(f"TEST ACCURACY: {test_acc:.4f}")
print("==============================")
