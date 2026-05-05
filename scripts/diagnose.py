# Check actual training results
import pandas as pd

# Your training script should have printed per-epoch results
# Let's check what the actual test accuracy was during training

# Load a few QR images and check predictions
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load(
    "models/stage0/checkpoints/mobilenet_best.pth",
    map_location=DEVICE
))
model = model.to(DEVICE)
model.eval()

# Test on validation set (same data training saw)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_data = datasets.ImageFolder("data/qr_images/val", transform=transform)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits = model(imgs)
        preds = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"Validation accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Expected from training: ~92.65%")
print(f"Got from fusion: 50.28%")

if acc > 0.85:
    print("\n✅ Model IS trained! Problem is in C0 extraction.")
else:
    print("\n🚨 Model is NOT trained. Need to retrain.")
