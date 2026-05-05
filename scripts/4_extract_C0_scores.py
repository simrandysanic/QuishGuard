"""
Script 4 FINAL FIX: Extract C0 Scores
Uses ImageFolder (same as training) to avoid
hash() non-determinism bug on Linux
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

IMG_SIZE   = 224
BATCH_SIZE = 64

MODEL_PATH = "models/stage0/checkpoints/mobilenet_best.pth"
QR_DIR     = "data/qr_images"
CSV_DIR    = "data/processed"
OUTPUT_DIR = "outputs/stage0/features"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Identical to training
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==============================
# LOAD MODEL
# ==============================
print("\nLoading MobileNetV2...")
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
print("✅ Model loaded\n")

# ==============================
# EXTRACTION
# ImageFolder gives us images +
# labels but NOT urls.
# We get urls from CSV separately
# and align by sort order.
# ==============================
def extract_C0(split):

    print(f"{'='*50}")
    print(f"Extracting C0: {split.upper()}")
    print(f"{'='*50}")

    # Load images via ImageFolder (same as training)
    dataset = datasets.ImageFolder(
        root      = os.path.join(QR_DIR, split),
        transform = transform
    )

    # ImageFolder sorts files alphabetically
    # Get the file paths in that same order
    img_paths = [s[0] for s in dataset.samples]
    img_labels = [s[1] for s in dataset.samples]

    print(f"  Class mapping : {dataset.class_to_idx}")
    print(f"  Total images  : {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,      # MUST be False to keep order
        num_workers = 4
    )

    all_labels_img = []
    all_C0         = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {split}"):
            images = images.to(DEVICE)
            logits = model(images)
            C0     = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_labels_img.extend(labels.numpy())
            all_C0.extend(C0)

    # Now load URLs from CSV
    # Match them to images using filename
    # Extract URL from filename via the CSV
    csv_df = pd.read_csv(os.path.join(CSV_DIR, f"{split}.csv"))

    # Build output using image file order
    # img_paths gives us the actual files processed
    # We extract the hash from filename and match
    df_out = pd.DataFrame({
        'img_path':   img_paths,
        'label_img':  all_labels_img,
        'C0':         all_C0
    })

    # Add class name from path
    df_out['cls_name'] = df_out['img_path'].apply(
        lambda p: os.path.basename(os.path.dirname(p))
    )

    # Remap label:
    # ImageFolder: benign=0, malicious=1 (alphabetical)
    # Verify this matches our CSV labels
    print(f"  Label dist from ImageFolder:")
    print(f"    0 (benign)   : {(df_out['label_img']==0).sum():,}")
    print(f"    1 (malicious): {(df_out['label_img']==1).sum():,}")

    # Save with image path as reference
    out_path = os.path.join(OUTPUT_DIR, f"{split}_with_C0.csv")
    df_out[['img_path', 'label_img', 'C0']].to_csv(out_path, index=False)

    print(f"  ✅ Saved  : {out_path}")
    print(f"  C0 mean  : {df_out['C0'].mean():.4f}")
    print(f"  C0 std   : {df_out['C0'].std():.4f}")
    print(f"  C0 min   : {df_out['C0'].min():.4f}")
    print(f"  C0 max   : {df_out['C0'].max():.4f}\n")

    return df_out


# ==============================
# MAIN
# ==============================
for split in ['train', 'val', 'test']:
    extract_C0(split)

print("="*50)
print("✅ C0 EXTRACTION COMPLETE")
print("="*50)
print(f"\nFiles in {OUTPUT_DIR}:")
print("  train_with_C0.csv")
print("  val_with_C0.csv")
print("  test_with_C0.csv")
print("\nNext: Train BERT (script 5)")

