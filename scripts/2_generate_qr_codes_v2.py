"""
QR Code Generation v2 - Paper-Aligned with Deterministic Filenames
Following: Alaca & Çelik (2023)

KEY FIX:
  OLD (broken): abs(hash(url))  → non-deterministic across processes/runs
                                  (PYTHONHASHSEED changes every Python launch)
  NEW (correct): hashlib.md5(url.encode('utf-8')).hexdigest()
                                → always the same for the same URL, everywhere

QR PARAMETERS (paper-aligned):
  - error_correction = ERROR_CORRECT_L  (≈7% data recovery, smallest QR)
  - box_size = 10                       (10px per module)
  - border   = 4                        (4-module quiet zone)
  - Images saved as RGB PNG, readable by MobileNetV2

ALSO WRITES per-split manifest CSV:
  url, label, split, img_path, filename
  → guaranteed URL↔image↔label traceability throughout pipeline

OUTPUT: data/qr_images_v2/{split}/{benign|malicious}/{md5hex}.png
"""

import os
import hashlib
import qrcode
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_L
QR_BOX_SIZE         = 10
QR_BORDER           = 4

DATA_DIR   = "data/processed"
OUTPUT_DIR = "data/qr_images_v2"
SPLITS     = ["train", "val", "test"]

LABEL_TO_DIR = {0: "benign", 1: "malicious"}


# ============================================================
# HELPERS
# ============================================================

def url_to_filename(url: str) -> str:
    """Return a stable, deterministic PNG filename for this URL."""
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ".png"


def normalize_label(label) -> int:
    """Map any label representation to 0 (benign) or 1 (malicious)."""
    if isinstance(label, (int, float)):
        return int(label)
    label = str(label).strip().lower()
    if label in ("0", "benign", "legitimate", "safe", "normal"):
        return 0
    if label in ("1", "malicious", "phishing", "spam", "malware"):
        return 1
    raise ValueError(f"Unrecognised label value: {label!r}")


def make_qr_image(data: str) -> Image.Image:
    """
    Generate a QR code image for *data* using the paper's parameters,
    returned as an RGB PIL Image (MobileNetV2 expects 3-channel input).
    """
    qr = qrcode.QRCode(
        error_correction=QR_ERROR_CORRECTION,
        box_size=QR_BOX_SIZE,
        border=QR_BORDER,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img.convert("RGB")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("QR CODE GENERATION v2")
    print("Following: Alaca & Çelik (2023)")
    print(f"  error_correction : ERROR_CORRECT_L")
    print(f"  box_size         : {QR_BOX_SIZE}")
    print(f"  border           : {QR_BORDER}")
    print(f"  hash method      : hashlib.md5  (deterministic)")
    print(f"  output dir       : {OUTPUT_DIR}/")
    print("=" * 60)

    for split in SPLITS:
        csv_path = os.path.join(DATA_DIR, f"{split}.csv")
        df = pd.read_csv(csv_path)

        print(f"\n{split.upper()}: {len(df):,} rows  |  columns: {df.columns.tolist()}")

        assert "url"   in df.columns, f"'url' column missing from {csv_path}"
        assert "label" in df.columns, f"'label' column missing from {csv_path}"

        # Create output sub-directories
        for dir_name in LABEL_TO_DIR.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, dir_name), exist_ok=True)

        manifest_rows = []
        errors        = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Generating {split}"):
            url       = str(row["url"]).strip()
            label_int = normalize_label(row["label"])
            dir_name  = LABEL_TO_DIR[label_int]
            filename  = url_to_filename(url)
            img_path  = os.path.join(OUTPUT_DIR, split, dir_name, filename)

            # Resume-safe: skip if image already exists
            if not os.path.exists(img_path):
                try:
                    img = make_qr_image(url)
                    img.save(img_path)
                except Exception as exc:
                    print(f"\n  ⚠ FAILED: {url[:70]!r} → {exc}")
                    errors += 1
                    continue

            manifest_rows.append({
                "url":      url,
                "label":    label_int,
                "split":    split,
                "img_path": os.path.join(split, dir_name, filename),
                "filename": filename,
            })

        # ---------- Write manifest ----------
        manifest_df   = pd.DataFrame(manifest_rows)
        manifest_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
        manifest_df.to_csv(manifest_path, index=False)

        ok = len(manifest_rows)
        print(f"  ✅  Generated: {ok:,}  |  Errors: {errors:,}")
        print(f"  Manifest : {manifest_path}")
        print(f"  Labels   : {manifest_df['label'].value_counts().to_dict()}")

    print("\n" + "=" * 60)
    print("✅  QR generation complete.")
    print(f"Output root : {OUTPUT_DIR}/")
    print("NEXT STEP   : Run 4b_extract_features_canonical.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
