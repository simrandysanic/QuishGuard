import os
import pandas as pd
import qrcode
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

BASE_DIR = "data/processed"
OUT_DIR = "data/qr_images"

IMG_SIZE = 224

# -----------------------------
# QR generation function
# -----------------------------
def generate_qr(args):
    url, label, split = args

    label_name = "malicious" if label == 1 else "benign"
    save_dir = os.path.join(OUT_DIR, split, label_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4
        )

        qr.add_data(url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        filename = os.path.join(save_dir, f"{abs(hash(url))}.png")
        img.save(filename)

    except Exception:
        pass


# -----------------------------
# Split processor
# -----------------------------
def process_split(split):

    print(f"\nProcessing {split}")

    df = pd.read_csv(os.path.join(BASE_DIR, f"{split}.csv"))

    data = [(row.url, row.label, split) for row in df.itertuples()]

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(generate_qr, data), total=len(data)))


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n✅ QR generation complete")
