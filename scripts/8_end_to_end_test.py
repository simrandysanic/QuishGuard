"""
End-to-End Pipeline Test (Script 8)
=====================================
Simulates REAL-WORLD usage of the full QuishGuard pipeline:

  QR image  →  decode URL (cv2 QRCodeDetector)
            →  MobileNetV2 features  →  SVM  → C0
            →  BERT tokenisation     →  BERT → C1
            →  LogisticRegression fusion     → final prediction

The URL is read FROM THE QR CODE IMAGE — not taken from any CSV.
Ground truth is looked up from the manifest after decoding.

SAMPLE: 25% of all QR codes across all splits (random, reproducible).

INPUT:
  data/qr_images_v2/{split}/{benign|malicious}/{md5}.png
  data/qr_images_v2/{split}_manifest.csv         (for ground truth lookup)

MODELS:
  models/stage0/mobilenet_features.pth
  models/stage0/svm_quadratic.pkl
  models/stage1/best/                            (BERT)
  models/stage2/fusion_model.pkl

OUTPUT:
  outputs/e2e_test/results.csv                   (url, true_label, C0, C1, fused_pred, correct)
  outputs/e2e_test/summary.txt
"""

import os
import pickle
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SAMPLE_FRACTION = 0.25          # sample 25% of all QR codes
RANDOM_SEED     = 42
BATCH_SIZE_BERT = 64
BATCH_SIZE_CNN  = 256
IMG_SIZE        = 224
BERT_MAX_LEN    = 128
NUM_WORKERS     = 4

QR_BASE_DIR  = "data/qr_images_v2"
SPLITS       = ["train", "val", "test"]
LABEL_TO_DIR = {0: "benign", 1: "malicious"}

MODEL_MOBILENET = "models/stage0/mobilenet_features.pth"
MODEL_SVM       = "models/stage0/svm_quadratic.pkl"
MODEL_BERT      = "models/stage1/best"
MODEL_FUSION    = "models/stage2/fusion_model.pkl"

OUTPUT_DIR = "outputs/e2e_test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# STEP 1 — BUILD SAMPLE LIST  (image paths + ground truth labels)
# ============================================================

def build_sample_list(fraction: float):
    """
    Walk all split manifests, collect every image path + ground truth label,
    then sample `fraction` of them randomly.
    """
    rows = []
    for split in SPLITS:
        manifest_path = os.path.join(QR_BASE_DIR, f"{split}_manifest.csv")
        assert os.path.exists(manifest_path), \
            f"Manifest missing: {manifest_path}\nRun 2_generate_qr_codes_v2.py first."
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            label    = int(row["label"])
            dir_name = LABEL_TO_DIR[label]
            # img_path column is relative: "train/benign/<md5>.png"
            full_path = os.path.join(QR_BASE_DIR, str(row["img_path"]))
            if os.path.exists(full_path):
                rows.append({
                    "img_path":   full_path,
                    "true_label": label,
                    "true_url":   str(row["url"]).strip(),   # kept for verification only
                    "split":      split,
                })

    all_df = pd.DataFrame(rows)
    n_sample = int(len(all_df) * fraction)
    sampled  = all_df.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)
    return sampled


# ============================================================
# STEP 2 — DECODE URL FROM QR IMAGE
# ============================================================

def decode_qr(img_path: str) -> str:
    """
    Read QR code image and return the decoded URL string.
    Uses OpenCV's built-in QRCodeDetector.
    Returns empty string if decoding fails.
    """
    img = cv2.imread(img_path)
    if img is None:
        return ""
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)
    return data if data else ""


# ============================================================
# STEP 3 — MOBILENET + SVM → C0
# ============================================================

class MobileNetV2FeatureExtractor(nn.Module):
    """Same architecture as training — must match exactly."""
    def __init__(self):
        super().__init__()
        mobilenet          = models.mobilenet_v2(pretrained=False)
        self.features      = mobilenet.features
        self.avgpool       = nn.AdaptiveAvgPool2d(1)
        self.feature_layer = nn.Linear(1280, 1000)
        self.relu          = nn.ReLU(inplace=True)
        # No classifier head — feature extraction only

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_layer(x)
        x = self.relu(x)
        return x


mobilenet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImagePathDataset(Dataset):
    """Loads QR images from a list of file paths (in order)."""
    def __init__(self, paths, transform):
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img    = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, idx, True
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), idx, False


def extract_cnn_features(img_paths: list, model: nn.Module) -> np.ndarray:
    """Extract 1000-d MobileNet features for each image. Returns (N, 1000)."""
    dataset = ImagePathDataset(img_paths, mobilenet_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE_CNN, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    features = np.zeros((len(img_paths), 1000), dtype=np.float32)

    with torch.no_grad():
        for tensors, idxs, valids in tqdm(loader, desc="  CNN features"):
            tensors = tensors.to(DEVICE)
            feats   = model(tensors).cpu().numpy()
            for b_i, (idx, valid) in enumerate(zip(idxs.numpy(), valids.numpy())):
                if valid:
                    features[idx] = feats[b_i]

    return features


# ============================================================
# STEP 4 — BERT → C1
# ============================================================

class URLListDataset(Dataset):
    """Tokenises a list of URL strings for BERT inference."""
    def __init__(self, urls: list, tokenizer, max_length: int):
        self.urls      = urls
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.urls[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "token_type_ids": enc["token_type_ids"].flatten(),
        }


def get_bert_scores(urls: list, tokenizer, bert_model) -> np.ndarray:
    """Return C1 = P(malicious) for each URL in list. Returns (N,)."""
    dataset = URLListDataset(urls, tokenizer, BERT_MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE_BERT, shuffle=False,
                         num_workers=0)   # num_workers=0 for tokeniser thread-safety

    all_C1 = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  BERT scores"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)

            outputs = bert_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
            probs = torch.softmax(outputs.logits, dim=1)
            all_C1.extend(probs[:, 1].cpu().numpy())

    return np.array(all_C1, dtype=np.float32)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("END-TO-END PIPELINE TEST")
    print("Flow: QR image → decode URL → CNN+SVM (C0) + BERT (C1) → Fusion")
    print(f"Sample: {SAMPLE_FRACTION*100:.0f}% of all QR images")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ------ Build sample list ------
    print("\n[1/6] Building sample list...")
    sample_df = build_sample_list(SAMPLE_FRACTION)
    print(f"  Total QR images sampled: {len(sample_df):,}")
    print(f"  Label dist: {sample_df['true_label'].value_counts().to_dict()}")
    print(f"  Splits: {sample_df['split'].value_counts().to_dict()}")

    # ------ Decode URLs from QR images ------
    print("\n[2/6] Decoding URLs from QR images...")
    decoded_urls = []
    failed_decode = 0

    for path in tqdm(sample_df["img_path"].tolist(), desc="  Decoding"):
        url = decode_qr(path)
        decoded_urls.append(url)
        if not url:
            failed_decode += 1

    sample_df["decoded_url"] = decoded_urls
    decode_fail_rate = failed_decode / len(sample_df) * 100
    print(f"  Decoded OK : {len(sample_df) - failed_decode:,}")
    print(f"  Failed     : {failed_decode:,}  ({decode_fail_rate:.1f}%)")

    # URL correctness check  (decoded == original URL used to generate the QR)
    url_match = (sample_df["decoded_url"] == sample_df["true_url"]).mean() * 100
    print(f"  URL match vs. original : {url_match:.2f}%  "
          f"(should be ~100%)")

    # Drop rows where decode failed
    valid_df = sample_df[sample_df["decoded_url"] != ""].reset_index(drop=True)
    print(f"  Proceeding with {len(valid_df):,} successfully decoded images")

    # ------ Load MobileNet ------
    print("\n[3/6] Loading MobileNetV2 & SVM...")
    cnn_model = MobileNetV2FeatureExtractor()
    state     = torch.load(MODEL_MOBILENET, map_location=DEVICE)
    # strip 'classifier.*' keys — they're not in this reduced model definition
    filtered_state = {k: v for k, v in state.items()
                      if not k.startswith("classifier")}
    cnn_model.load_state_dict(filtered_state, strict=False)
    cnn_model.to(DEVICE)
    cnn_model.eval()
    print("  ✅ MobileNetV2 loaded")

    with open(MODEL_SVM, "rb") as f:
        svm = pickle.load(f)
    print(f"  ✅ SVM loaded  (kernel={svm.kernel}, degree={svm.degree})")

    # ------ Extract CNN features → C0 ------
    print("\n[4/6] Extracting CNN features → C0 scores...")
    features = extract_cnn_features(valid_df["img_path"].tolist(), cnn_model)

    proba    = svm.predict_proba(features)
    mal_idx  = list(svm.classes_).index(1)
    C0       = proba[:, mal_idx]
    print(f"  C0 extracted for {len(C0):,} samples  "
          f"(mean={C0.mean():.3f}, std={C0.std():.3f})")

    # ------ Load BERT ------
    print("\n[5/6] Loading BERT & extracting C1 scores...")
    tokenizer  = BertTokenizer.from_pretrained(MODEL_BERT)
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_BERT)
    bert_model = bert_model.to(DEVICE)
    bert_model.eval()
    print("  ✅ BERT loaded")

    C1 = get_bert_scores(valid_df["decoded_url"].tolist(), tokenizer, bert_model)
    print(f"  C1 extracted for {len(C1):,} samples  "
          f"(mean={C1.mean():.3f}, std={C1.std():.3f})")

    # ------ Load fusion model → final prediction ------
    print("\n[6/6] Fusing C0 + C1 → final predictions...")
    with open(MODEL_FUSION, "rb") as f:
        fusion = pickle.load(f)
    print(f"  ✅ Fusion model loaded  "
          f"(C0 w={fusion.coef_[0][0]:+.3f}, C1 w={fusion.coef_[0][1]:+.3f})")

    X_fuse      = np.column_stack([C0, C1])
    fused_proba  = fusion.predict_proba(X_fuse)[:, 1]
    fused_pred   = fusion.predict(X_fuse)
    y_true       = valid_df["true_label"].values

    # ------ Metrics ------
    print("\n" + "=" * 60)
    print("END-TO-END RESULTS")
    print("=" * 60)

    def print_metrics(y_true, y_pred, y_proba, name):
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_proba)
        cm   = confusion_matrix(y_true, y_pred)
        print(f"\n  {name}")
        print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1-score  : {f1:.4f}")
        print(f"    AUC-ROC   : {auc:.4f}")
        print(f"    TN={cm[0,0]:7,}  FP={cm[0,1]:7,}")
        print(f"    FN={cm[1,0]:7,}  TP={cm[1,1]:7,}")
        return acc, prec, rec, f1, auc

    c0_acc, c0_prec, c0_rec, c0_f1, c0_auc = print_metrics(
        y_true, (C0 >= 0.5).astype(int), C0, "Stage 0: MobileNetV2+SVM (C0)")

    c1_acc, c1_prec, c1_rec, c1_f1, c1_auc = print_metrics(
        y_true, (C1 >= 0.5).astype(int), C1, "Stage 1: BERT (C1)")

    fu_acc, fu_prec, fu_rec, fu_f1, fu_auc = print_metrics(
        y_true, fused_pred, fused_proba, "Stage 2: Fusion (C0+C1)")

    # ------ Summary table ------
    summary = pd.DataFrame([
        {"Stage": "MobileNetV2 + SVM  (QR image only)",
         "Accuracy": round(c0_acc, 4), "AUC": round(c0_auc, 4),
         "F1": round(c0_f1, 4)},
        {"Stage": "BERT               (decoded URL only)",
         "Accuracy": round(c1_acc, 4), "AUC": round(c1_auc, 4),
         "F1": round(c1_f1, 4)},
        {"Stage": "Fusion             (image + URL)",
         "Accuracy": round(fu_acc, 4), "AUC": round(fu_auc, 4),
         "F1": round(fu_f1, 4)},
    ])

    print("\n" + "=" * 60)
    print("SUMMARY TABLE  (URL decoded live from QR image)")
    print("=" * 60)
    print(summary.to_string(index=False))

    # ------ Save results ------
    results_df = valid_df[["img_path", "split", "true_label", "decoded_url", "true_url"]].copy()
    results_df["C0"]         = C0
    results_df["C1"]         = C1
    results_df["fused_prob"] = fused_proba
    results_df["fused_pred"] = fused_pred
    results_df["correct"]    = (fused_pred == y_true).astype(int)
    results_df["url_correct"]= (results_df["decoded_url"] == results_df["true_url"]).astype(int)

    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    results_df.to_csv(results_path, index=False)

    with open(summary_path, "w") as f:
        f.write("END-TO-END TEST SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Sample fraction : {SAMPLE_FRACTION}\n")
        f.write(f"Samples total   : {len(valid_df):,}\n")
        f.write(f"Decode failures : {failed_decode}  ({decode_fail_rate:.1f}%)\n")
        f.write(f"URL match rate  : {url_match:.2f}%\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    print(f"\n✅ Full results : {results_path}")
    print(f"✅ Summary      : {summary_path}")

    # ------ Failure analysis ------
    errors = results_df[results_df["correct"] == 0]
    print(f"\nError analysis ({len(errors):,} misclassified):")
    print(f"  FP (benign predicted malicious): {((fused_pred==1)&(y_true==0)).sum():,}")
    print(f"  FN (malicious predicted benign): {((fused_pred==0)&(y_true==1)).sum():,}")

    print("\n" + "=" * 60)
    print("✅  End-to-end test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
