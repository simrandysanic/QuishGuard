"""
Script 6: Extract C1 Scores from trained BERT
Output: [url, label, C1] CSVs for Stage 2 fusion
C1 = softmax probability of malicious class
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
MAX_LENGTH = 128
BATCH_SIZE = 32

MODEL_PATH = "models/stage1/best"
CSV_DIR    = "data/processed"
OUTPUT_DIR = "outputs/stage1/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==============================
# DATASET
# ==============================
class URLDataset(Dataset):

    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df         = pd.read_csv(csv_path)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        print(f"  Loaded {len(self.df):,} URLs")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        url   = str(row['url'])
        label = int(row['label'])

        encoding = self.tokenizer(
            url,
            add_special_tokens = True,
            max_length         = self.max_length,
            padding            = 'max_length',
            truncation         = True,
            return_tensors     = 'pt'
        )

        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label':          torch.tensor(label, dtype=torch.long),
            'url':            url
        }

# ==============================
# LOAD MODEL
# ==============================
print("\nLoading BERT...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model     = model.to(DEVICE)
model.eval()
print("✅ BERT loaded\n")

# ==============================
# EXTRACTION
# ==============================
def extract_C1(split):

    print(f"{'='*50}")
    print(f"Extracting C1: {split.upper()}")
    print(f"{'='*50}")

    dataset = URLDataset(
        csv_path  = f"{CSV_DIR}/{split}.csv",
        tokenizer = tokenizer,
        max_length = MAX_LENGTH
    )

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4
    )

    all_urls   = []
    all_labels = []
    all_C1     = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split}"):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
            )

            # softmax → probability of malicious (class 1)
            probs = torch.softmax(outputs.logits, dim=1)
            C1    = probs[:, 1].cpu().numpy()

            all_urls.extend(batch['url'])
            all_labels.extend(batch['label'].numpy())
            all_C1.extend(C1)

    df_out = pd.DataFrame({
        'url':   all_urls,
        'label': all_labels,
        'C1':    all_C1
    })

    out_path = f"{OUTPUT_DIR}/{split}_with_C1.csv"
    df_out.to_csv(out_path, index=False)

    print(f"  Saved   : {out_path}")
    print(f"  Samples : {len(df_out):,}")
    print(f"  C1 mean : {df_out['C1'].mean():.4f}")
    print(f"  C1 std  : {df_out['C1'].std():.4f}")
    print(f"  C1 min  : {df_out['C1'].min():.4f}")
    print(f"  C1 max  : {df_out['C1'].max():.4f}\n")

    return df_out


if __name__ == "__main__":

    for split in ['train', 'val', 'test']:
        extract_C1(split)

    print("=" * 50)
    print("C1 EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"\nFiles in {OUTPUT_DIR}:")
    print("  train_with_C1.csv")
    print("  val_with_C1.csv")
    print("  test_with_C1.csv")
    print("\nNext: Train fusion model (script 7)")
