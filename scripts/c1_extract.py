import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_WORKERS = 4

MODEL_PATH = "models/stage1/best"
C0_ALIGNED_PATH = "outputs/stage0/{split}_with_C0_fusion_v2.csv"
OUT_PATH = "outputs/stage1/features/{split}_with_C1_fusion_v2.csv"

os.makedirs("outputs/stage1/features", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

class URLAlignedDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = str(row["url"])
        enc = self.tokenizer(
            url,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sample_id": row["sample_id"],
            "url": url,
            "label": int(row["label"])
        }

        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        return item

print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("Loaded:", MODEL_PATH)

@torch.no_grad()
def extract_split(split):
    print(f"\n--- {split.upper()} ---")
    base = pd.read_csv(C0_ALIGNED_PATH.format(split=split))
    req_cols = {"sample_id", "url", "label", "C0"}
    if not req_cols.issubset(set(base.columns)):
        raise ValueError(f"{split}: missing required cols in C0 aligned file")

    ds = URLAlignedDataset(base[["sample_id", "url", "label"]], tokenizer, MAX_LENGTH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    sample_ids, urls, labels, c1_scores = [], [], [], []

    for batch in tqdm(dl, desc=f"C1 {split}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)

        logits = model(**kwargs).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        sample_ids.extend(batch["sample_id"])
        urls.extend(batch["url"])
        labels.extend(batch["label"].cpu().numpy().tolist())
        c1_scores.extend(probs.tolist())

    out = pd.DataFrame({
        "sample_id": sample_ids,
        "url": urls,
        "label": labels,
        "C1": c1_scores
    }).sort_values("sample_id").reset_index(drop=True)

    path = OUT_PATH.format(split=split)
    out.to_csv(path, index=False)

    print(f"Saved: {path}")
    print(f"Rows: {len(out):,} | C1 mean={out['C1'].mean():.6f} std={out['C1'].std():.6f}")
    print(f"C1 min={out['C1'].min():.6f} max={out['C1'].max():.6f}")

for sp in ["train", "val", "test"]:
    extract_split(sp)

print("\nDone: C1 fusion v2 files created.")
