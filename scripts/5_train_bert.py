"""
Script 5: BERT Fine-tuning for URL Classification
==================================================
Following: Su & Su (2023) - Sensors 2023, 23, 8499
"BERT-Based Approaches to Identifying Malicious URLs"

Specs:
- Model:      bert-base-cased
- Max length: 128 tokens
- Batch size: 16
- LR:         1e-5  (paper value)
- Epochs:     10    (paper tests 10/20/30, we use 10)
- Optimizer:  AdamW
- Warmup:     500 steps
- Weight decay: 0.01
- Grad clipping: 1.0
- Loss:       CrossEntropyLoss
- Strategy:   CLS token + linear head, end-to-end
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tqdm import tqdm

# ==============================
# CONFIG - Su & Su (2023)
# ==============================
BERT_MODEL   = 'bert-base-cased'
MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 10      # paper value (tests 10/20/30)
LR           = 1e-5    # paper value
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

CSV_DIR        = "data/processed"
MODEL_SAVE_DIR = "models/stage1"
OUTPUT_DIR     = "outputs/stage1/features"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==============================
# DATASET
# Raw URL string → BERT tokenizer
# Following Track 1 of Su & Su 2023
# ==============================
class URLDataset(Dataset):

    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df         = pd.read_csv(csv_path)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        print(f"  Loaded {len(self.df):,} URLs from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        url   = str(row['url'])
        label = int(row['label'])

        # Feed raw URL string directly
        # bert-base-cased: case sensitive (important for URLs)
        # [CLS] url tokens [SEP] [PAD...]
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
# TRAIN ONE EPOCH
# ==============================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc="  Training")

    for batch in pbar:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels         = batch['label'].to(device)

        # Forward pass
        # BertForSequenceClassification internally:
        # 1. BERT encoder → hidden states
        # 2. Take [CLS] token (index 0)
        # 3. Dropout + Linear → logits
        # This is exactly what Su & Su 2023 does
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels         = labels
        )

        loss   = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - standard for BERT stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        pbar.set_postfix({
            'loss': f"{total_loss/(pbar.n+1):.4f}",
            'acc':  f"{correct/total:.4f}"
        })

    return total_loss / len(loader), correct / total

# ==============================
# EVALUATE
# ==============================
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                labels         = labels
            )

            total_loss += outputs.loss.item()
            preds       = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)

    return total_loss / len(loader), acc, prec, rec, f1

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 1: BERT Fine-tuning")
    print("Following: Su & Su (2023)")
    print("=" * 60)
    print(f"  Model      : {BERT_MODEL}")
    print(f"  Max length : {MAX_LENGTH}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  LR         : {LR}")
    print(f"  Strategy   : CLS token + linear head, end-to-end")
    print(f"  Device     : {DEVICE}")
    print("=" * 60)

    # Tokenizer
    print(f"\nLoading tokenizer: {BERT_MODEL}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = URLDataset(
        f"{CSV_DIR}/train.csv", tokenizer, MAX_LENGTH
    )
    val_dataset = URLDataset(
        f"{CSV_DIR}/val.csv", tokenizer, MAX_LENGTH
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4
    )

    # Model
    # BertForSequenceClassification:
    # BERT encoder + dropout + linear(768 → 2)
    # Trained end-to-end, CLS token used for classification
    print(f"\nLoading BERT: {BERT_MODEL}")
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels = 2
    )
    model = model.to(DEVICE)

    # AdamW optimizer (standard for BERT fine-tuning)
    optimizer = AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = WEIGHT_DECAY
    )

    # Linear warmup then decay scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = WARMUP_STEPS,
        num_training_steps = total_steps
    )

    print(f"\n  Total steps  : {total_steps:,}")
    print(f"  Warmup steps : {WARMUP_STEPS}")
    print(f"  Steps/epoch  : {len(train_loader):,}")

    # Training loop
    best_val_acc = 0
    history      = []

    for epoch in range(EPOCHS):

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, DEVICE
        )

        print(f"\n  Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}")
        print(f"  Val Loss   : {val_loss:.4f} | Val Acc   : {val_acc:.4f}")
        print(f"  Precision  : {val_prec:.4f} | Recall    : {val_rec:.4f} | F1: {val_f1:.4f}")

        history.append({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'val_loss':   val_loss,
            'val_acc':    val_acc,
            'precision':  val_prec,
            'recall':     val_rec,
            'f1':         val_f1
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(f"{MODEL_SAVE_DIR}/best")
            tokenizer.save_pretrained(f"{MODEL_SAVE_DIR}/best")
            print(f"  Best model saved (Val Acc: {val_acc:.4f})")

        # Also save per-epoch checkpoint
        # Useful if training crashes
        model.save_pretrained(f"{MODEL_SAVE_DIR}/epoch_{epoch+1}")
        tokenizer.save_pretrained(f"{MODEL_SAVE_DIR}/epoch_{epoch+1}")

    # Save training history
    pd.DataFrame(history).to_csv(
        f"{MODEL_SAVE_DIR}/training_history.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("BERT TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val Acc : {best_val_acc:.4f}")
    print(f"  Model saved  : {MODEL_SAVE_DIR}/best")
    print(f"  History      : {MODEL_SAVE_DIR}/training_history.csv")
    print("\nNext: Extract C1 scores (script 6)")
