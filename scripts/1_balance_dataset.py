import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

PROJECT_ROOT = os.path.expanduser('~/QuishGuard')

print("="*60)
print("DATASET BALANCING")
print("="*60)

df = pd.read_csv(f'{PROJECT_ROOT}/data/raw/malicious_phishing.csv')

print("Original:", len(df))
print(df['type'].value_counts())

df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)

benign_df = df[df['label'] == 0]
mal_df = df[df['label'] == 1]

# balance (change if RAM issue)
SAMPLES_PER_CLASS = 200000

benign_sampled = benign_df.sample(n=SAMPLES_PER_CLASS, random_state=42)
mal_sampled = mal_df.sample(n=SAMPLES_PER_CLASS, random_state=42)

balanced_df = pd.concat([benign_sampled, mal_sampled]).sample(frac=1, random_state=42)

print("Balanced:", len(balanced_df))

# 60 / 15 / 25
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.40,
    stratify=balanced_df['label'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.625,   # 0.25 / 0.40
    stratify=temp_df['label'],
    random_state=42
)

processed_dir = f'{PROJECT_ROOT}/data/processed'
os.makedirs(processed_dir, exist_ok=True)

train_df[['url','label']].to_csv(f'{processed_dir}/train.csv', index=False)
val_df[['url','label']].to_csv(f'{processed_dir}/val.csv', index=False)
test_df[['url','label']].to_csv(f'{processed_dir}/test.csv', index=False)

print("✓ Saved splits")
