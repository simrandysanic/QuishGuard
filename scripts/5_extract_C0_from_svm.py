"""
Extract C0 scores from SVM predictions
C0 = SVM probability of malicious class (class 1)

This creates the C0 scores needed for fusion with BERT's C1
"""

import numpy as np
import pandas as pd
import pickle
import os

os.makedirs("outputs/stage0", exist_ok=True)

print("="*60)
print("EXTRACTING C0 SCORES FROM SVM")
print("="*60)

# Load SVM
print("\nLoading SVM model...")
with open("models/stage0/svm_quadratic.pkl", 'rb') as f:
    svm = pickle.load(f)
print("✅ SVM loaded")

# Extract C0 for each split
for split in ['train', 'val', 'test']:
    
    print(f"\n{'-'*60}")
    print(f"Processing {split.upper()}")
    print(f"{'-'*60}")
    
    # Load features
    features = np.load(f"outputs/stage0/{split}_features_1000d.npy")
    labels   = np.load(f"outputs/stage0/{split}_labels.npy")
    
    print(f"  Loaded {len(features):,} samples")
    
    # Get probabilities from SVM
    # predict_proba returns [P(class_0), P(class_1)]
    # C0 = P(malicious) = P(class_1)
    proba = svm.predict_proba(features)
    C0 = proba[:, 1]
    
    # Create DataFrame
    df = pd.DataFrame({
        'label': labels,
        'C0': C0
    })
    
    # Save
    output_path = f"outputs/stage0/{split}_with_C0.csv"
    df.to_csv(output_path, index=False)
    
    # Stats
    print(f"  Saved: {output_path}")
    print(f"  C0 Statistics:")
    print(f"    Mean : {C0.mean():.4f}")
    print(f"    Std  : {C0.std():.4f}")
    print(f"    Min  : {C0.min():.4f}")
    print(f"    Max  : {C0.max():.4f}")
    
    # Sanity check
    benign_C0 = C0[labels == 0]
    malicious_C0 = C0[labels == 1]
    
    print(f"  C0 by class:")
    print(f"    Benign (0):    mean={benign_C0.mean():.4f}")
    print(f"    Malicious (1): mean={malicious_C0.mean():.4f}")

print("\n" + "="*60)
print("C0 EXTRACTION COMPLETE")
print("="*60)
print("\nFiles created:")
print("  outputs/stage0/train_with_C0.csv")
print("  outputs/stage0/val_with_C0.csv")
print("  outputs/stage0/test_with_C0.csv")
print("\nEach file contains:")
print("  - label: ground truth (0=benign, 1=malicious)")
print("  - C0:    SVM probability of malicious")
print("\nNext: Fusion with BERT C1 (script 7)")
