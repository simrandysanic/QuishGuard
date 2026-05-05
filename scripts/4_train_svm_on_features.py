"""
Train SVM Quadratic on MobileNetV2 features
Following: Alaca & Çelik (2023)

Paper best result: SVM Quadratic, 94.67% accuracy
"""

import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

os.makedirs("models/stage0", exist_ok=True)

print("="*60)
print("TRAINING SVM ON MOBILENETV2 FEATURES")
print("="*60)

# Load features
print("\nLoading features...")
train_feat = np.load("outputs/stage0/train_features_1000d.npy")
train_lab  = np.load("outputs/stage0/train_labels.npy")
val_feat   = np.load("outputs/stage0/val_features_1000d.npy")
val_lab    = np.load("outputs/stage0/val_labels.npy")
test_feat  = np.load("outputs/stage0/test_features_1000d.npy")
test_lab   = np.load("outputs/stage0/test_labels.npy")

print(f"  Train: {train_feat.shape} samples")
print(f"  Val:   {val_feat.shape} samples")
print(f"  Test:  {test_feat.shape} samples")

# Train SVM Quadratic (paper spec)
print("\n" + "="*60)
print("Training SVM (Quadratic kernel)...")
print("This may take 5-10 minutes on large dataset")
print("="*60)

svm = SVC(
    kernel='poly',
    degree=2,                  # quadratic (paper spec)
    gamma='scale',
    C=1.0,
    class_weight='balanced',   # handle any imbalance
    probability=True,          # needed for C0 scores
    verbose=True,
    max_iter=10000
)

svm.fit(train_feat, train_lab)
print("\n✅ SVM trained successfully")

# Evaluate on all splits
def evaluate(feat, lab, split_name):
    preds = svm.predict(feat)
    proba = svm.predict_proba(feat)
    
    acc  = accuracy_score(lab, preds)
    prec = precision_score(lab, preds, zero_division=0)
    rec  = recall_score(lab, preds, zero_division=0)
    f1   = f1_score(lab, preds, zero_division=0)
    cm   = confusion_matrix(lab, preds)
    
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN = {cm[0,0]:6,}  |  FP = {cm[0,1]:6,}")
    print(f"    FN = {cm[1,0]:6,}  |  TP = {cm[1,1]:6,}")
    
    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(
        lab, preds,
        target_names=['Benign', 'Malicious'],
        digits=4
    ))
    
    return acc, prec, rec, f1

# Evaluate
train_acc, train_prec, train_rec, train_f1 = evaluate(train_feat, train_lab, "train")
val_acc, val_prec, val_rec, val_f1 = evaluate(val_feat, val_lab, "val")
test_acc, test_prec, test_rec, test_f1 = evaluate(test_feat, test_lab, "test")

# Save SVM model
model_path = "models/stage0/svm_quadratic.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(svm, f)

print(f"\n{'='*60}")
print("SVM TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Model saved: {model_path}")
print(f"\n  Paper baseline: ~94.67%")
print(f"  Your result:    {test_acc*100:.2f}%")

if test_acc >= 0.93:
    print(f"\n  ✅ EXCELLENT! Close to paper performance!")
elif test_acc >= 0.90:
    print(f"\n  ✅ GOOD! Slightly below paper but acceptable!")
else:
    print(f"\n  ⚠️  Lower than expected. Check feature extraction.")

print("\nNext: Extract C0 scores (script 5)")
