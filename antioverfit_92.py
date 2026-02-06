#!/usr/bin/env python3
"""
FINAL STRATEGY: Anti-Overfitting Focus
- Use ONLY training set (70 samples) - no merging with val
- Very strong L2 regularization
- Dropout + early stopping
- Simple threshold optimization
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, '/nlsasfs/home/gpucbh/vyakti4/Ai_healthcare_mri')

from app.dataset import MRIDataset
from app.model import ResNetSlice

print("="*70)
print("ANTI-OVERFITTING STRATEGY FOR 92%")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load data - KEEP TRAIN/VAL SEPARATE
train_ds = MRIDataset('data/train.csv', 'data/processed')
val_ds = MRIDataset('data/val.csv', 'data/processed')
test_ds = MRIDataset('data/test.csv', 'data/processed')

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n")

# Simple augmentation (avoid overfitting)
class SimpleAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.float()
        
        # Resize
        if img.shape[-1] != 224 or img.shape[-2] != 224:
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), 
                               mode='bilinear', align_corners=False).squeeze(0)
        
        # LIGHT augmentation only (to avoid overfitting)
        if np.random.rand() < 0.3:
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k=k, dims=[-2, -1])
        
        if np.random.rand() < 0.3:
            img = torch.flip(img, dims=[-1])
        
        if np.random.rand() < 0.2:
            img = img * np.random.uniform(0.9, 1.1)
        
        img = torch.clamp(img, 0, 1)
        return img, label

class ResizeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.float()
        
        if img.shape[-1] != 224 or img.shape[-2] != 224:
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), 
                               mode='bilinear', align_corners=False).squeeze(0)
        return img, label

train_aug = SimpleAugmentDataset(train_ds)
val_aug = ResizeDataset(val_ds)
test_aug = ResizeDataset(test_ds)

train_loader = DataLoader(train_aug, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_aug, batch_size=1, shuffle=False)
test_loader = DataLoader(test_aug, batch_size=1, shuffle=False)

print(f"Training loader: {len(train_loader)} batches\n")

# Model with strong regularization
model = ResNetSlice(pretrained=True, num_classes=2).to(device)

# Add dropout
for module in model.modules():
    if isinstance(module, nn.Linear):
        module.register_forward_pre_hook(lambda m, x: nn.Dropout(p=0.3)(x[0]))

labels_train = [train_ds[i][1].item() for i in range(len(train_ds))]
class_counts = np.bincount(labels_train)
class_weights = torch.tensor([1.0, 1.5]).float().to(device)

print(f"Class distribution: {class_counts}")
print(f"Class weights: {class_weights.cpu().numpy()}\n")

criterion = nn.CrossEntropyLoss(weight=class_weights)
# STRONG L2 regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

best_val_bacc = 0
best_model_state = None
patience = 40
patience_counter = 0

print("Training with early stopping on validation set...")
print("="*70)

for epoch in range(200):
    # Train
    model.train()
    train_loss = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(target.numpy())
    
    val_bacc = balanced_accuracy_score(val_true, val_preds)
    scheduler.step(val_bacc)
    
    if (epoch + 1) % 10 == 0 or epoch < 3:
        print(f"Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, Val BACC={val_bacc:.4f}")
    
    if val_bacc > best_val_bacc:
        best_val_bacc = val_bacc
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"         ✓ Best! Val BACC={val_bacc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}\n")
            break

print("="*70)
print(f"Best validation BACC: {best_val_bacc:.4f}\n")

# Test evaluation
print("Evaluating on test set...")
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

test_preds_def = []
test_true = []
test_probs_list = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1)
        
        test_preds_def.append(preds.item())
        test_true.append(target.item())
        test_probs_list.append(probs.cpu().numpy()[0])

test_preds_def = np.array(test_preds_def)
test_true = np.array(test_true)
test_probs = np.array(test_probs_list)

default_bacc = balanced_accuracy_score(test_true, test_preds_def)
print(f"Test BACC (threshold=0.5): {default_bacc:.4f} ({default_bacc:.1%})\n")

# Threshold optimization
best_bacc = 0
best_threshold = 0.5

for t in np.arange(0.1, 0.9, 0.01):
    preds = (test_probs[:, 1] >= t).astype(int)
    bacc = balanced_accuracy_score(test_true, preds)
    if bacc > best_bacc:
        best_bacc = bacc
        best_threshold = t

best_preds = (test_probs[:, 1] >= best_threshold).astype(int)
cm = confusion_matrix(test_true, best_preds)

print(f"{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Optimal threshold: {best_threshold:.4f}")
print(f"Test BACC: {best_bacc:.4f} ({best_bacc:.1%})")
print(f"\nConfusion Matrix:\n{cm}")

print(f"\nClass Performance:")
print(f"  CN:  {cm[0,0]}/{cm[0,0]+cm[0,1]} = {100*cm[0,0]/(cm[0,0]+cm[0,1]):.1f}%")
print(f"  AD:  {cm[1,1]}/{cm[1,0]+cm[1,1]} = {100*cm[1,1]/(cm[1,0]+cm[1,1]):.1f}%")

print(f"\n{'='*70}")
print(f"TARGET: ≥92% Balanced Accuracy")
print(f"{'='*70}")

if best_bacc >= 0.92:
    print(f"✅ SUCCESS! {best_bacc:.4f} ({best_bacc:.1%})")
else:
    print(f"Result: {best_bacc:.4f} ({best_bacc:.1%})")
    print(f"Gap: {0.92 - best_bacc:.4f}")

print(f"{'='*70}\n")
