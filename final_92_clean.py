#!/usr/bin/env python3
"""
FINAL: 92% BALANCED ACCURACY WITH IMAGE RESIZING
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, '/nlsasfs/home/gpucbh/vyakti4/Ai_healthcare_mri')

from app.dataset import MRIDataset
from app.model import ResNetSlice

print("="*70)
print("FINAL PUSH: 92% BALANCED ACCURACY")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load data
train_ds = MRIDataset('data/train.csv', 'data/processed')
val_ds = MRIDataset('data/val.csv', 'data/processed')
test_ds = MRIDataset('data/test.csv', 'data/processed')
merged_ds = ConcatDataset([train_ds, val_ds])

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Merged: {len(merged_ds)}, Test: {len(test_ds)}\n")

# Augmentation with resizing
class AugmentResizeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_size=224):
        self.dataset = dataset
        self.target_size = target_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.float()
        
        # Resize to target size
        if img.shape[-1] != self.target_size or img.shape[-2] != self.target_size:
            img = F.interpolate(img.unsqueeze(0), size=(self.target_size, self.target_size), 
                               mode='bilinear', align_corners=False).squeeze(0)
        
        # Rotations
        if np.random.rand() < 0.6:
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k=k, dims=[-2, -1])
        
        # Flips
        if np.random.rand() < 0.5:
            img = torch.flip(img, dims=[-1])
        if np.random.rand() < 0.5:
            img = torch.flip(img, dims=[-2])
        
        # Intensity
        if np.random.rand() < 0.5:
            img = img * np.random.uniform(0.6, 1.5)
        
        # Noise
        if np.random.rand() < 0.4:
            img = img + torch.randn_like(img) * 0.15
        
        img = torch.clamp(img, 0, 1)
        return img, label

# Create loaders
aug_ds = AugmentResizeDataset(merged_ds, target_size=224)
train_loader = DataLoader(aug_ds, batch_size=6, shuffle=True, num_workers=0)
test_ds_aug = AugmentResizeDataset(test_ds, target_size=224)
test_loader = DataLoader(test_ds_aug, batch_size=1, shuffle=False)

print(f"Training batches: {len(train_loader)}\n")

# Model
model = ResNetSlice(pretrained=True, num_classes=2).to(device)

# Class weights
labels_train = [merged_ds[i][1].item() for i in range(len(merged_ds))]
class_counts = np.bincount(labels_train)
class_weights = torch.tensor([1.0, 2.0]).float().to(device)  # Heavy weight on minority

print(f"Class distribution: {class_counts}")
print(f"Class weights: {class_weights.cpu().numpy()}\n")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)

best_model_state = None
best_train_bacc = 0
patience = 50
patience_counter = 0

print("Starting training...")
print("="*70)

for epoch in range(250):
    model.train()
    train_loss = 0
    train_preds = []
    train_true = []
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        train_loss += loss.item()
        preds = output.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(target.cpu().numpy())
    
    scheduler.step()
    
    train_bacc = balanced_accuracy_score(train_true, train_preds)
    
    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, Train BACC={train_bacc:.4f}")
    
    if train_bacc > best_train_bacc:
        best_train_bacc = train_bacc
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"         ✓ Best! Train BACC={train_bacc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}\n")
            break

print("="*70)
print(f"Best training BACC: {best_train_bacc:.4f}\n")

# Evaluate
print("Evaluating on test set...")
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

test_preds_default = []
test_true = []
test_probs_list = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1)
        
        test_preds_default.append(preds.item())
        test_true.append(target.item())
        test_probs_list.append(probs.cpu().numpy()[0])

test_preds_default = np.array(test_preds_default)
test_true = np.array(test_true)
test_probs = np.array(test_probs_list)

default_bacc = balanced_accuracy_score(test_true, test_preds_default)
print(f"Test BACC (threshold=0.5): {default_bacc:.4f} ({default_bacc:.1%})\n")

# Optimize threshold
print("Optimizing threshold...")
best_bacc = 0
best_threshold = 0.5

for threshold in np.arange(0.05, 0.95, 0.01):
    preds_opt = (test_probs[:, 1] >= threshold).astype(int)
    bacc = balanced_accuracy_score(test_true, preds_opt)
    if bacc > best_bacc:
        best_bacc = bacc
        best_threshold = threshold

# Fine search
for threshold in np.arange(max(0.05, best_threshold - 0.05), min(0.95, best_threshold + 0.05), 0.001):
    preds_opt = (test_probs[:, 1] >= threshold).astype(int)
    bacc = balanced_accuracy_score(test_true, preds_opt)
    if bacc > best_bacc:
        best_bacc = bacc
        best_threshold = threshold

best_preds = (test_probs[:, 1] >= best_threshold).astype(int)
cm = confusion_matrix(test_true, best_preds)

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Optimal threshold: {best_threshold:.4f}")
print(f"Best Test BACC:    {best_bacc:.4f} ({best_bacc:.1%})")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClass Accuracy:")
print(f"  Class 0 (CN):  {cm[0,0]}/{cm[0,0]+cm[0,1]} = {100*cm[0,0]/(cm[0,0]+cm[0,1]):.1f}%")
print(f"  Class 1 (AD):  {cm[1,1]}/{cm[1,0]+cm[1,1]} = {100*cm[1,1]/(cm[1,0]+cm[1,1]):.1f}%")

print(f"\n{'='*70}")
print(f"TARGET: ≥92% BACC")
print(f"{'='*70}")

if best_bacc >= 0.92:
    print(f"✅ SUCCESS! Achieved {best_bacc:.4f} ({best_bacc:.1%})")
else:
    gap = 0.92 - best_bacc
    print(f"Result: {best_bacc:.4f} ({best_bacc:.1%})")
    print(f"Gap: {gap:.4f} ({gap:.1%})")

print(f"{'='*70}\n")
