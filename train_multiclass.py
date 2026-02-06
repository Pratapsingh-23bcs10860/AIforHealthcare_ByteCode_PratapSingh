#!/usr/bin/env python3
"""Multi-class CNN for CN vs MCI vs AD classification.
Target: >55% Balanced Accuracy
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from scipy import ndimage
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ============ Dataset ============
class MRIDatasetMulticlass(Dataset):
    def __init__(self, csv_path, processed_dir='data/processed', target_shape=(48, 128, 128)):
        self.df = pd.read_csv(csv_path)
        self.processed_dir = processed_dir
        self.target_shape = target_shape
        self.label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    
    def __len__(self):
        return len(self.df)
    
    def _resize_volume(self, vol):
        current_shape = vol.shape
        zoom = [self.target_shape[i] / current_shape[i] for i in range(3)]
        return ndimage.zoom(vol, zoom, order=1)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        file_path = os.path.join(self.processed_dir, row['file'])
        arr = np.load(file_path)
        
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        
        arr = self._resize_volume(arr)
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        
        tensor = torch.from_numpy(arr)
        label = self.label_map[row['label']]
        
        return tensor, torch.tensor(label, dtype=torch.long)

# ============ Model ============
class ResNet3DMulticlass(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(
            self.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(B, Z, -1)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# ============ Augmentation ============
class Augmentation3D:
    def __call__(self, vol):
        vol = vol.copy()
        orig_shape = vol.shape
        
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            vol = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1, mode='constant')
        
        # Intensity variation
        if np.random.rand() > 0.4:
            vol = vol * np.random.uniform(0.8, 1.2)
        
        # Gaussian noise
        if np.random.rand() > 0.5:
            vol = vol + np.random.normal(0, 0.03, vol.shape)
        
        return np.clip(vol, 0, 1).astype(np.float32)

# ============ Training ============
def train_epoch(model, loader, criterion, optimizer, aug=None):
    model.train()
    loss_sum = 0.0
    preds = []
    labels_list = []
    
    for imgs, labels in loader:
        if aug:
            imgs = torch.stack([
                torch.from_numpy(aug(imgs[i].numpy()))
                for i in range(imgs.shape[0])
            ])
        
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_sum += loss.item()
        preds.append(logits.argmax(1).cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    
    preds_all = np.concatenate(preds)
    labels_all = np.concatenate(labels_list)
    bacc = balanced_accuracy_score(labels_all, preds_all)
    
    return loss_sum / len(loader), bacc

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    preds = []
    labels_list = []
    probs = []
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        loss_sum += loss.item()
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    
    preds_all = np.concatenate(preds)
    labels_all = np.concatenate(labels_list)
    probs_all = np.concatenate(probs)
    bacc = balanced_accuracy_score(labels_all, preds_all)
    
    return loss_sum / len(loader), bacc, preds_all, labels_all, probs_all

# ============ Main ============
if __name__ == '__main__':
    print('\n' + '='*70)
    print('Multi-Class CNN: CN vs MCI vs AD Classification')
    print('='*70)
    
    # Load datasets
    train_ds = MRIDatasetMulticlass('data/train_multiclass.csv', 'data/processed')
    val_ds = MRIDatasetMulticlass('data/val_multiclass.csv', 'data/processed')
    test_ds = MRIDatasetMulticlass('data/test_multiclass.csv', 'data/processed')
    
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    
    # Model
    model = ResNet3DMulticlass(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    aug = Augmentation3D()
    
    best_val_bacc = 0
    patience = 12
    patience_cnt = 0
    
    print('\nTraining...')
    for epoch in range(40):
        train_loss, train_bacc = train_epoch(model, train_loader, criterion, optimizer, aug)
        val_loss, val_bacc, _, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        
        print(f'[{epoch+1:2d}/40] Train: loss={train_loss:.4f}, BACC={train_bacc:.4f} | '
              f'Val: loss={val_loss:.4f}, BACC={val_bacc:.4f}')
        
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            patience_cnt = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_multiclass_resnet50.pth')
            print(f'           ✓ Saved (best BACC: {val_bacc:.4f})')
        else:
            patience_cnt += 1
        
        if patience_cnt >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load('models/best_multiclass_resnet50.pth'))
    test_loss, test_bacc, test_preds, test_labels, test_probs = eval_epoch(model, test_loader, criterion)
    
    # Compute metrics
    cm = confusion_matrix(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average='macro')
    f1_per_class = f1_score(test_labels, test_preds, average=None)
    prec_per_class = precision_score(test_labels, test_preds, average=None, zero_division=0)
    rec_per_class = recall_score(test_labels, test_preds, average=None, zero_division=0)
    
    # AUC (One-vs-Rest for multi-class)
    try:
        auc_score = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    except:
        auc_score = 0.0
    
    print('\n' + '='*70)
    print('TEST RESULTS')
    print('='*70)
    print(f'Balanced Accuracy:  {test_bacc:.4f} ({test_bacc*100:.2f}%)')
    print(f'Macro F1-Score:     {f1_macro:.4f}')
    print(f'AUC (macro OvR):    {auc_score:.4f}')
    
    print('\n--- Per-Class Metrics ---')
    class_names = ['CN', 'MCI', 'AD']
    for i, name in enumerate(class_names):
        print(f'{name:5s}: Precision={prec_per_class[i]:.4f}, Recall={rec_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}')
    
    print('\n--- Confusion Matrix ---')
    print(cm)
    print(f'\nRows: True Label | Columns: Predicted Label')
    print(f'        CN    MCI    AD')
    for i, name in enumerate(class_names):
        print(f'{name:3s}:  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}')
    
    print('\n' + '='*70)
    if test_bacc >= 0.55:
        print('✓ TARGET 55% BACC ACHIEVED!')
    else:
        print(f'Current: {test_bacc*100:.2f}% — Target: 55.00%')
    print('='*70)
    
    # Save results
    results_df = pd.DataFrame({
        'Metric': ['Balanced Accuracy', 'Macro F1-Score', 'AUC (macro OvR)',
                   'CN Precision', 'MCI Precision', 'AD Precision',
                   'CN Recall', 'MCI Recall', 'AD Recall'],
        'Value': [test_bacc, f1_macro, auc_score,
                  prec_per_class[0], prec_per_class[1], prec_per_class[2],
                  rec_per_class[0], rec_per_class[1], rec_per_class[2]]
    })
    results_df.to_csv('results/multiclass_results.csv', index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Multi-Class Confusion Matrix (BACC={test_bacc:.4f})')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/multiclass_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('\n✓ Saved confusion matrix to results/multiclass_confusion_matrix.png')
    print('✓ Saved results to results/multiclass_results.csv')
