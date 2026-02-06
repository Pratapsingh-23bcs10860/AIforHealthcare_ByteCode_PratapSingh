#!/usr/bin/env python3
"""Aggressive ensemble trainer targeting >92% BACC on CN/AD classification.
Combines heavy augmentation, multiple architectures, TTA, threshold optimization.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from app.dataset import MRIDataset
from app.model import ResNetSlice
import torchvision.models as models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ============ Enhanced Augmentation Pipeline ============
class HeavyAugmentation:
    """Heavy augmentation for 3D MRI slices"""
    def __init__(self):
        self.aug = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ])
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = transforms.ToTensor()(x).squeeze()
        x = self.aug(x)
        return x

# ============ Model Definitions ============
class EfficientNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        efn = models.efficientnet_b0(weights='DEFAULT')
        self.features = nn.Sequential(*list(efn.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, 2)
    
    def forward(self, x):
        # x: (B, Z, H, W) -> process each slice, aggregate
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W).repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(B, Z, -1)
        x = x.mean(dim=1)  # aggregate over slices
        x = self.classifier(x)
        return x

class DenseNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        dn = models.densenet121(weights='DEFAULT')
        self.features = dn.features
        self.classifier = nn.Linear(1024, 2)
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W).repeat(1, 3, 1, 1)
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(B, Z, -1)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# ============ Training Loop ============
def train_epoch(model, loader, criterion, optimizer, augment=None):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc='Train')):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        # heavy aug
        if augment:
            imgs = augment(imgs)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_sum += loss.item()
        _, pred = logits.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    acc = correct / total
    loss_avg = loss_sum / (batch_idx + 1)
    return loss_avg, acc

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc='Eval')):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        loss_sum += loss.item()
        _, pred = logits.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    acc = correct / total
    loss_avg = loss_sum / (batch_idx + 1)
    return loss_avg, acc

@torch.no_grad()
def get_predictions(model, loader):
    """Get soft predictions for threshold optimization"""
    model.eval()
    probs = []
    labels = []
    for imgs, lbl in tqdm(loader, desc='Predict'):
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        soft = torch.softmax(logits, dim=1)
        probs.append(soft[:, 1].cpu().numpy())  # prob of AD
        labels.append(lbl.numpy())
    return np.concatenate(probs), np.concatenate(labels)

def balanced_accuracy(pred, true):
    """Balanced accuracy = (TPR + TNR) / 2"""
    tp = ((pred == 1) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    return (tpr + tnr) / 2

def optimize_threshold(probs, labels):
    """Grid search for best threshold"""
    best_bacc = 0
    best_thresh = 0.5
    for t in np.linspace(0.2, 0.8, 31):
        pred = (probs >= t).astype(int)
        bacc = balanced_accuracy(pred, labels)
        if bacc > best_bacc:
            best_bacc = bacc
            best_thresh = t
    return best_thresh, best_bacc

# ============ Main Training ============
def train_model(model_class, model_name, epochs=50):
    print(f'\n{"="*60}')
    print(f'Training {model_name}')
    print(f'{"="*60}')
    
    # Data
    train_ds = MRIDataset('data/train.csv', 'data/processed')
    val_ds = MRIDataset('data/val.csv', 'data/processed')
    test_ds = MRIDataset('data/test.csv', 'data/processed')
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)
    
    # Model
    model = model_class().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    aug = HeavyAugmentation()
    
    best_val_acc = 0
    patience = 15
    patience_cnt = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, aug)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        
        print(f'[{epoch+1}/{epochs}] Train: loss={train_loss:.4f}, acc={train_acc:.4f} | '
              f'Val: loss={val_loss:.4f}, acc={val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/best_{model_name}.pth')
        else:
            patience_cnt += 1
        
        if patience_cnt >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best and evaluate
    model.load_state_dict(torch.load(f'models/best_{model_name}.pth'))
    
    # Get probs for threshold optimization
    val_probs, val_labels = get_predictions(model, val_loader)
    best_thresh, val_bacc = optimize_threshold(val_probs, val_labels)
    
    test_probs, test_labels = get_predictions(model, test_loader)
    test_pred = (test_probs >= best_thresh).astype(int)
    test_bacc = balanced_accuracy(test_pred, test_labels)
    
    print(f'\n{model_name} Results:')
    print(f'  Val BACC (threshold={best_thresh:.3f}): {val_bacc:.4f}')
    print(f'  Test BACC: {test_bacc:.4f}')
    
    return model, best_thresh, test_bacc, test_probs, test_labels

# ============ Ensemble & TTA ============
def ensemble_inference(models, thresholds, test_loader, device=DEVICE):
    """Ensemble of models with TTA"""
    print('\nEnsemble Inference (with TTA)...')
    all_preds = []
    all_labels = []
    
    for imgs, labels in tqdm(test_loader, desc='Ensemble'):
        imgs = imgs.to(device)
        ensemble_probs = []
        
        # TTA: original + 3 augmentations
        aug = HeavyAugmentation()
        img_variants = [imgs, aug(imgs), aug(imgs), aug(imgs)]
        
        for model, thresh in zip(models, thresholds):
            model.eval()
            model_probs = []
            for img_var in img_variants:
                with torch.no_grad():
                    logits = model(img_var)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    model_probs.append(probs)
            ensemble_probs.append(np.mean(model_probs, axis=0))
        
        # Average across models
        avg_prob = np.mean(ensemble_probs, axis=0)
        pred = (avg_prob >= 0.5).astype(int)
        all_preds.append(pred)
        all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ensemble_bacc = balanced_accuracy(all_preds, all_labels)
    
    return ensemble_bacc, all_preds, all_labels

# ============ Run Training ============
if __name__ == '__main__':
    # Train multiple models
    models_list = []
    thresholds = []
    results = {}
    
    for model_class, model_name in [
        (ResNetSlice, 'resnet50'),
        (EfficientNet3D, 'efficientnet'),
        (DenseNet3D, 'densenet'),
    ]:
        try:
            model, thresh, test_bacc, probs, labels = train_model(model_class, model_name, epochs=40)
            models_list.append(model)
            thresholds.append(thresh)
            results[model_name] = {'bacc': test_bacc, 'thresh': thresh}
        except Exception as e:
            print(f'Error training {model_name}: {e}')
            continue
    
    # Ensemble
    print(f'\n{"="*60}')
    print('Ensemble Results')
    print(f'{"="*60}')
    
    test_ds = MRIDataset('data/test.csv', 'data/processed')
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)
    
    ensemble_bacc, preds, labels = ensemble_inference(models_list, thresholds, test_loader)
    
    print(f'\nFinal Ensemble BACC: {ensemble_bacc:.4f} ({ensemble_bacc*100:.2f}%)')
    
    # Summary
    print(f'\n{"="*60}')
    print('Summary')
    print(f'{"="*60}')
    for name, res in results.items():
        print(f'{name:20s}: {res["bacc"]:.4f}')
    print(f'{"Ensemble":20s}: {ensemble_bacc:.4f}')
    
    if ensemble_bacc >= 0.92:
        print('\n✓ TARGET 92% BACC ACHIEVED!')
    else:
        print(f'\n○ Target 92% - Current {ensemble_bacc*100:.2f}%')
