#!/usr/bin/env python3
"""Simple ResNet50 training targeting >92% BACC on CN/AD classification."""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from app.dataset import MRIDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

class ResNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        # adapt conv1 to accept 1 channel
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
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        # x: (B, Z, H, W) -> process each slice, aggregate
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(B, Z, -1)
        x = x.mean(dim=1)  # aggregate over slices
        x = self.classifier(x)
        return x

def balanced_accuracy(pred, true):
    """Balanced accuracy = (TPR + TNR) / 2"""
    tp = ((pred == 1) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    return (tpr + tnr) / 2

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
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
        
        if (batch_idx + 1) % 3 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)}: loss={loss.item():.4f}')
    
    acc = correct / total
    loss_avg = loss_sum / (batch_idx + 1)
    return loss_avg, acc

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    preds = []
    labels_list = []
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        loss_sum += loss.item()
        _, pred = logits.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        preds.append(pred.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    
    acc = correct / total
    loss_avg = loss_sum / max(len(loader), 1)
    preds_all = np.concatenate(preds)
    labels_all = np.concatenate(labels_list)
    bacc = balanced_accuracy(preds_all, labels_all)
    
    return loss_avg, acc, bacc, preds_all, labels_all

if __name__ == '__main__':
    print(f'\n{"="*60}')
    print('Training ResNet50 for CN/AD Classification')
    print(f'{"="*60}')
    
    # Data
    print('\nLoading datasets...')
    train_ds = MRIDataset('data/train.csv', 'data/processed')
    val_ds = MRIDataset('data/val.csv', 'data/processed')
    test_ds = MRIDataset('data/test.csv', 'data/processed')
    
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    
    # Model
    model = ResNet3D().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    best_val_bacc = 0
    patience = 10
    patience_cnt = 0
    
    print('\nTraining...')
    for epoch in range(30):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_bacc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        
        print(f'[{epoch+1:2d}/30] Train: loss={train_loss:.4f}, acc={train_acc:.4f} | '
              f'Val: loss={val_loss:.4f}, acc={val_acc:.4f}, BACC={val_bacc:.4f}')
        
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            patience_cnt = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_resnet50.pth')
            print(f'          ✓ Saved best model (BACC={val_bacc:.4f})')
        else:
            patience_cnt += 1
        
        if patience_cnt >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Evaluate on test
    model.load_state_dict(torch.load('models/best_resnet50.pth'))
    test_loss, test_acc, test_bacc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
    
    # Metrics
    tp = ((test_preds == 1) & (test_labels == 1)).sum()
    tn = ((test_preds == 0) & (test_labels == 0)).sum()
    fp = ((test_preds == 1) & (test_labels == 0)).sum()
    fn = ((test_preds == 0) & (test_labels == 1)).sum()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    sensitivity = tpr
    specificity = tnr
    
    print(f'\n{"="*60}')
    print('Test Results')
    print(f'{"="*60}')
    print(f'Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)')
    print(f'BACC:          {test_bacc:.4f} ({test_bacc*100:.2f}%)')
    print(f'Sensitivity:   {sensitivity:.4f} ({sensitivity*100:.2f}%)')
    print(f'Specificity:   {specificity:.4f} ({specificity*100:.2f}%)')
    print(f'TP={tp}, TN={tn}, FP={fp}, FN={fn}')
    
    if test_bacc >= 0.92:
        print('\n✓ TARGET 92% BACC ACHIEVED!')
    else:
        print(f'\nCurrent: {test_bacc*100:.2f}% — Target: 92.00%')
