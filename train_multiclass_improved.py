#!/usr/bin/env python3
"""Improved multi-class training: CN / MCI / AD
Features:
- Heavy 3D augmentation
- Transfer learning with ResNet50 adapter
- Class-weighted loss
- Training logs, final metrics (BACC, AUC(one-vs-rest), Macro F1, precision/recall)
- Saves plots and CSV results to results/
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

class Aug3D:
    def __call__(self, vol):
        vol = vol.astype(np.float32).copy()
        if vol.ndim == 4 and vol.shape[0] == 1:
            vol = vol.squeeze(0)
        if vol.ndim == 2:
            vol = np.expand_dims(vol, 0)
        # small random rotation
        if np.random.rand() > 0.4:
            ang = np.random.uniform(-20, 20)
            try:
                vol = ndimage.rotate(vol, ang, axes=(1,2), reshape=False, order=1)
            except:
                pass
        # intensity
        vol = vol * np.random.uniform(0.7, 1.3)
        # noise
        if np.random.rand() > 0.6:
            vol += np.random.normal(0, 0.03, vol.shape)
        # flip
        if np.random.rand() > 0.5:
            vol = np.flip(vol, axis=np.random.choice([0,1,2]))
        return np.clip(vol, 0, 1).astype(np.float32)

class MRIDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_shape=(48,128,128)):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {'CN':0, 'MCI':1, 'AD':2}
        self.target_shape = target_shape
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(os.path.join('data','processed', row['file']))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        z,h,w = arr.shape
        tz,th,tw = self.target_shape
        zoom = (tz/z, th/h, tw/w)
        arr = ndimage.zoom(arr, zoom, order=1)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        if self.transform:
            arr = self.transform(arr)
        # return tensor shape (B, 1, Z, H, W) usage: we'll pass as (B, Z, H, W) and adapt model
        return torch.from_numpy(arr).unsqueeze(0).float(), torch.tensor(self.label_map[row['label']])

class ResNet3DAdapter(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        rn = resnet50(weights='DEFAULT')
        rn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone = rn
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        # x: (B,1,Z,H,W)
        B = x.shape[0]
        x = x.squeeze(1)  # (B,Z,H,W)
        x = x.reshape(B * x.shape[1], 1, x.shape[2], x.shape[3])
        feat = self.backbone(x)  # (B*Z, 2048)
        feat = feat.view(B, -1, feat.shape[-1]).mean(1)
        return self.classifier(feat)

# load datasets
train_csv = 'data/train.csv'
val_csv = 'data/val.csv'
test_csv = 'data/test.csv'

train_ds = MRIDataset(train_csv, transform=Aug3D())
val_ds = MRIDataset(val_csv, transform=None)
test_ds = MRIDataset(test_csv, transform=None)

train_loader = DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# class weights from training labels
train_labels = pd.read_csv(train_csv)['label'].map({'CN':0,'MCI':1,'AD':2}).values
class_counts = np.bincount(train_labels, minlength=3)
class_weights = torch.tensor(class_counts.sum() / (class_counts + 1e-6), dtype=torch.float32).to(DEVICE)
print('Class counts:', class_counts)
print('Class weights:', class_weights.cpu().numpy())

model = ResNet3DAdapter(num_classes=3).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training loop
best_val = 0
best_state = None
patience = 12
patience_cnt = 0
train_history = {'loss':[], 'val_loss':[], 'val_bacc':[]}

for epoch in range(1, 61):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_history['loss'].append(epoch_loss)

    # validation
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    val_loss = val_loss / len(val_loader.dataset)
    train_history['val_loss'].append(val_loss)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    val_bacc = balanced_accuracy_score(all_labels, all_preds)
    train_history['val_bacc'].append(val_bacc)
    scheduler.step(val_bacc)

    print(f'Epoch {epoch:02d}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, val_BACC={val_bacc:.4f}')

    if val_bacc > best_val:
        best_val = val_bacc
        best_state = model.state_dict().copy()
        patience_cnt = 0
    else:
        patience_cnt += 1
    if patience_cnt >= patience:
        print('Early stopping')
        break

# load best
if best_state is not None:
    model.load_state_dict(best_state)

# Test evaluation
model.eval()
probs_list = []
preds_list = []
labels_list = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = logits.argmax(1).cpu().numpy()[0]
        probs_list.append(probs)
        preds_list.append(pred)
        labels_list.append(labels.numpy()[0])
probs_arr = np.vstack(probs_list)
preds = np.array(preds_list)
labels = np.array(labels_list)

# Metrics
bacc = balanced_accuracy_score(labels, preds)
try:
    auc = roc_auc_score(pd.get_dummies(labels), probs_arr, multi_class='ovr')
except Exception:
    auc = np.nan
macro_f1 = f1_score(labels, preds, average='macro')
precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
cm = confusion_matrix(labels, preds)

os.makedirs('results', exist_ok=True)
pd.DataFrame({'BalancedAccuracy':[bacc], 'AUC':[auc], 'MacroF1':[macro_f1], 'Precision_macro':[precision_macro], 'Recall_macro':[recall_macro]}).to_csv('results/multiclass_improved_results.csv', index=False)
print('\nTest Results saved to results/multiclass_improved_results.csv')
print(f'BACC={bacc:.4f}, AUC={auc}, MacroF1={macro_f1:.4f}')

# Save confusion matrix plot
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Pred')
plt.ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig('results/multiclass_improved_confusion.png')

# Save curves
plt.figure()
plt.plot(train_history['loss'], label='train_loss')
plt.plot(train_history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/multiclass_loss_curves.png')

plt.figure()
plt.plot(train_history['val_bacc'], label='val_bacc')
plt.xlabel('Epoch')
plt.ylabel('Val BACC')
plt.legend()
plt.savefig('results/multiclass_val_bacc.png')

print('Plots saved to results/')
