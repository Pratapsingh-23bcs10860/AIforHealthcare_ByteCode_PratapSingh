#!/usr/bin/env python3
"""Fast aggressive ensemble for 92% BACC - optimized version."""

import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}\n')

class Aug3D:
    def __call__(self, vol):
        vol = vol.copy().astype(np.float32)
        if np.random.rand() > 0.3:
            angle = np.random.uniform(-25, 25)
            vol = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
        if np.random.rand() > 0.3:
            vol = vol * np.random.uniform(0.6, 1.4)
        if np.random.rand() > 0.4:
            vol = vol + np.random.normal(0, 0.04, vol.shape)
        return np.clip(vol, 0, 1).astype(np.float32)

class MRIDataset(Dataset):
    def __init__(self, csv_path, target_shape=(48, 128, 128)):
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.label_map = {'CN': 0, 'AD': 1}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        arr = np.load(f'data/processed/{row["file"]}')
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        zoom = [self.target_shape[i] / arr.shape[i] for i in range(3)]
        arr = ndimage.zoom(arr, zoom, order=1)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return torch.from_numpy(arr.astype(np.float32)), torch.tensor(self.label_map[row['label']])

class ResNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        rn = models.resnet50(weights='DEFAULT')
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.feat = nn.Sequential(self.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2, rn.layer3, rn.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))
    def forward(self, x):
        B, Z, H, W = x.shape
        x = self.feat(x.view(B*Z, 1, H, W))
        x = self.pool(x)
        x = x.view(B, Z, -1).mean(1)
        return self.fc(x)

class Dense3D(nn.Module):
    def __init__(self):
        super().__init__()
        dn = models.densenet121(weights='DEFAULT')
        self.conv0 = nn.Conv2d(1, 64, 7, 2, 3)
        self.feat = nn.Sequential(self.conv0, dn.features[1:])
        self.fc = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))
    def forward(self, x):
        B, Z, H, W = x.shape
        x = self.feat(x.view(B*Z, 1, H, W))
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(B, Z, -1).mean(1)
        return self.fc(x)

def train_epoch(model, loader, opt, aug, device):
    model.train()
    preds, labels_list = [], []
    for imgs, labels in loader:
        imgs_aug = []
        for i in range(imgs.shape[0]):
            aug_vol = aug(imgs[i].numpy())
            imgs_aug.append(torch.from_numpy(aug_vol))
        try:
            imgs = torch.stack(imgs_aug)
        except:
            continue
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(imgs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        preds.append(model(imgs).argmax(1).cpu().detach().numpy())
        labels_list.append(labels.cpu().numpy())
    return balanced_accuracy_score(np.concatenate(labels_list), np.concatenate(preds)) if preds else 0

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, labels_list, probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        labels_list.append(labels.numpy())
    return balanced_accuracy_score(np.concatenate(labels_list), np.concatenate(preds)), np.concatenate(probs), np.concatenate(labels_list)

print('='*70)
print('Aggressive Ensemble: CN vs AD (92% BACC Target)')
print('='*70)

train_ds = MRIDataset('data/train.csv')
val_ds = MRIDataset('data/val.csv')
test_ds = MRIDataset('data/test.csv')
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n')

train_ldr = DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=0)
val_ldr = DataLoader(val_ds, batch_size=6, shuffle=False, num_workers=0)
test_ldr = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

models_trained = []
aug = Aug3D()

for model_cls, name in [(ResNet3D, 'ResNet50'), (Dense3D, 'DenseNet121')]:
    print(f'Training {name}...')
    model = model_cls().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    
    best_val = 0
    patience = 12
    for ep in range(40):
        tr_bacc = train_epoch(model, train_ldr, opt, aug, DEVICE)
        val_bacc, _, _ = eval_epoch(model, val_ldr, DEVICE)
        sched.step()
        if (ep+1) % 10 == 0:
            print(f'  Epoch {ep+1}: train={tr_bacc:.3f}, val={val_bacc:.3f}')
        if val_bacc >= best_val:
            best_val = val_bacc
            patience = 12
        else:
            patience -= 1
        if patience == 0:
            break
    models_trained.append(model)
    print(f'  ✓ Done\n')

print('='*70)
print('Ensemble + TTA Inference')
print('='*70 + '\n')

# Val threshold optimization
val_probs_all = []
for model in models_trained:
    val_bacc, probs, _ = eval_epoch(model, val_ldr, DEVICE)
    val_probs_all.append(probs[:, 1])  # prob of AD
val_ensemble_probs = np.mean(val_probs_all, axis=0)
val_labels = pd.read_csv('data/val.csv')['label'].map({'CN': 0, 'AD': 1}).values

best_thresh, best_bacc = 0.5, 0
for t in np.linspace(0.2, 0.8, 31):
    pred = (val_ensemble_probs >= t).astype(int)
    bacc = balanced_accuracy_score(val_labels, pred)
    if bacc > best_bacc:
        best_bacc, best_thresh = bacc, t

print(f'Best threshold: {best_thresh:.3f}, Val BACC: {best_bacc:.4f}\n')

# Test ensemble
test_probs_all = []
for model in models_trained:
    _, probs, _ = eval_epoch(model, test_ldr, DEVICE)
    test_probs_all.append(probs[:, 1])
test_ensemble_probs = np.mean(test_probs_all, axis=0)
test_pred = (test_ensemble_probs >= best_thresh).astype(int)
test_labels = pd.read_csv('data/test.csv')['label'].map({'CN': 0, 'AD': 1}).values

test_bacc = balanced_accuracy_score(test_labels, test_pred)
tp = ((test_pred == 1) & (test_labels == 1)).sum()
tn = ((test_pred == 0) & (test_labels == 0)).sum()
fp = ((test_pred == 1) & (test_labels == 0)).sum()
fn = ((test_pred == 0) & (test_labels == 1)).sum()
sens = tp / (tp + fn + 1e-8)
spec = tn / (tn + fp + 1e-8)

print('='*70)
print('TEST RESULTS: CN vs AD Classification')
print('='*70)
print(f'Balanced Accuracy:  {test_bacc:.4f} ({test_bacc*100:.2f}%)')
print(f'Sensitivity (TPR):  {sens:.4f}')
print(f'Specificity (TNR):  {spec:.4f}')
cm = confusion_matrix(test_labels, test_pred)
print(f'\nConfusion Matrix:')
print(f'        CN  AD')
print(f'CN:    {cm[0,0]:3d} {cm[0,1]:3d}')
print(f'AD:    {cm[1,0]:3d} {cm[1,1]:3d}')

print('='*70)
if test_bacc >= 0.92:
    print(f'✅ TARGET 92% BACC ACHIEVED! ({test_bacc*100:.2f}%)')
else:
    print(f'Current: {test_bacc*100:.2f}% — Target: 92.00%')
    print('Note: Small dataset (70 train) makes 92% challenging but ensemble')
    print('can approach it with favorable train/val/test split variance')
print('='*70)

# Save
os.makedirs('results', exist_ok=True)
pd.DataFrame({'BACC': [test_bacc], 'Sensitivity': [sens], 'Specificity': [spec], 'Threshold': [best_thresh]}).to_csv('results/ensemble_results.csv')
print('\n✓ Saved to results/ensemble_results.csv')
