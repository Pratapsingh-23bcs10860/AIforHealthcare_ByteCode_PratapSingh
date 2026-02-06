#!/usr/bin/env python3
"""Ultra-aggressive ensemble to reach 92% BACC on CN vs AD binary classification.
Combines: Heavy augmentation, multi-architecture ensemble, TTA, threshold optimization.
"""

import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ============ Ultra-Heavy Augmentation ============
class UltraAugmentation3D:
    """Extreme augmentation for small datasets"""
    def __call__(self, vol):
        vol = vol.copy().astype(np.float32)
        orig_shape = vol.shape
        
        # Heavy rotation
        if np.random.rand() > 0.2:
            angle = np.random.uniform(-30, 30)
            vol = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
        
        # Aggressive elastic deformation
        if np.random.rand() > 0.2:
            shape = vol.shape
            dx = np.random.randn(*shape) * 3
            dy = np.random.randn(*shape) * 3
            dz = np.random.randn(*shape) * 3
            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            x = (x + 25 * dx).astype(np.float32)
            y = (y + 25 * dy).astype(np.float32)
            z = (z + 25 * dz).astype(np.float32)
            x = np.clip(x, 0, shape[1]-1)
            y = np.clip(y, 0, shape[0]-1)
            z = np.clip(z, 0, shape[2]-1)
            try:
                vol = ndimage.map_coordinates(vol, [y, x, z], order=1, mode='constant', cval=0)
            except:
                pass
        
        # Aggressive intensity variation
        if np.random.rand() > 0.2:
            vol = vol * np.random.uniform(0.5, 1.5)
        
        # Heavy noise
        if np.random.rand() > 0.3:
            vol = vol + np.random.normal(0, 0.05, vol.shape)
        
        # Cutout regions
        if np.random.rand() > 0.5:
            vol[np.random.randint(0, orig_shape[0]):np.random.randint(10, orig_shape[0]),
                np.random.randint(0, orig_shape[1]):np.random.randint(50, orig_shape[1]),
                np.random.randint(0, orig_shape[2]):np.random.randint(50, orig_shape[2])] *= 0.5
        
        vol = vol[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
        if vol.shape != orig_shape:
            pad_width = [(0, max(0, orig_shape[i] - vol.shape[i])) for i in range(3)]
            vol = np.pad(vol, pad_width, mode='constant')
        
        return np.clip(vol, 0, 1).astype(np.float32)

# ============ Dataset ============
class MRIDataset(Dataset):
    def __init__(self, csv_path, processed_dir='data/processed', target_shape=(48, 128, 128)):
        self.df = pd.read_csv(csv_path)
        self.processed_dir = processed_dir
        self.target_shape = target_shape
        self.label_map = {'CN': 0, 'AD': 1}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        arr = np.load(os.path.join(self.processed_dir, row['file']))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        # Resize
        current_shape = arr.shape
        zoom = [self.target_shape[i] / current_shape[i] for i in range(3)]
        arr = ndimage.zoom(arr, zoom, order=1)
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return torch.from_numpy(arr), torch.tensor(self.label_map[row['label']], dtype=torch.long)

# ============ Models ============
class ResNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(self.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.6), nn.Linear(512, 2))
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(B, Z, -1).mean(dim=1)
        return self.fc(x)

class DenseNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        dn = models.densenet121(weights='DEFAULT')
        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.features = nn.Sequential(self.conv0, dn.features[1:])
        self.fc = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.6), nn.Linear(256, 2))
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W)
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(B, Z, -1).mean(dim=1)
        return self.fc(x)

class EfficientNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        efn = models.efficientnet_b0(weights='DEFAULT')
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(self.conv0, *list(efn.children())[1:-1])
        self.fc = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.6), nn.Linear(256, 2))
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W).repeat(1, 3, 1, 1)
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(B, Z, -1).mean(dim=1)
        return self.fc(x)

# ============ Training ============
def train_epoch(model, loader, opt, aug, device):
    model.train()
    loss_sum = 0
    preds, labels_list = [], []
    for imgs, labels in loader:
        # Intense augmentation
        imgs_aug = []
        for i in range(imgs.shape[0]):
            vol_aug = aug(imgs[i].numpy())
            imgs_aug.append(torch.from_numpy(vol_aug))
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
        loss_sum += loss.item()
        preds.append(model(imgs).argmax(1).cpu().detach().numpy())
        labels_list.append(labels.cpu().numpy())
    
    if preds:
        bacc = balanced_accuracy_score(np.concatenate(labels_list), np.concatenate(preds))
    else:
        bacc = 0
    return loss_sum / max(len(loader), 1), bacc

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, labels_list, probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    
    preds_all = np.concatenate(preds)
    labels_all = np.concatenate(labels_list)
    probs_all = np.concatenate(probs)
    bacc = balanced_accuracy_score(labels_all, preds_all)
    
    return bacc, preds_all, labels_all, probs_all

@torch.no_grad()
def predict_tta(model, imgs, num_tta=8, device=DEVICE):
    """Test-time augmentation with many variants"""
    aug = UltraAugmentation3D()
    probs_list = []
    
    for _ in range(num_tta):
        imgs_var = imgs.numpy()
        if _ > 0:
            imgs_var = aug(imgs_var)
        imgs_var = torch.from_numpy(imgs_var).unsqueeze(0).to(device)
        logits = model(imgs_var)
        probs = torch.softmax(logits, dim=1)
        probs_list.append(probs.cpu().numpy())
    
    return np.mean(probs_list, axis=0)[0]

# ============ Main ============
print('\n' + '='*70)
print('Ultra-Aggressive Ensemble: CN vs AD Classification (92% Target)')
print('='*70)

train_ds = MRIDataset('data/train.csv')
val_ds = MRIDataset('data/val.csv')
test_ds = MRIDataset('data/test.csv')

print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

models_trained = []
aug = UltraAugmentation3D()

# Train multiple models
for model_class, name in [(ResNet3D, 'ResNet50'), (DenseNet3D, 'DenseNet121'), (EfficientNet3D, 'EfficientNet')]:
    print(f'\nTraining {name}...')
    model = model_class().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    
    best_val_bacc = 0
    patience_cnt = 0
    patience = 20
    
    for epoch in range(60):
        train_loss, train_bacc = train_epoch(model, train_loader, opt, aug, DEVICE)
        val_bacc, _, _, _ = eval_epoch(model, val_loader, DEVICE)
        sched.step()
        
        if (epoch+1) % 10 == 0:
            print(f'  Epoch {epoch+1:2d}: train_BACC={train_bacc:.4f}, val_BACC={val_bacc:.4f}')
        
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            patience_cnt = 0
        else:
            patience_cnt += 1
        
        if patience_cnt >= patience:
            break
    
    models_trained.append(model)
    print(f'  ✓ Best val BACC: {best_val_bacc:.4f}')

# ============ Ensemble + TTA ============
print('\n' + '='*70)
print('Ensemble Inference with Heavy TTA')
print('='*70)

# Get validation predictions for threshold optimization
val_ensemble_probs = []
for model in models_trained:
    model.eval()
    probs = []
    for imgs, _ in val_loader:
        p = predict_tta(model, imgs[0], num_tta=6, device=DEVICE)
        probs.append(p)
    val_ensemble_probs.append(np.array(probs))

val_ensemble_avg = np.mean(val_ensemble_probs, axis=0)
val_labels = pd.read_csv('data/val.csv')['label'].map({'CN': 0, 'AD': 1}).values

# Optimize threshold
best_thresh = 0.5
best_bacc = 0
for t in np.linspace(0.1, 0.9, 41):
    pred = (val_ensemble_avg >= t).astype(int)
    bacc = balanced_accuracy_score(val_labels, pred)
    if bacc > best_bacc:
        best_bacc = bacc
        best_thresh = t

print(f'Best threshold: {best_thresh:.3f}, Val BACC: {best_bacc:.4f}')

# Test ensemble
test_ensemble_probs = []
for model_idx, model in enumerate(models_trained):
    model.eval()
    probs = []
    for imgs, _ in test_loader:
        p = predict_tta(model, imgs[0], num_tta=8, device=DEVICE)
        probs.append(p)
    test_ensemble_probs.append(np.array(probs))

test_ensemble_avg = np.mean(test_ensemble_probs, axis=0)
test_pred = (test_ensemble_avg >= best_thresh).astype(int)
test_labels = pd.read_csv('data/test.csv')['label'].map({'CN': 0, 'AD': 1}).values

# Metrics
test_bacc = balanced_accuracy_score(test_labels, test_pred)
tp = ((test_pred == 1) & (test_labels == 1)).sum()
tn = ((test_pred == 0) & (test_labels == 0)).sum()
fp = ((test_pred == 1) & (test_labels == 0)).sum()
fn = ((test_pred == 0) & (test_labels == 1)).sum()
sens = tp / (tp + fn + 1e-8)
spec = tn / (tn + fp + 1e-8)
prec = tp / (tp + fp + 1e-8)

print('\n' + '='*70)
print('TEST RESULTS: CN vs AD Classification')
print('='*70)
print(f'Balanced Accuracy:  {test_bacc:.4f} ({test_bacc*100:.2f}%)')
print(f'Sensitivity (TPR):  {sens:.4f} ({sens*100:.2f}%)')
print(f'Specificity (TNR):  {spec:.4f} ({spec*100:.2f}%)')
print(f'Precision:          {prec:.4f} ({prec*100:.2f}%)')
print(f'\nConfusion Matrix:')
print(f'  TP={tp}, TN={tn}, FP={fp}, FN={fn}')

cm = confusion_matrix(test_labels, test_pred)
print(f'\n        CN  AD')
print(f'CN:    {cm[0,0]:3d} {cm[0,1]:3d}')
print(f'AD:    {cm[1,0]:3d} {cm[1,1]:3d}')

print('\n' + '='*70)
if test_bacc >= 0.92:
    print(f'✅ TARGET 92% BACC ACHIEVED! ({test_bacc*100:.2f}%)')
else:
    print(f'Current: {test_bacc*100:.2f}% — Target: 92.00%')
print('='*70)

# Save results
os.makedirs('results', exist_ok=True)
results = {
    'BACC': test_bacc,
    'Sensitivity': sens,
    'Specificity': spec,
    'Precision': prec,
    'Threshold': best_thresh,
    'Models': 3,
    'TTA_Iterations': 8
}
pd.Series(results).to_csv('results/ensemble_92_results.csv')
print('\n✓ Results saved to results/ensemble_92_results.csv')
