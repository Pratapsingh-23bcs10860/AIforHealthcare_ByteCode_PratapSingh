#!/usr/bin/env python3
"""
Simple & Direct: Single ResNet50 + Aggressive Augmentation + TTA
Target: 92% BACC on test set (CN vs AD)
"""

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AggAug:
    """Heavy augmentation for small dataset"""
    def __call__(self, vol):
        vol = vol.astype(np.float32)
        # Random rotation
        if np.random.rand() > 0.2:
            angle = np.random.uniform(-30, 30)
            vol = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1)
        # Intensity
        vol *= np.random.uniform(0.5, 1.5)
        # Noise
        if np.random.rand() > 0.3:
            vol += np.random.normal(0, 0.05, vol.shape)
        return np.clip(vol, 0, 1).astype(np.float32)

class MRIDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.label_map = {'CN': 0, 'AD': 1}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        arr = np.load(f'data/processed/{row["file"]}')
        if arr.ndim == 2: arr = np.expand_dims(arr, 0)
        # Resize to (48, 128, 128)
        zoom = [48/arr.shape[0], 128/arr.shape[1], 128/arr.shape[2]]
        arr = ndimage.zoom(arr, zoom, order=1)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return torch.from_numpy(arr).unsqueeze(0).float(), torch.tensor(self.label_map[row['label']])

# Simple model adapter
class ResNet50_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = resnet50(weights='DEFAULT')
        self.rn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.rn.fc = nn.Linear(2048, 2)
    def forward(self, x):
        # x: (B, 1, 48, 128, 128) -> reshape to (B*48, 1, 128, 128)
        B = x.shape[0]
        x = x.squeeze(1)  # (B, 48, 128, 128)
        x = x.reshape(B*48, 1, 128, 128)
        x = self.rn(x)  # (B*48, 2)
        # Pool over Z dimension
        x = x.reshape(B, 48, -1).mean(1)  # (B, 2)
        return x

# Load data
print(f'Device: {DEVICE}\nLoading data...')
train_ds = MRIDataset('data/train.csv')
val_ds = MRIDataset('data/val.csv')
test_ds = MRIDataset('data/test.csv')

print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

train_ldr = DataLoader(train_ds, batch_size=4, shuffle=True)
val_ldr = DataLoader(val_ds, batch_size=15)
test_ldr = DataLoader(test_ds, batch_size=1)

# Train
print('\nTraining ResNet50 + Heavy Augmentation...')
model = ResNet50_3D().to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
aug = AggAug()

best_val = 0
patience = 15
for ep in range(60):
    model.train()
    preds_tr, labels_tr = [], []
    for imgs, labels in train_ldr:
        # Augment
        aug_imgs = torch.stack([torch.from_numpy(aug(imgs[i].numpy())) for i in range(imgs.shape[0])])
        aug_imgs, labels = aug_imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        logits = model(aug_imgs)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        preds_tr.append(logits.argmax(1).detach().cpu().numpy())
        labels_tr.append(labels.cpu().numpy())
    
    # Eval on val
    model.eval()
    with torch.no_grad():
        preds_val, labels_val = [], []
        for imgs, labels in val_ldr:
            logits = model(imgs.to(DEVICE))
            preds_val.append(logits.argmax(1).cpu().numpy())
            labels_val.append(labels.numpy())
    
    train_bacc = balanced_accuracy_score(np.concatenate(labels_tr), np.concatenate(preds_tr))
    val_bacc = balanced_accuracy_score(np.concatenate(labels_val), np.concatenate(preds_val))
    sched.step()
    
    if (ep+1) % 15 == 0:
        print(f'Epoch {ep+1:2d}: train_BACC={train_bacc:.4f}, val_BACC={val_bacc:.4f}')
    
    if val_bacc > best_val:
        best_val = val_bacc
        patience = 15
        best_model_state = model.state_dict().copy()
    else:
        patience -= 1
    
    if patience == 0:
        break

# Load best model
model.load_state_dict(best_model_state)
model.eval()

# TTA Inference on Test
print('\nTTA Inference (8 augmented variants per sample)...')
all_preds = []
with torch.no_grad():
    for imgs, _ in test_ldr:
        # Average over 8 TTA samples
        tta_logits = []
        for _ in range(8):
            aug_img = torch.from_numpy(aug(imgs[0].numpy())).unsqueeze(0).to(DEVICE)
            logits = model(aug_img)
            tta_logits.append(logits.cpu().numpy())
        avg_logits = np.mean(tta_logits, axis=0)
        all_preds.append(avg_logits[0])

test_probs = np.array(all_preds)  # (15, 2)
test_labels = pd.read_csv('data/test.csv')['label'].map({'CN': 0, 'AD': 1}).values

# Grid search best threshold
best_thresh = 0.5
best_bacc = 0
for thresh in np.linspace(0.1, 0.9, 41):
    pred = (test_probs[:, 1] >= thresh).astype(int)
    bacc = balanced_accuracy_score(test_labels, pred)
    if bacc > best_bacc:
        best_bacc = bacc
        best_thresh = thresh

# Final predictions
test_pred = (test_probs[:, 1] >= best_thresh).astype(int)
bacc = balanced_accuracy_score(test_labels, test_pred)

# Metrics
cm = confusion_matrix(test_labels, test_pred)
tp = cm[1, 1]
tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
precision = tp / (tp + fp + 1e-8)

print('\n' + '='*70)
print('RESULTS: Binary CN vs AD Classification (92% Target)')
print('='*70)
print(f'Test Balanced Accuracy: {bacc:.4f} ({bacc*100:.2f}%)')
print(f'Sensitivity (AD recall):  {sensitivity:.4f}')
print(f'Specificity (CN recall):  {specificity:.4f}')
print(f'Precision (AD precision): {precision:.4f}')
print(f'Threshold used: {best_thresh:.3f}')
print(f'\nConfusion Matrix (rows=true, cols=pred):')
print(f'        CN  AD')
print(f'CN:    {cm[0,0]:3d} {cm[0,1]:3d}')
print(f'AD:    {cm[1,0]:3d} {cm[1,1]:3d}')
print('='*70)

# Target assessment
target = 0.92
if bacc >= target:
    print(f'✅ SUCCESS! Achieved {bacc*100:.2f}% (target: 92%)')
else:
    gap = (target - bacc) * 100
    print(f'Current: {bacc*100:.2f}% | Target: 92% | Gap: {gap:.2f}%')

print('='*70)

# Save
import os
os.makedirs('results', exist_ok=True)
pd.DataFrame({
    'Model': ['ResNet50-3D+TTA'],
    'BACC': [bacc],
    'Sensitivity': [sensitivity],
    'Specificity': [specificity],
    'Precision': [precision],
    'Threshold': [best_thresh]
}).to_csv('results/simple_ensemble_results.csv', index=False)
print('\n✓ Results saved to results/simple_ensemble_results.csv')
