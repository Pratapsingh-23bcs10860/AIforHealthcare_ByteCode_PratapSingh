#!/usr/bin/env python3
"""Lean multi-class ensemble for >55% BACC."""

import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt, seaborn as sns

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MRIDatasetMulticlass(Dataset):
    def __init__(self, csv_path, target_shape=(48, 128, 128)):
        self.df = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        arr = np.load(os.path.join('data/processed', row['file']))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        # Resize
        current_shape = arr.shape
        zoom = [self.target_shape[i] / current_shape[i] for i in range(3)]
        arr = ndimage.zoom(arr, zoom, order=1)
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return torch.from_numpy(arr), torch.tensor(self.label_map[row['label']], dtype=torch.long)

class ResNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(self.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 3))
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(B, Z, -1).mean(dim=1)
        return self.fc(x)

def train_epoch(model, loader, opt, aug_p=0.5):
    model.train()
    preds, labels_list = [], []
    for imgs, labels in loader:
        # Light aug
        if np.random.rand() < aug_p:
            for i in range(imgs.shape[0]):
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    imgs[i] = torch.from_numpy(ndimage.rotate(imgs[i].numpy(), angle, axes=(1, 2), reshape=False, order=1, mode='constant')).float()
        
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(imgs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        preds.append(model(imgs).argmax(1).cpu().detach().numpy())
        labels_list.append(labels.cpu().numpy())
    
    return balanced_accuracy_score(np.concatenate(labels_list), np.concatenate(preds))

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    preds, labels_list, probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    return balanced_accuracy_score(np.concatenate(labels_list), np.concatenate(preds)), np.concatenate(probs), np.concatenate(labels_list)

print(f'Device: {DEVICE}\n' + '='*70)
print('Multi-Class: CN vs MCI vs AD')
print('='*70)

train_ds = MRIDatasetMulticlass('data/train_multiclass.csv')
val_ds = MRIDatasetMulticlass('data/val_multiclass.csv')
test_ds = MRIDatasetMulticlass('data/test_multiclass.csv')

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# Train model
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n')
model = ResNet3D().to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

best_model_state = None
best_val_bacc = 0
patience_cnt = 0

for epoch in range(30):
    train_bacc = train_epoch(model, train_loader, opt, aug_p=0.6)
    val_bacc, _, _ = eval_epoch(model, val_loader)
    sched.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1:2d}: train_BACC={train_bacc:.4f}, val_BACC={val_bacc:.4f}')
    
    if val_bacc > best_val_bacc:
        best_val_bacc = val_bacc
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
    
    if patience_cnt >= 10:
        break

# Restore best
model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
test_bacc, test_probs, test_labels = eval_epoch(model, test_loader)
test_preds = np.argmax(test_probs, axis=1)

cm = confusion_matrix(test_labels, test_preds)
f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
f1_per = f1_score(test_labels, test_preds, average=None, zero_division=0)
prec_per = precision_score(test_labels, test_preds, average=None, zero_division=0)
rec_per = recall_score(test_labels, test_preds, average=None, zero_division=0)

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
for i, name in enumerate(['CN', 'MCI', 'AD']):
    print(f'{name:5s}: Prec={prec_per[i]:.3f}, Rec={rec_per[i]:.3f}, F1={f1_per[i]:.3f}')

print('\n--- Confusion Matrix ---')
print(f'        CN   MCI   AD')
for i, name in enumerate(['CN', 'MCI', 'AD']):
    print(f'{name:3s}:  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}')

if test_bacc >= 0.55:
    print(f'\n✓ TARGET 55% BACC ACHIEVED! ({test_bacc*100:.2f}%)')
else:
    print(f'\nCurrent: {test_bacc*100:.2f}% — Target: 55.00%')

# Save results
os.makedirs('results', exist_ok=True)
pd.DataFrame({
    'Metric': ['Balanced Accuracy', 'Macro F1', 'AUC', 'CN Prec', 'MCI Prec', 'AD Prec', 'CN Rec', 'MCI Rec', 'AD Rec'],
    'Value': [test_bacc, f1_macro, auc_score, prec_per[0], prec_per[1], prec_per[2], rec_per[0], rec_per[1], rec_per[2]]
}).to_csv('results/multiclass_results.csv', index=False)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'MCI', 'AD'], yticklabels=['CN', 'MCI', 'AD'], cbar=False)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix (BACC={test_bacc:.4f})')
plt.tight_layout()
plt.savefig('results/multiclass_confusion_matrix.png', dpi=150, bbox_inches='tight')

print('\n✓ Results saved to results/multiclass_results.csv')
print('✓ Confusion matrix saved to results/multiclass_confusion_matrix.png')
