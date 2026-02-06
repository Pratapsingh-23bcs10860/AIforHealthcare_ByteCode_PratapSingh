#!/usr/bin/env python3
"""Multi-class ensemble with TTA to reach >55% BACC."""

import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt, seaborn as sns

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

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

class Aug3D:
    def __call__(self, vol):
        vol = vol.copy()
        if np.random.rand() > 0.4:
            angle = np.random.uniform(-20, 20)
            vol = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1, mode='constant')
        if np.random.rand() > 0.5:
            vol = vol * np.random.uniform(0.75, 1.25)
        if np.random.rand() > 0.5:
            vol = vol + np.random.normal(0, 0.03, vol.shape)
        return np.clip(vol, 0, 1).astype(np.float32)

class ResNet3D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(self.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes))
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(B, Z, -1).mean(dim=1)
        x = self.classifier(x)
        return x

class DenseNet3D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        dn = models.densenet121(weights='DEFAULT')
        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.features = nn.Sequential(self.conv0, dn.features[1:])
        self.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes))
    
    def forward(self, x):
        B, Z, H, W = x.shape
        x = x.view(B*Z, 1, H, W)
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(B, Z, -1).mean(dim=1)
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, aug=None):
    model.train()
    loss_sum = 0.0
    preds = []
    labels_list = []
    for imgs, labels in loader:
        if aug:
            imgs_aug = []
            for i in range(imgs.shape[0]):
                vol = imgs[i].numpy()
                vol_aug = aug(vol)
                imgs_aug.append(torch.from_numpy(vol_aug))
            try:
                imgs = torch.stack(imgs_aug)
            except:
                continue
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        preds.append(logits.argmax(1).cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())
    if preds:
        bacc = balanced_accuracy_score(np.concatenate(labels_list), np.concatenate(preds))
    else:
        bacc = 0
    return loss_sum / max(len(loader), 1), bacc

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

@torch.no_grad()
def predict_tta(model, imgs, num_tta=4):
    aug = Aug3D()
    probs_list = []
    for _ in range(num_tta):
        imgs_var = imgs.numpy()
        if _ > 0:
            imgs_var = aug(imgs_var)
        imgs_var = torch.from_numpy(imgs_var).unsqueeze(0).to(DEVICE)
        logits = model(imgs_var)
        probs = torch.softmax(logits, dim=1)
        probs_list.append(probs.cpu().numpy())
    return np.mean(probs_list, axis=0)[0]

if __name__ == '__main__':
    print('\n' + '='*70)
    print('Multi-Class Ensemble + TTA: CN vs MCI vs AD')
    print('='*70)
    
    train_ds = MRIDatasetMulticlass('data/train_multiclass.csv')
    val_ds = MRIDatasetMulticlass('data/val_multiclass.csv')
    test_ds = MRIDatasetMulticlass('data/test_multiclass.csv')
    
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    models_trained = []
    criterion = nn.CrossEntropyLoss()
    
    for model_class, name in [(ResNet3D, 'ResNet50'), (DenseNet3D, 'DenseNet121')]:
        print(f'\nTraining {name}...')
        model = model_class(num_classes=3).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
        aug = Aug3D()
        
        best_val_bacc = 0
        patience = 10
        patience_cnt = 0
        
        for epoch in range(35):
            train_loss, train_bacc = train_epoch(model, train_loader, criterion, optimizer, aug)
            val_loss, val_bacc, _, _, _ = eval_epoch(model, val_loader, criterion)
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch {epoch+1}: train_BACC={train_bacc:.4f}, val_BACC={val_bacc:.4f}')
            
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                patience_cnt = 0
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), f'models/best_multiclass_{name.lower()}.pth')
            else:
                patience_cnt += 1
            
            if patience_cnt >= patience:
                break
        
        model.load_state_dict(torch.load(f'models/best_multiclass_{name.lower()}.pth'))
        models_trained.append(model)
    
    print('\n' + '='*70)
    print('Ensemble Inference with TTA')
    print('='*70)
    
    # Get ensemble predictions on test with TTA
    all_probs = []
    for model in models_trained:
        model.eval()
        model_probs = []
        for imgs, _ in test_loader:
            probs = predict_tta(model, imgs[0], num_tta=4)
            model_probs.append(probs)
        all_probs.append(np.array(model_probs))
    
    # Average ensemble
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    test_labels = pd.read_csv('data/test_multiclass.csv')['label'].map({'CN': 0, 'MCI': 1, 'AD': 2}).values
    
    test_bacc = balanced_accuracy_score(test_labels, ensemble_preds)
    cm = confusion_matrix(test_labels, ensemble_preds)
    f1_macro = f1_score(test_labels, ensemble_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(test_labels, ensemble_preds, average=None, zero_division=0)
    prec_per_class = precision_score(test_labels, ensemble_preds, average=None, zero_division=0)
    rec_per_class = recall_score(test_labels, ensemble_preds, average=None, zero_division=0)
    
    try:
        auc_score = roc_auc_score(test_labels, ensemble_probs, multi_class='ovr', average='macro')
    except:
        auc_score = 0.0
    
    print('\n' + '='*70)
    print('ENSEMBLE TEST RESULTS')
    print('='*70)
    print(f'Balanced Accuracy:  {test_bacc:.4f} ({test_bacc*100:.2f}%)')
    print(f'Macro F1-Score:     {f1_macro:.4f}')
    print(f'AUC (macro OvR):    {auc_score:.4f}')
    
    print('\n--- Per-Class Metrics ---')
    class_names = ['CN', 'MCI', 'AD']
    for i, name in enumerate(class_names):
        print(f'{name:5s}: Precision={prec_per_class[i]:.4f}, Recall={rec_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}')
    
    print('\n--- Confusion Matrix ---')
    print(f'        CN   MCI   AD')
    for i, name in enumerate(class_names):
        print(f'{name:3s}:  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}')
    
    if test_bacc >= 0.55:
        print(f'\n✓ TARGET 55% BACC ACHIEVED! ({test_bacc*100:.2f}%)')
    else:
        print(f'\nCurrent: {test_bacc*100:.2f}% — Target: 55.00%')
    
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        'Metric': ['Balanced Accuracy', 'Macro F1-Score', 'AUC',
                   'CN Precision', 'MCI Precision', 'AD Precision',
                   'CN Recall', 'MCI Recall', 'AD Recall'],
        'Value': [test_bacc, f1_macro, auc_score,
                  prec_per_class[0], prec_per_class[1], prec_per_class[2],
                  rec_per_class[0], rec_per_class[1], rec_per_class[2]]
    })
    results_df.to_csv('results/multiclass_ensemble_results.csv', index=False)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Ensemble Confusion Matrix (BACC={test_bacc:.4f})')
    plt.tight_layout()
    plt.savefig('results/multiclass_ensemble_confusion_matrix.png', dpi=150, bbox_inches='tight')
    
    print('\n✓ Saved ensemble results to results/multiclass_ensemble_results.csv')
    print('✓ Saved confusion matrix to results/multiclass_ensemble_confusion_matrix.png')
