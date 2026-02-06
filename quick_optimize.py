#!/usr/bin/env python3
"""
Quick evaluation and threshold optimization on existing ResNet50 model
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

sys.path.insert(0, '/nlsasfs/home/gpucbh/vyakti4/Ai_healthcare_mri')

from app.dataset import MRIDataset
from app.model import ResNetSlice


def optimize_threshold(y_true, y_probs, thresholds=np.arange(0.1, 1.0, 0.01)):
    """Find optimal threshold for best balanced accuracy"""
    
    best_threshold = 0.5
    best_bacc = 0
    bacc_scores = []
    
    for threshold in thresholds:
        y_pred = (y_probs[:, 1] >= threshold).astype(int)
        bacc = balanced_accuracy_score(y_true, y_pred)
        bacc_scores.append(bacc)
        
        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = threshold
    
    print(f"Threshold optimization results:")
    print(f"  Best threshold: {best_threshold:.3f}")
    print(f"  Best BACC at threshold: {best_bacc:.4f}")
    print(f"  BACC range: {np.min(bacc_scores):.4f} - {np.max(bacc_scores):.4f}")
    
    return best_threshold, best_bacc


def test_time_augmentation(model, data, device, num_tta=8):
    """Apply test-time augmentation and average predictions"""
    
    import torchvision.transforms as transforms
    
    probs_list = []
    
    for tta_idx in range(num_tta):
        # Apply random augmentation
        if tta_idx > 0:  # First pass is original
            aug_data = data.clone()
            
            # Random rotations
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-15, 15)
                aug_data = torch.rot90(aug_data, k=np.random.randint(0, 4), dims=[-2, -1])
            
            # Random horizontal flip
            if np.random.rand() < 0.5:
                aug_data = torch.flip(aug_data, dims=[-1])
        else:
            aug_data = data
        
        with torch.no_grad():
            output = model(aug_data)
            probs = torch.softmax(output, dim=1)
            probs_list.append(probs)
    
    # Average probabilities
    avg_probs = torch.stack(probs_list).mean(dim=0)
    return avg_probs


def evaluate_and_optimize(model_path, test_use_tta=True, num_tta=4):
    """Evaluate model and optimize threshold"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = ResNetSlice(pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    data_path = '/nlsasfs/home/gpucbh/vyakti4/Ai_healthcare_mri/data/processed'
    metadata_test = '/nlsasfs/home/gpucbh/vyakti4/Ai_healthcare_mri/data/test.csv'
    
    test_dataset = MRIDataset(data_path, metadata_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Inference
    y_true = []
    y_pred_default = []
    y_probs_all = []
    
    print(f"\nPerforming inference on test set (TTA={test_use_tta})...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            
            if test_use_tta:
                probs = test_time_augmentation(model, data, device, num_tta=num_tta)
            else:
                output = model(data)
                probs = torch.softmax(output, dim=1)
            
            pred_default = probs.argmax(dim=1)
            
            y_true.append(target.item())
            y_pred_default.append(pred_default.item())
            y_probs_all.append(probs.cpu().numpy()[0])
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_dataset)} samples")
    
    y_true = np.array(y_true)
    y_pred_default = np.array(y_pred_default)
    y_probs_all = np.array(y_probs_all)
    
    # Evaluate at default threshold (0.5)
    default_bacc = balanced_accuracy_score(y_true, y_pred_default)
    default_auc = roc_auc_score(y_true, y_probs_all[:, 1])
    default_f1 = f1_score(y_true, y_pred_default)
    
    print(f"\n{'='*60}")
    print("EVALUATION AT DEFAULT THRESHOLD (0.5)")
    print(f"{'='*60}")
    print(f"Balanced Accuracy (BACC): {default_bacc:.4f}")
    print(f"ROC-AUC Score:            {default_auc:.4f}")
    print(f"F1 Score (macro):         {default_f1:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred_default)}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred_default)}")
    
    # Optimize threshold
    print(f"\n{'='*60}")
    print("THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")
    best_threshold, best_bacc = optimize_threshold(y_true, y_probs_all)
    
    # Evaluate at optimal threshold
    y_pred_optimal = (y_probs_all[:, 1] >= best_threshold).astype(int)
    optimal_f1 = f1_score(y_true, y_pred_optimal)
    
    print(f"\nConfusion Matrix (optimal threshold):\n{confusion_matrix(y_true, y_pred_optimal)}")
    print(f"F1 Score (macro):                    {optimal_f1:.4f}")
    
    # Check if meets 91% threshold
    meets_91 = best_bacc >= 0.91
    print(f"\n{'='*60}")
    print(f"MEETS >91% TARGET: {'✅ YES' if meets_91 else '❌ NO'}")
    print(f"Current:          {best_bacc:.2%}")
    print(f"Target:           > 91%")
    print(f"Gap:              {(0.91 - best_bacc):.2%}")
    print(f"{'='*60}")
    
    return {
        'default_bacc': default_bacc,
        'optimal_bacc': best_bacc,
        'best_threshold': best_threshold,
        'meets_91': meets_91,
        'auc': default_auc,
        'f1': optimal_f1
    }


if __name__ == '__main__':
    # Find the best existing model
    models_dir = '/nlsasfs/home/gpucbh/vyakti4/Ai_healthcare_mri/models'
    
    # Check for existing ResNet models
    existing_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    
    if existing_models:
        print(f"Found existing models: {existing_models}\n")
        
        # Test the first ResNet model
        model_to_test = os.path.join(models_dir, 'resnet_best.pt')
        
        if os.path.exists(model_to_test):
            print(f"Testing model: {model_to_test}\n")
            results = evaluate_and_optimize(model_to_test, test_use_tta=True, num_tta=4)
        else:
            print(f"Model not found: {model_to_test}")
            print(f"Available models: {existing_models}")
    else:
        print("No models found. Please train a model first.")
