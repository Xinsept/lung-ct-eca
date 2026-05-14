import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from models import ResNet50_Baseline, ResNet50_ECA
from utils import train_transform, val_transform, compute_metrics, print_metrics, CLASS_NAMES

# ===================== CONFIG =====================
DATASET_PATH = r"C:\Users\Shalom\Desktop\RESNET50+ECA\Dataset"
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
K_FOLDS = 5
SEED = 42

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# Output dir for checkpoints and logs
OUTPUT_DIR = os.path.join(os.path.dirname(DATASET_PATH), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== DATA =====================
train_path = os.path.join(DATASET_PATH, "train")
valid_path = os.path.join(DATASET_PATH, "valid")

train_dataset = ImageFolder(train_path, transform=train_transform)
valid_dataset = ImageFolder(valid_path, transform=val_transform)

# Merge train + valid for k-fold
full_dataset = ConcatDataset([train_dataset, valid_dataset])
full_labels = [train_dataset.targets[i] for i in range(len(train_dataset))] + \
              [valid_dataset.targets[i] for i in range(len(valid_dataset))]
full_labels = np.array(full_labels)

print(f"Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}")
print(f"Full dataset size: {len(full_dataset)}")
print(f"Class distribution: {np.bincount(full_labels)}")

# ===================== K-FOLD TRAINING =====================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    return running_loss / len(loader), metrics

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    return running_loss / len(loader), metrics

def run_kfold(model_name, model_class):
    """Run K-fold CV for a given model class"""
    print(f"\n{'='*60}")
    print(f"Running {K_FOLDS}-Fold CV for {model_name}")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    best_model_path = os.path.join(OUTPUT_DIR, f"{model_name}_best_fold.pt")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_labels)), full_labels)):
        print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        model = model_class(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        best_val_acc = 0.0
        
        for epoch in range(EPOCHS):
            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_metrics = validate(model, val_loader, criterion)
            scheduler.step()
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), best_model_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        _, final_val_metrics = validate(model, val_loader, criterion)
        
        print(f"\n  Fold {fold+1} Best Val Accuracy: {best_val_acc:.4f}")
        print_metrics(final_val_metrics, prefix=f"Fold {fold+1}")
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_accuracy': best_val_acc,
            'final_metrics': final_val_metrics
        })
    
    # Summary across folds
    accuracies = [r['best_val_accuracy'] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"{model_name} {K_FOLDS}-Fold CV Summary")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Per-fold: {[f'{a:.4f}' for a in accuracies]}")
    
    # Save results
    results = {
        'model': model_name,
        'k_folds': K_FOLDS,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'per_fold_accuracies': accuracies,
        'fold_details': fold_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_kfold_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

# ===================== RUN =====================
if __name__ == "__main__":
    baseline_results = run_kfold("ResNet50_Baseline", ResNet50_Baseline)
    eca_results = run_kfold("ResNet50_ECA", ResNet50_ECA)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Baseline: {baseline_results['mean_accuracy']:.4f} ± {baseline_results['std_accuracy']:.4f}")
    print(f"+ECA:     {eca_results['mean_accuracy']:.4f} ± {eca_results['std_accuracy']:.4f}")
    print(f"Δ:        {eca_results['mean_accuracy'] - baseline_results['mean_accuracy']:+.4f}")