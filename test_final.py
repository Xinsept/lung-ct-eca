import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import json
from datetime import datetime

from models import ResNet50_Baseline, ResNet50_ECA
from utils import val_transform, compute_metrics, print_metrics, CLASS_NAMES
from train_kfold import validate, DEVICE, BATCH_SIZE, NUM_CLASSES, OUTPUT_DIR, full_dataset, full_labels

# ===================== CONFIG =====================
DATASET_PATH = r"C:\Users\Shalom\Desktop\RESNET50+ECA\Dataset"
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
K_FOLDS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== TRAIN ON FULL TRAIN+VALID =====================
def train_final(model_name, model_class):
    print(f"\n{'='*60}")
    print(f"Training {model_name} on full train+valid for final test evaluation")
    print(f"{'='*60}")
    
    from torch.utils.data import DataLoader, ConcatDataset
    from train_kfold import train_one_epoch
    
    EPOCHS = 50
    LR = 0.001
    
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = model_class(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")
    
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}_final.pt")
    torch.save(model.state_dict(), model_path)
    return model, model_path

def evaluate_test(model, model_name):
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    
    test_path = os.path.join(DATASET_PATH, "test")
    test_dataset = ImageFolder(test_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = validate(model, test_loader, criterion)
    
    print(f"\n{model_name} Test Results:")
    print_metrics(test_metrics, prefix="Test")
    
    results = {
        'model': model_name,
        'test_accuracy': test_metrics['accuracy'],
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_test_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == "__main__":
    # Train final models on full data
    baseline_model, _ = train_final("ResNet50_Baseline_final", ResNet50_Baseline)
    eca_model, _ = train_final("ResNet50_ECA_final", ResNet50_ECA)
    
    # Evaluate on test set
    baseline_test = evaluate_test(baseline_model, "ResNet50_Baseline")
    eca_test = evaluate_test(eca_model, "ResNet50_ECA")
    
    print("\n" + "="*60)
    print("FINAL TEST SET COMPARISON")
    print("="*60)
    print(f"Baseline Test Accuracy: {baseline_test['test_accuracy']:.4f}")
    print(f"+ECA Test Accuracy:     {eca_test['test_accuracy']:.4f}")
    print(f"Δ:                      {eca_test['test_accuracy'] - baseline_test['test_accuracy']:+.4f}")