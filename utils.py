import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchvision import transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

CLASS_NAMES = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']


def compute_metrics(all_labels, all_preds):
    """Compute accuracy, per-class precision/recall/f1, confusion matrix"""
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    
    metrics_dict = {
        'accuracy': accuracy,
        'per_class': {}
    }
    for i, name in enumerate(CLASS_NAMES):
        metrics_dict['per_class'][name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    metrics_dict['confusion_matrix'] = cm.tolist()
    
    return metrics_dict


def print_metrics(metrics_dict, prefix=""):
    """Pretty print metrics"""
    print(f"\n{prefix} Accuracy: {metrics_dict['accuracy']:.4f}")
    for name, m in metrics_dict['per_class'].items():
        print(f"  {name:25s} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f} | N: {m['support']}")