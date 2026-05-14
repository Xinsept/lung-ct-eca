# ECA-Enhanced ResNet50 for Lung CT Classification

Code repository for "ECA-Enhanced ResNet50 for Lung Cancer Subtype Classification on Small-Scale CT Datasets."

## Files
- `models.py` — ResNet50 baseline and ECA-enhanced architectures
- `train_kfold.py` — 5-fold stratified cross-validation
- `test_final.py` — final training on all 685 images and test evaluation
- `utils.py` — transforms, metrics, and utilities
- `*.json` — complete CV and test results
- `*.png` — confusion matrices and dataset samples

## Dataset
Lung CT images from Hammad et al. (2025), available on Kaggle:
https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

## Environment
Python 3.11, PyTorch 2.x, torchvision, numpy, scikit-learn, tqdm

## Usage
1. Download the dataset from the Kaggle link above and place it in `Dataset/`
2. `python train_kfold.py`
3. `python test_final.py`
