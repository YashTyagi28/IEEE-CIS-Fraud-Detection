# IEEE-CIS Fraud Detection

## Overview
Fraud detection system using TabNet and CatBoost with stacking ensemble for the IEEE-CIS dataset.

## Workflow

### 1. Data Processing
The dataset is preprocessed and cleaned in Jupyter notebooks.

### 2. Base Model Training
Two models are trained independently:
- **TabNet** (`scripts/tabnet_train.py`): Deep neural network with attention mechanism
- **CatBoost** (`scripts/catboost_train.py`): Gradient boosting classifier

Each model saves:
- Trained model file in `models/`  
- Validation predictions in `results/`

### 3. Stacking Ensemble
The stacking script (`scripts/stacking.py`) combines both models:
- Loads validation predictions from both base models
- Uses predicted probabilities as features for a Logistic Regression meta-model
- Trains the meta-model to optimally combine base predictions

### 4. Results
Final stacked model achieves >95% ROC-AUC on validation set.

## How to Run
```bash
# Train base models
python scripts/tabnet_train.py
python scripts/catboost_train.py

# Train stacking ensemble  
python scripts/stacking.py
```

## Dependencies
```bash
pip install pytorch-tabnet catboost scikit-learn pandas numpy torch joblib
```