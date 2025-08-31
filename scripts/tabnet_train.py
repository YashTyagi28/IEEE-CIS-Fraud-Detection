import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import torch  # Add this import

# === CONFIG ===
DATA_PATH = r"data\processed\CleanedData150_Balanced.csv"
RESULTS_DIR = "results"
MODEL_DIR = "models"
TARGET = 'isFraud'
RANDOM_STATE = 42

# === Load Data ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET]).values
y = df[TARGET].values

# === Train/Val Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# === TabNet Model ===
clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='sparsemax',
    seed=RANDOM_STATE
)

# === Train ===
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['auc'],
    max_epochs=25,
    patience=10,
    batch_size=512,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# === Evaluate and Save ===
val_preds = clf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC-AUC: {roc_auc:.4f}")
print(classification_report(y_val, (val_preds > 0.5).astype(int)))

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

# Save model
clf.save_model(os.path.join(MODEL_DIR, "tabnet_model.zip"))

# Save predictions and metrics
pd.DataFrame({'y_true': y_val, 'y_pred_prob': val_preds}).to_csv(
    os.path.join(RESULTS_DIR, 'tabnet_val_predictions.csv'), index=False
)
with open(os.path.join(RESULTS_DIR, 'tabnet_val_metrics.txt'), 'w') as f:
    f.write(f'ROC-AUC: {roc_auc}\n')
    f.write(classification_report(y_val, (val_preds > 0.5).astype(int)))

# Save feature importance
pd.DataFrame({
    'feature': df.drop(columns=[TARGET]).columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False).to_csv(
    os.path.join(RESULTS_DIR, 'tabnet_feature_importances.csv'), index=False
)
