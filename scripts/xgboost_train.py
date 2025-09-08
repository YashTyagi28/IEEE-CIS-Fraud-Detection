import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

# === CONFIG ===
DATA_PATH = r"data\processed\CleanedData150_Balanced.csv"
RESULTS_DIR = "results"
MODEL_DIR = "models"
TARGET = 'isFraud'
RANDOM_STATE = 42

# === Load Data ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET].values

# === Train/Val Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=RANDOM_STATE, 
    stratify=y
)

# === XGBoost Model ===
# Calculate scale_pos_weight for imbalance handling
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=4000,
    seed=RANDOM_STATE,
    verbosity=1,
    early_stopping_rounds=50
)

# === Train ===
print("Training XGBoost model...")
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print("Training completed!")

# === Evaluate ===
print("Making predictions...")
val_preds = clf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC-AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_val, (val_preds > 0.5).astype(int)))

# === Create directories ===
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# === Save model ===
model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
clf.save_model(model_path)
print(f"Model saved to {model_path}")

# === Save predictions and metrics ===
pd.DataFrame({'y_true': y_val, 'y_pred_prob': val_preds}).to_csv(
    os.path.join(RESULTS_DIR, 'xgboost_val_predictions.csv'), index=False
)

with open(os.path.join(RESULTS_DIR, 'xgboost_val_metrics.txt'), 'w') as f:
    f.write(f'ROC-AUC: {roc_auc}\n')
    f.write(classification_report(y_val, (val_preds > 0.5).astype(int)))

print(f"Results saved to {RESULTS_DIR}")
print("XGBoost training completed successfully!")
