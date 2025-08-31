import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from catboost import CatBoostClassifier

# === CONFIG ===
DATA_PATH = r"data\processed\CleanedData150_Balanced.csv"
RESULTS_DIR = "results"
MODEL_DIR = "models"
TARGET = 'isFraud'
RANDOM_STATE = 42

# === Load Data ===
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df[TARGET].value_counts()}")

X = df.drop(columns=[TARGET])
y = df[TARGET].values

# === Train/Val Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# === CatBoost Model ===
clf = CatBoostClassifier(
    iterations=4000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    random_seed=RANDOM_STATE,
    verbose=100  # Print every 50 iterations
)

# === Train ===
print("Training CatBoost model...")
clf.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=50,
    plot=False
)

print("Training completed!")

# === Evaluate ===
print("Making predictions...")
val_preds = clf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC-AUC: {roc_auc:.4f}")
print('Classification Report:')
print(classification_report(y_val, (val_preds > 0.5).astype(int)))

# === Create directories ===
if not os.path.exists(MODEL_DIR): 
    os.makedirs(MODEL_DIR)
if not os.path.exists(RESULTS_DIR): 
    os.makedirs(RESULTS_DIR)

# === Save model ===
model_path = os.path.join(MODEL_DIR, "catboost_model.cbm")
clf.save_model(model_path)
print(f"Model saved to {model_path}")

# === Save predictions and metrics ===
pd.DataFrame({'y_true': y_val, 'y_pred_prob': val_preds}).to_csv(
    os.path.join(RESULTS_DIR, 'catboost_val_predictions.csv'), index=False
)

with open(os.path.join(RESULTS_DIR, 'catboost_val_metrics.txt'), 'w') as f:
    f.write(f'ROC-AUC: {roc_auc}\n')
    f.write(classification_report(y_val, (val_preds > 0.5).astype(int)))

# === Save feature importance ===
feature_importance = clf.get_feature_importance()
pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False).to_csv(
    os.path.join(RESULTS_DIR, 'catboost_feature_importances.csv'), index=False
)

print(f"Results saved to {RESULTS_DIR}")
print("CatBoost training completed successfully!")
