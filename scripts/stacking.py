import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# === CONFIG ===
RESULTS_DIR = "results"
MODEL_DIR = "models"
RANDOM_STATE = 42

# === Load Base Model Predictions ===
tabnet_preds = pd.read_csv(os.path.join(RESULTS_DIR, 'tabnet_val_predictions.csv'))
catboost_preds = pd.read_csv(os.path.join(RESULTS_DIR, 'catboost_val_predictions.csv'))

# === Validate alignment of labels ===
assert (tabnet_preds['y_true'] == catboost_preds['y_true']).all(), "Mismatch in validation labels between models!"

# === Prepare meta-features and target ===
X_meta = pd.DataFrame({
    'tabnet': tabnet_preds['y_pred_prob'],
    'catboost': catboost_preds['y_pred_prob']
})
y_meta = tabnet_preds['y_true']

# === Train/Val split for meta-model (to avoid overfitting on meta-model) ===
X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
    X_meta, y_meta, test_size=0.2, random_state=RANDOM_STATE, stratify=y_meta
)

# === Train Logistic Regression Meta-Model ===
meta_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
meta_clf.fit(X_train_meta, y_train_meta)

# === Prediction & Evaluation ===
val_probs = meta_clf.predict_proba(X_val_meta)[:, 1]
roc_auc = roc_auc_score(y_val_meta, val_probs)
print(f"Stacked Meta-Model Validation ROC-AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_val_meta, (val_probs > 0.5).astype(int)))

# === Save meta-model ===
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
meta_model_path = os.path.join(MODEL_DIR, "stacking_meta_model.pkl")
joblib.dump(meta_clf, meta_model_path)
print(f"Meta-model saved to {meta_model_path}")

# === Save stacked validation predictions ===
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
stacked_results = X_val_meta.copy()
stacked_results['y_true'] = y_val_meta
stacked_results['meta_pred_prob'] = val_probs
stacked_results.to_csv(os.path.join(RESULTS_DIR, 'stacked_val_predictions.csv'), index=False)

# === Save stacked validation metrics to CSV ===
metrics_report = classification_report(y_val_meta, (val_probs > 0.5).astype(int), output_dict=True)
metrics_df = pd.DataFrame(metrics_report).transpose()
metrics_df.to_csv(os.path.join(RESULTS_DIR, 'stacked_val_metrics.csv'))

print("Stacking pipeline completed successfully and metrics saved.")
