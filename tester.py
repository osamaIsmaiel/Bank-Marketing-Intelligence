import json
from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).parent
data = pd.read_csv(ROOT / "data" / "bank-additional-full.csv", sep=";").drop_duplicates()
y = (data["y"] == "yes").astype(int)
X = data.drop(columns=["y", "duration"])  # duration is excluded to avoid leakage

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

model = load(ROOT / "artifacts" / "best_model.joblib")
metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
threshold = float(metadata["decision_threshold"])
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, probabilities)

print(f"Model: {metadata['best_model']}")
print(f"Threshold: {threshold:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("PASS:", precision >= 0.30 and recall >= 0.30)
