import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, classification_report)
from src.features import BankFeatureEngineer

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "bank-additional-full.csv", sep=";").drop_duplicates()
y = (df["y"] == "yes").astype(int)
X = df.drop(columns=["y", "duration"])  # duration is post-call information leakage

# Stratified 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

engineered_train = BankFeatureEngineer().fit_transform(X_train)
numeric = engineered_train.select_dtypes(include="number").columns.tolist()
categorical = engineered_train.select_dtypes(exclude="number").columns.tolist()
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])

def make_pipeline(classifier):
    return Pipeline([
        ("feature_engineering", BankFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("feature_selection", SelectFromModel(
            ExtraTreesClassifier(n_estimators=40, random_state=42,
                                 class_weight="balanced", n_jobs=-1),
            threshold="median")),
        ("classifier", classifier)
    ])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(
    make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced",
                                      solver="liblinear")),
    {"classifier__C": [0.1, 1.0, 3.0]},
    scoring="f1", cv=cv, n_jobs=-1
)
grid.fit(X_train, y_train)

models = {
    "Logistic Regression (tuned)": grid.best_estimator_,
    "Random Forest": make_pipeline(RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    "Extra Trees": make_pipeline(ExtraTreesClassifier(
        n_estimators=100, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1))
}
for name, model in models.items():
    if name != "Logistic Regression (tuned)":
        model.fit(X_train, y_train)

def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12)
    valid = (precision[:-1] >= 0.30) & (recall[:-1] >= 0.30)
    index = (np.where(valid)[0][np.argmax(f1_values[valid])]
             if valid.any() else int(np.argmax(f1_values)))
    return float(thresholds[index])

rows = []
thresholds = {}
for name, model in models.items():
    validation_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, validation_prob)
    validation_pred = (validation_prob >= threshold).astype(int)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    thresholds[name] = threshold
    rows.append({
        "model": name,
        "validation_threshold": threshold,
        "validation_precision": precision_score(y_val, validation_pred, zero_division=0),
        "validation_recall": recall_score(y_val, validation_pred, zero_division=0),
        "validation_f1": f1_score(y_val, validation_pred, zero_division=0),
        "test_accuracy_before_refit": accuracy_score(y_test, test_pred),
        "test_precision_before_refit": precision_score(y_test, test_pred, zero_division=0),
        "test_recall_before_refit": recall_score(y_test, test_pred, zero_division=0),
        "test_f1_before_refit": f1_score(y_test, test_pred, zero_division=0),
        "test_roc_auc_before_refit": roc_auc_score(y_test, test_prob),
        "test_average_precision_before_refit": average_precision_score(y_test, test_prob)
    })

# Select the model using validation F1 only — never choose a model using test results.
comparison = pd.DataFrame(rows).sort_values("validation_f1", ascending=False)
best_name = comparison.iloc[0]["model"]
best_threshold = thresholds[best_name]
best_model = models[best_name]

# Refit selected model on training + validation data; keep threshold from validation.
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
best_model.fit(X_train_val, y_train_val)
dump(best_model, ROOT / "artifacts" / "best_model.joblib")

# Final evaluation of the saved production pipeline on the untouched test split.
test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)
final_metrics = {
    "selected_model": best_name,
    "selection_basis": "highest validation F1; test set not used for model selection",
    "decision_threshold": best_threshold,
    "precision": precision_score(y_test, test_pred, zero_division=0),
    "recall": recall_score(y_test, test_pred, zero_division=0),
    "f1": f1_score(y_test, test_pred, zero_division=0),
    "accuracy": accuracy_score(y_test, test_pred),
    "roc_auc": roc_auc_score(y_test, test_prob),
    "average_precision": average_precision_score(y_test, test_prob),
    "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "best_logistic_grid_params": grid.best_params_
}
(ROOT / "reports" / "model_comparison.csv").write_text(comparison.to_csv(index=False))
(ROOT / "reports" / "final_test_metrics.json").write_text(json.dumps(final_metrics, indent=2))
(ROOT / "reports" / "classification_report.txt").write_text(
    classification_report(y_test, test_pred, target_names=["no", "yes"], zero_division=0))
metadata = {
    "best_model": best_name,
    "decision_threshold": best_threshold,
    "target": "y (yes=1, no=0)",
    "excluded_feature": "duration (post-call information leakage)",
    "best_logistic_grid_params": grid.best_params_,
    "threshold_selected_on": "validation split",
    "model_selection_basis": "validation F1",
    "final_model_refit_on": "training + validation"
}
(ROOT / "artifacts" / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
print(comparison.to_string(index=False))
print("\\nFINAL HELD-OUT TEST METRICS")
print(json.dumps({k: final_metrics[k] for k in
    ["selected_model","precision","recall","f1","accuracy","roc_auc","average_precision"]}, indent=2))
