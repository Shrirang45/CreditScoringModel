# ============================================================
# Credit Scoring Model - Training & Evaluation
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from data_pipeline import (
    load_data,
    clean_data,
    engineer_features,
    select_model_features
)

from model_config import (
    MODEL_THRESHOLD,
    MODEL_NAME
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "UCI_Credit_Card.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD + CLEAN + FEATURE ENGINEERING
# ============================================================

print("\nLoading dataset...")

df_raw = load_data(DATA_PATH)

print("Raw dataset shape:", df_raw.shape)

df_clean = clean_data(df_raw)

df_engineered = engineer_features(df_clean)

print(
    "After feature engineering:",
    df_engineered.shape
)

df = select_model_features(
    df_engineered
)

print(
    "After feature selection:",
    df.shape
)


# ============================================================
# TARGET / FEATURES
# ============================================================

TARGET = "default"

X = df
y = df_engineered[TARGET]

print(
    "\nNumber of features:",
    X.shape[1]
)

print("\nFeature names:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget distribution (%):")
print(
    y.value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Test set remains completely untouched
# until final evaluation.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42
    )
}


# ============================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

cv_results = []


for name, model in models.items():

    print(
        f"\nTraining: {name}"
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    result = {
        "Model": name,
        "Accuracy": scores[
            "test_accuracy"
        ].mean(),
        "Precision": scores[
            "test_precision"
        ].mean(),
        "Recall": scores[
            "test_recall"
        ].mean(),
        "F1": scores[
            "test_f1"
        ].mean(),
        "ROC-AUC": scores[
            "test_roc_auc"
        ].mean(),
        "PR-AUC": scores[
            "test_pr_auc"
        ].mean()
    }

    cv_results.append(result)

    print(
        f"ROC-AUC: {result['ROC-AUC']:.4f} | "
        f"PR-AUC: {result['PR-AUC']:.4f} | "
        f"Recall: {result['Recall']:.4f} | "
        f"F1: {result['F1']:.4f}"
    )


# ============================================================
# CV RESULTS TABLE
# ============================================================

cv_results_df = pd.DataFrame(
    cv_results
)

cv_results_df = cv_results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    cv_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

# Model selection is based on ROC-AUC
# from cross-validation only.

best_model_name = cv_results_df.iloc[0]["Model"]

print(
    "\nSelected model:",
    best_model_name
)


# ============================================================
# TRAIN SELECTED MODEL
# ============================================================

best_model = models[
    best_model_name
]

print(
    f"\nTraining final {best_model_name} "
    "model on the complete training set..."
)

best_model.fit(
    X_train,
    y_train
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

y_test_proba = best_model.predict_proba(
    X_test
)[:, 1]


# Test classification metrics are reported
# at the standard 0.50 threshold.
#
# The production application uses the
# optimized MODEL_THRESHOLD = 0.25.

TEST_THRESHOLD = 0.50

y_test_pred = (
    y_test_proba >= TEST_THRESHOLD
).astype(int)


accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_test_proba
)

pr_auc = average_precision_score(
    y_test,
    y_test_proba
)


print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE FINAL MODEL ARTIFACT
# ============================================================

artifact = {

    "model": best_model,

    "feature_names": X.columns.tolist(),

    "target_name": TARGET,

    "threshold": MODEL_THRESHOLD,

    "model_name": MODEL_NAME,

    "feature_count": len(
        X.columns
    ),

    "cv_results": cv_results_df,

    "test_metrics_at_0_50": {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc
    }
}


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "credit_scoring_model.joblib"
)


joblib.dump(
    artifact,
    MODEL_PATH
)


print(
    f"\nArtifact saved to:\n{MODEL_PATH}"
)

print(
    "\nTraining completed successfully."
)