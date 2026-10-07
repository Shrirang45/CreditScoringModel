# ============================================================
# Credit Scoring - Production Prediction Module
# ============================================================

import os
import joblib
import pandas as pd

from src.data_pipeline import (
    clean_data,
    engineer_features,
    select_model_features
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_scoring_model.joblib"
)


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

artifact = joblib.load(
    MODEL_PATH
)

model = artifact["model"]

FEATURE_NAMES = artifact["feature_names"]

THRESHOLD = artifact["threshold"]

MODEL_NAME = artifact["model_name"]


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df):
    """
    Validate that the input contains the
    required original customer features.
    """

    required_columns = [
        'LIMIT_BAL',
        'SEX',
        'EDUCATION',
        'MARRIAGE',
        'AGE',
        'PAY_1',
        'PAY_2',
        'PAY_3',
        'PAY_4',
        'PAY_5',
        'PAY_6',
        'BILL_AMT1',
        'BILL_AMT2',
        'BILL_AMT3',
        'BILL_AMT4',
        'BILL_AMT5',
        'BILL_AMT6',
        'PAY_AMT1',
        'PAY_AMT2',
        'PAY_AMT3',
        'PAY_AMT4',
        'PAY_AMT5',
        'PAY_AMT6'
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return True


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(df):
    df_clean = clean_data(df)
    validate_input(df_clean)

    df_engineered = engineer_features(df_clean)
    X = select_model_features(df_engineered)
    X = X[FEATURE_NAMES]

    return X


# ============================================================
# PREDICT
# ============================================================

def predict(df):
    """
    Generate credit-default predictions.

    Returns:
        DataFrame containing:
        - default_probability
        - prediction
        - risk
    """

    X = prepare_input(df)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    results = pd.DataFrame({
        "default_probability": probabilities,
        "prediction": predictions
    })

    results["risk"] = results[
        "prediction"
    ].map({
        0: "Low Risk",
        1: "High Risk"
    })

    return results