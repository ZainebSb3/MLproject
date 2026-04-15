import joblib
import numpy as np
import os
import pandas as pd

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =========================
# LOAD ARTIFACTS
# =========================
model = joblib.load(os.path.join(MODELS_DIR, "final_model.pkl"))
imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

# =========================
# FEATURE ENGINEERING (SAME AS TRAINING)
# =========================
def build_features(data: dict):
    df = pd.DataFrame([data])

    # features engineering cohérentes avec training
    df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    df["TenureRatio"] = df["Recency"] / (df["CustomerTenureDays"] + 1)

    return df


# =========================
# ALIGN FEATURES (CRITICAL FIX)
# =========================
def align_features(df: pd.DataFrame):
    # ajouter colonnes manquantes
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # ordre EXACT identique au training
    df = df[feature_columns]

    return df


# =========================
# MAIN PREDICTION
# =========================
def predict_churn(input_data: dict):

    # 1. build features
    df = build_features(input_data)

    # 2. align with training features
    df = align_features(df)

    # 3. convert to numpy
    x = df.values

    # 4. preprocessing pipeline (same as training)
    x = imputer.transform(x)
    x = scaler.transform(x)
    x = pca.transform(x)

    # 5. prediction
    proba = model.predict_proba(x)[0][1]

    return round(float(proba) * 100, 2)