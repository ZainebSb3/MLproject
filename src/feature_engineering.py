import pandas as pd
import numpy as np
import os

# =========================
# LOAD DATA
# =========================
X_train = pd.read_csv("data/train_test/X_train.csv")
X_test = pd.read_csv("data/train_test/X_test.csv")
y_train = pd.read_csv("data/train_test/y_train.csv")
y_test = pd.read_csv("data/train_test/y_test.csv")
print("Columns:", X_train.columns)

# =========================
# SAFE FEATURE ENGINEERING
# =========================
def add_features(df):
    df = df.copy()

    # ===== SAFE ACCESS =====
    recency = df["Recency"] if "Recency" in df.columns else 1
    frequency = df["Frequency"] if "Frequency" in df.columns else 1
    monetary = df["MonetaryTotal"] if "MonetaryTotal" in df.columns else 1

    # CustomerTenure peut ne pas exister
    if "CustomerTenure" in df.columns:
        tenure = df["CustomerTenure"].replace(0, 1)
    elif "CustomerTenureDays" in df.columns:
        tenure = df["CustomerTenureDays"].replace(0, 1)
    else:
        print("⚠️ No tenure column → proxy used")
        tenure = recency + 1  # approximation réaliste

    recency = recency.replace(0, 1)

    # =========================
    # FEATURES (aligned rapport)
    # =========================
    df["MonetaryPerDay"] = monetary / recency
    df["AvgBasketValue"] = monetary / (frequency + 1)
    df["TenureRatio"] = recency / tenure
    df["PurchaseIntensity"] = frequency / tenure
    df["RecencyScore"] = 1 / (df["Recency"] + 1)
    df["HighValueCustomer"] = (df["MonetaryTotal"] > df["MonetaryTotal"].median()).astype(int)
    df["LogMonetary"] = np.log1p(df["MonetaryTotal"])
    df["RecencySquared"] = df["Recency"] ** 2
    df["FreqPerRecency"] = df["Frequency"] / (df["Recency"] + 1)
    df["ValueScore"] = df["MonetaryTotal"] * df["Frequency"]
    df["EngagementScore"] = df["TotalTransactions"] / (df["Recency"] + 1)
    return df

# Apply
X_train = add_features(X_train)
X_test = add_features(X_test)

# =========================
# SAVE
# =========================
os.makedirs("data/train_test", exist_ok=True)

X_train.to_csv("data/train_test/X_train_fe.csv", index=False)
X_test.to_csv("data/train_test/X_test_fe.csv", index=False)
y_train.to_csv("data/train_test/y_train_fe.csv", index=False)
y_test.to_csv("data/train_test/y_test_fe.csv", index=False)
print("✔ Feature engineering OK")
print("Train:", X_train.shape)
print("Test:", X_test.shape)