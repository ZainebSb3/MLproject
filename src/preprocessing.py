import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "retail_customers.csv")

df = pd.read_csv(DATA_PATH)

print("Shape initial:", df.shape)

# =========================
# TARGET
# =========================
target = "Churn"

# =========================
# DROP IDENTIFIERS + DATES (IMPORTANT)
# =========================
drop_cols = []

for col in df.columns:
    if "id" in col.lower():
        drop_cols.append(col)

# essayer de détecter les dates automatiquement
for col in df.columns:
    if "date" in col.lower():
        drop_cols.append(col)

df = df.drop(columns=drop_cols, errors="ignore")

# =========================
# SPLIT X / Y
# =========================
X = df.drop(columns=[target])
y = df[target]

# =========================
# KEEP ONLY NUMERIC FOR NOW (IMPORTANT FIX)
# =========================
X = X.select_dtypes(include=[np.number])

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# IMPUTATION
# =========================
imputer = KNNImputer(n_neighbors=5)

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

X = pd.get_dummies(X, drop_first=True)
# =========================
# SCALING
# =========================
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# =========================
# SMOTE
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# SAVE
# =========================
# =========================
# SAVE CORRECT
# =========================
os.makedirs(os.path.join(BASE_DIR, "data", "train_test"), exist_ok=True)

X_train.to_csv(os.path.join(BASE_DIR, "data", "train_test", "X_train.csv"), index=False)
X_test.to_csv(os.path.join(BASE_DIR, "data", "train_test", "X_test.csv"), index=False)
y_train.to_csv(os.path.join(BASE_DIR, "data", "train_test", "y_train.csv"), index=False)
y_test.to_csv(os.path.join(BASE_DIR, "data", "train_test", "y_test.csv"), index=False)

print("✔ preprocessing OK")
print("\n=== DATA SHAPES ===")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)