import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

import joblib

# =========================
# LOAD DATA
# =========================
X_test = pd.read_csv("data/train_test/X_test_fe.csv")
y_test = pd.read_csv("data/train_test/y_test.csv").values.ravel()

# imputer + scaler (mêmes que training si sauvegardés)
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

X_test = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test)

# =========================
# LOAD MODEL (BEST = XGBOOST)
# =========================
model = joblib.load("models/xgboost.pkl")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.show()