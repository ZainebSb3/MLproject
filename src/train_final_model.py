import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# =========================
# LOAD DATA
# =========================
print("Loading data...")
X_train = pd.read_csv("data/train_test/X_train_fe.csv")
X_test = pd.read_csv("data/train_test/X_test_fe.csv")
y_train = pd.read_csv("data/train_test/y_train.csv").values.ravel()
y_test = pd.read_csv("data/train_test/y_test.csv").values.ravel()

print(f"Initial shapes: X_train={X_train.shape}, X_test={X_test.shape}")
print(f"Churn rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

# =========================
# REMOVE CONSTANT FEATURES
# =========================
const_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
X_train = X_train.drop(columns=const_cols)
X_test = X_test.drop(columns=const_cols)
print(f"Removed {len(const_cols)} constant features")

# =========================
# REMOVE LEAK FEATURES (VOTRE APPROCHE - CORRECTE)
# =========================
leak_cols = ["ChurnRiskCategory", "RFMSegment", "LoyaltyLevel", "ChurnRisk"]
X_train = X_train.drop(columns=[c for c in leak_cols if c in X_train.columns], errors="ignore")
X_test = X_test.drop(columns=[c for c in leak_cols if c in X_test.columns], errors="ignore")
print("Leak-like features removed ✔")

# =========================
# TRAIN/VALID SPLIT (AVANT TOUT TRAITEMENT)
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
print(f"After split - Train: {X_train.shape}, Val: {X_val.shape}")

# =========================
# IMPUTATION (FIT SUR TRAIN UNIQUEMENT)
# =========================
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

# =========================
# PCA
# =========================
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]} components")

# =========================
# SMOTE (UNIQUEMENT SUR TRAIN)
# =========================
print(f"Before SMOTE - Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)}")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)
print(f"After SMOTE - Class 0: {sum(y_train_smote==0)}, Class 1: {sum(y_train_smote==1)}")

# =========================
# MODELS AVEC GRIDSEARCH (OPTIONNEL MAIS RECOMMANDÉ)
# =========================
models = {
    "logistic_regression": {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C': [0.1, 0.5, 1.0, 2.0]}
    },
    "random_forest": {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10]
        }
    },
    "xgboost": {
        'model': XGBClassifier(eval_metric="logloss", random_state=42),
        'params': {
            'n_estimators': [150, 200, 250],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.03, 0.05, 0.07]
        }
    }
}

results = {}
trained_models = {}

for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    
    # GridSearchCV
    grid = GridSearchCV(
        config['model'], config['params'],
        cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid.fit(X_train_smote, y_train_smote)
    
    best_model = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    
    # Validation pour threshold tuning
    val_proba = best_model.predict_proba(X_val_pca)[:, 1]
    
    # Optimisation du seuil sur validation
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_t = 0.5
    best_f1 = 0
    
    for t in thresholds:
        pred = (val_proba >= t).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    
    # Évaluation finale sur test
    test_proba = best_model.predict_proba(X_test_pca)[:, 1]
    y_pred = (test_proba >= best_t).astype(int)
    
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, test_proba)
    }
    
    trained_models[name] = best_model
    print(f"  ✓ Test Accuracy: {results[name]['accuracy']:.4f}")
    print(f"  ✓ Test ROC-AUC: {results[name]['roc_auc']:.4f}")
    print(f"  ✓ Best threshold: {best_t:.2f}")

# =========================
# BEST MODEL
# =========================
results_df = pd.DataFrame(results).T
best_name = results_df["roc_auc"].idxmax()
best_model = trained_models[best_name]

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(results_df.round(4))
print("="*60)
print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   ROC-AUC: {results_df.loc[best_name, 'roc_auc']:.4f}")
print(f"   Accuracy: {results_df.loc[best_name, 'accuracy']:.4f}")

# =========================
# SAVE
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/final_model.pkl")
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")
print("\n✅ Models saved to models/")