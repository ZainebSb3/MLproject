# src/utils.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

def charger_train_test():
    """Charge les données train/test"""
    X_train = pd.read_csv('data/train_test/X_train.csv')
    X_test = pd.read_csv('data/train_test/X_test.csv')
    y_train = pd.read_csv('data/train_test/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/train_test/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def sauvegarder_modele(model, filename):
    """Sauvegarde un modèle"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{filename}')
    print(f"✅ Modèle sauvegardé: models/{filename}")

def sauvegarder_figure(filename):
    """Sauvegarde une figure"""
    os.makedirs('reports', exist_ok=True)
    plt.savefig(f'reports/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure sauvegardée: reports/{filename}")