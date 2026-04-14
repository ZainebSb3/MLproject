# 📊 Prédiction du Churn Client avec Machine Learning

## 🎯 Objectif du projet
Ce projet a pour objectif de prédire le **churn client** (départ des clients) dans un contexte e-commerce à l’aide de techniques de Machine Learning.

L’enjeu est d’identifier les clients à risque afin de mettre en place des actions de fidélisation ciblées et réduire la perte de revenus.

---

## 📁 Structure du projet


projet_ml_retail/
│
├── app/ # Application Flask (API + interface web)
├── src/ # Scripts de traitement et entraînement
├── notebooks/ # Analyse exploratoire (EDA)
├── reports/ # Graphiques et résultats
├── data/ # Données (non versionnées)
├── models/ # Modèles sauvegardés (non versionnés)
└── README.md


---

## ⚙️ Pipeline Machine Learning

Le pipeline complet suit les étapes suivantes :

1. Nettoyage des données  
2. Feature Engineering  
3. Normalisation (StandardScaler)  
4. Réduction dimensionnelle (PCA : 43 → 23 composantes)  
5. Équilibrage des classes (SMOTE)  
6. Entraînement des modèles  
7. Optimisation avec GridSearchCV  

---

## 🤖 Modèles utilisés

- Régression Logistique  
- Random Forest  
- XGBoost  

---

## 📈 Résultats

| Modèle | Accuracy | ROC-AUC |
|--------|----------|---------|
| Logistic Regression | 0.9794 | 0.9956 |
| Random Forest | 0.9657 | 0.9945 |
| **XGBoost** | **0.9680** | **0.9963** |

🏆 **Meilleur modèle : XGBoost**

---

## 🚀 Lancer le projet

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
2. Entraîner le modèle
python src/train_final_model.py
3. Lancer l’application Flask
cd app
python app.py
4. Accéder à l’application
http://localhost:5000
📡 API
Endpoint : POST /predict
Requête :
{
  "Recency": 30,
  "Frequency": 8,
  "MonetaryTotal": 1200,
  "AvgBasketValue": 150,
  "CustomerTenureDays": 365
}
Réponse :
{
  "probability": 23.5
}
🧠 Technologies utilisées
Python
scikit-learn
XGBoost
Pandas / NumPy
Flask
Matplotlib
⚠️ Remarque

Les dossiers suivants ne sont pas versionnés pour alléger le dépôt :

data/
models/
venv/
👨‍💻 Auteur
Zaineb Sboui
Master GI2S4
📌 Conclusion

Ce projet démontre l’utilisation complète d’un pipeline de Machine Learning :

Prétraitement des données
Modélisation
Évaluation
Déploiement via une API Flask

Il permet de répondre à une problématique métier réelle : la réduction du churn client.


