import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "processed", "train_fe.csv")
df = pd.read_csv(train_path)

target = "Churn"

X = df.drop(columns=[target])

# =========================
# PCA
# =========================
pca = PCA()
X_pca = pca.fit_transform(X)

# variance expliquée
explained = np.cumsum(pca.explained_variance_ratio_)

# plot variance
plt.figure()
plt.plot(explained)
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.title("PCA Variance")
plt.grid()

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "reports", "pca_variance.png"))
plt.close()

# =========================
# PCA 2D (visualisation rapport)
# =========================
pca_2 = PCA(n_components=2)
X_2d = pca_2.fit_transform(X)

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df[target], alpha=0.5)
plt.title("Projection PCA 2D (Churn)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig(os.path.join(BASE_DIR, "reports", "pca_2d.png"))
plt.close()

# =========================
# KMEANS — ELBOW + SILHOUETTE
# =========================
inertia = []
silhouette = []

K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_2d)

    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_2d, labels))

# Elbow plot
plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.savefig(os.path.join(BASE_DIR, "reports", "kmeans_elbow.png"))
plt.close()

# Silhouette plot
plt.figure()
plt.plot(K_range, silhouette, marker='o')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")

plt.savefig(os.path.join(BASE_DIR, "reports", "kmeans_silhouette.png"))
plt.close()

# =========================
# BEST K
# =========================
best_k = K_range[np.argmax(silhouette)]
print("Best k (silhouette):", best_k)

# =========================
# FINAL KMEANS
# =========================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_2d)

df["Cluster"] = clusters

# =========================
# SAVE MODEL + DATA
# =========================
import joblib

joblib.dump(kmeans, os.path.join(BASE_DIR, "models", "kmeans_model.joblib"))

df.to_csv(os.path.join(BASE_DIR, "data", "processed", "train_clustered.csv"), index=False)

# =========================
# CLUSTER VISUALIZATION
# =========================
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.title(f"KMeans Clusters (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig(os.path.join(BASE_DIR, "reports", "kmeans_clusters.png"))
plt.close()

print("✔ PCA + KMeans terminé")