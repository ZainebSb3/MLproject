import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/retail_customers.csv")  # adapte si besoin

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())

# =========================
# TARGET DISTRIBUTION
# =========================
plt.figure()
df["Churn"].value_counts().plot(kind="bar")
plt.title("Distribution Churn")
plt.show()

# =========================
# MISSING VALUES
# =========================
missing = df.isnull().sum().sort_values(ascending=False)
print("\nMissing values:\n", missing[missing > 0])

# =========================
# CORRELATION MATRIX (IMPORTANT CAHIER DES CHARGES)
# =========================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# =========================
# TOP FEATURES VS CHURN
# =========================
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols[:6]:
    plt.figure()
    sns.boxplot(x=df["Churn"], y=df[col])
    plt.title(f"{col} vs Churn")
    plt.show()

# =========================
# OUTLIERS SIMPLE CHECK
# =========================
print("\nOutlier summary:")
print(df.describe())