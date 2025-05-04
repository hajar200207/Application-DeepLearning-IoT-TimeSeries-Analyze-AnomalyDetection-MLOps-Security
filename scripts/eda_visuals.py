
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/processed_data/IoT_Modbus_cleaned.csv", index_col=0)

# Histogramme des labels
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Distribution des classes")
plt.xlabel("Classe")
plt.ylabel("Nombre d'occurrences")
plt.tight_layout()
plt.savefig("logs/label_distribution.png")

# Heatmap des corrélations
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("logs/heatmap_correlation.png")
