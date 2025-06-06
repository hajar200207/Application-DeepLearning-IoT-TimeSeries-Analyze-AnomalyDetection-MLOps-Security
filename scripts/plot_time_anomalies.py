import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats déjà enrichis
df = pd.read_csv("data/processed_data/detection_with_predictions.csv")

# Créer un index temporel fictif si absent
df["index"] = df.index

plt.figure(figsize=(15, 6))

# 1. Annotations de l'attaque réelle
plt.plot(df["index"], df["label"], label="Label Réel (Attaque)", color="black", linewidth=1)

# 2. Anomalies Isolation Forest
plt.scatter(df[df["iso_pred"] == 1]["index"], [1.2]*len(df[df["iso_pred"] == 1]), 
            color="red", label="Anomalie - Isolation Forest", marker="x", s=10)

# 3. Anomalies Z-Score
plt.scatter(df[df["zscore_pred"] == 1]["index"], [1.4]*len(df[df["zscore_pred"] == 1]), 
            color="green", label="Anomalie - Z-Score", marker="^", s=10)

plt.title(" Détection d'Anomalies sur les Séries Temporelles IoT")
plt.xlabel("Index (ordre temporel)")
plt.ylabel("Label / Anomalie")
plt.yticks([0, 1, 1.2, 1.4], ["Normal", "Attaque réelle", "IsoForest", "Z-Score"])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("data/processed_data/time_series_anomaly_plot.png")
plt.show()
