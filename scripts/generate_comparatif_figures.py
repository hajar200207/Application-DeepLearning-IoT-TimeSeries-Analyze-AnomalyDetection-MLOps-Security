import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
import os
import mlflow

# Définir les chemins
csv_path = "results/comparatifs/comparatif_global_models.csv"
output_dir = "results/comparatifs/figures"
os.makedirs(output_dir, exist_ok=True)

# Colonnes d'indicateurs
categories = ["Accuracy", "AUC-ROC", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]

# Charger le fichier comparatif
df = pd.read_csv(csv_path)

# Démarrer une expérience MLflow
mlflow.set_experiment("AnomalyDetection_Comparatif_Figures")
with mlflow.start_run(run_name="Figures_Comparatives"):

    # === 1. GRAPHIQUE À BARRES ===
    df_bars = df.melt(id_vars=["Modèle", "Type"], value_vars=categories,
                      var_name="Indicateur", value_name="Score")

    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_bars, x="Modèle", y="Score", hue="Indicateur")
    plt.title("Comparaison des indicateurs de performance par modèle", fontsize=16)
    plt.ylabel("Score (%)")
    plt.xticks(rotation=45)
    plt.legend(title="Indicateur", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    barplot_path = os.path.join(output_dir, "barplot_comparaison_models.png")
    plt.savefig(barplot_path)
    plt.close()
    mlflow.log_artifact(barplot_path, artifact_path="figures")

    # === 2. GRAPHIQUE RADAR ===
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)

    for i, row in df.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Modèle"])
        ax.fill(angles, values, alpha=0.08)

    plt.xticks(angles[:-1], categories, color='black', size=12)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
    plt.ylim(0, 100)
    plt.title("Comparaison radar de tous les modèles (DL & traditionnels)", size=14, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1), fontsize=9)
    plt.tight_layout()

    radarplot_path = os.path.join(output_dir, "radar_comparaison_models.png")
    plt.savefig(radarplot_path)
    plt.close()
    mlflow.log_artifact(radarplot_path, artifact_path="figures")

print("✅ Figures générées et loggées dans MLflow.")
