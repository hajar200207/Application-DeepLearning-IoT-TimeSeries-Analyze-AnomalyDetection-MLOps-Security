import pandas as pd
import os

# Créer les données de comparaison
data = [
    # Modèle, Type, Accuracy, AUC-ROC, Precision, Recall, F1
    ["LSTM", "Deep Learning", 98.16, 99.77, 98.16, 98.16, 98.16],
    ["Hybrid CNN+LSTM", "Deep Learning", 97.95, 99.85, 98.00, 97.95, 97.95],
    ["GRU", "Deep Learning", 97.63, 99.70, 97.64, 97.63, 97.63],
    ["CNN1D", "Deep Learning", 76.62, 86.93, 76.89, 76.62, 76.56],
    ["Transformer", "Deep Learning", 52.63, 52.40, 53.26, 52.63, 50.26],
    ["Autoencoder", "Deep Learning", 50.00, 23.84, 25.00, 50.00, 33.33],
    ["Random Forest", "Traditionnel", 100.00, 100.00, 100.00, 100.00, 100.00],
    ["PCA", "Traditionnel", 49.32, 49.32, 49.32, 49.32, 49.32],
    ["DBSCAN", "Traditionnel", 48.08, 48.08, 24.51, 48.08, 32.47],
    ["Isolation Forest", "Traditionnel", 33.36, 33.36, 33.36, 33.36, 33.36],
    ["OneClass SVM", "Traditionnel", 32.03, 32.03, 32.03, 32.03, 32.03],
    ["LocalOutlierFactor", "Traditionnel", 26.79, 26.79, 26.79, 26.79, 26.79],
]

columns = ["Modèle", "Type", "Accuracy", "AUC-ROC", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
df = pd.DataFrame(data, columns=columns)

# Création du dossier de destination
save_dir = os.path.join("results", "comparatifs")
os.makedirs(save_dir, exist_ok=True)

# Sauvegarde du fichier CSV
csv_path = os.path.join(save_dir, "comparatif_global_models.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Fichier enregistré dans : {csv_path}")
