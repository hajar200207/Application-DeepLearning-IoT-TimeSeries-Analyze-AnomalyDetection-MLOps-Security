import pandas as pd
import numpy as np
import os

# Chargement des données .npy
X = np.load("data/processed_data/X_windows.npy", allow_pickle=True)
y = np.load("data/processed_data/y_labels.npy", allow_pickle=True)

# On a besoin de savoir combien de features dans chaque time-step
n_features = X.shape[2]
n_samples = X.shape[0]

# Création d’un DataFrame aplati pour visualisation (pas pour entraînement)
data_flat = pd.DataFrame({
    f"f{i}" : [window[:, i].flatten().tolist() for window in X] for i in range(n_features)
})
data_flat["label"] = y

# Sauvegarde
os.makedirs("data/processed_data", exist_ok=True)
output_path = "data/processed_data/final_dataset_for_review.csv"
data_flat.to_csv(output_path, index=False)
print(f"Dataset final sauvegardé sous : {output_path}")
