import numpy as np
import pandas as pd
import os

# Chemins
X_path = "data/processed_data/X_windows_balanced.npy"
y_path = "data/processed_data/y_labels_balanced.npy"

# Chargement
X = np.load(X_path)
y = np.load(y_path)

print(f" X shape: {X.shape}")  # (n_samples, window_size, n_features)
print(f" y shape: {y.shape}")
print(" Nombre d'exemples par classe :", pd.Series(y).value_counts())

# Optionnel : Sauvegarder une version aplatie en CSV
X_flat = X.reshape(X.shape[0], -1)
df_flat = pd.DataFrame(X_flat, columns=[f"f{i}" for i in range(X_flat.shape[1])])
df_flat["label"] = y

os.makedirs("data/review", exist_ok=True)
df_flat.to_csv("data/review/balanced_dataset_for_review.csv", index=False)
print(" Sauvegardé : balanced_dataset_for_review.csv")
