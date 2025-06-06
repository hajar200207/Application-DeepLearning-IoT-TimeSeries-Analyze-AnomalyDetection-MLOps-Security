import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Chargement des données
X = np.load("data/processed_data/X_windows.npy", allow_pickle=True)
y = np.load("data/processed_data/y_labels.npy", allow_pickle=True)

# Aplatir les séquences pour SMOTE (2D)
X_flat = X.reshape((X.shape[0], -1))

# Appliquer StandardScaler (z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# Sauvegarde des statistiques
max_val = np.max(X_scaled)
min_val = np.min(X_scaled)
mean_val = np.mean(X_scaled)
std_val = np.std(X_scaled)

# Rééquilibrage avec SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Remise en forme 3D pour LSTM
timesteps = X.shape[1]
features = X.shape[2]
X_resampled_3d = X_resampled.reshape((-1, timesteps, features))

# Sauvegarde des fichiers
os.makedirs("data/processed_data", exist_ok=True)
np.save("data/processed_data/X_windows_balanced.npy", X_resampled_3d)
np.save("data/processed_data/y_labels_balanced.npy", y_resampled)

# Sauvegarde des stats
with open("data/processed_data/zscore_stats.txt", "w") as f:
    f.write(f"Z-Score Normalization Stats:\n")
    f.write(f"Mean: {mean_val}\n")
    f.write(f"Std: {std_val}\n")
    f.write(f"Max: {max_val}\n")
    f.write(f"Min: {min_val}\n")
