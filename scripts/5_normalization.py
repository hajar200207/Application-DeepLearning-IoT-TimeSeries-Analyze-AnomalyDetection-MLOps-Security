import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

X = np.load("data/processed_data/X_windows.npy")

scaler = MinMaxScaler()

# Normaliser chaque fenêtre indépendamment
#Application d’une normalisation MinMax sur les données pour les mettre à l’échelle.
X_scaled = X.copy()
for i in range(X.shape[0]):
    X_scaled[i] = scaler.fit_transform(X[i])

np.save("data/processed_data/X_windows_normalized.npy", X_scaled)
print(" 5_normalization terminé")
