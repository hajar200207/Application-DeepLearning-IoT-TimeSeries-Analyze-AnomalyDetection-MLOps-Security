import pandas as pd
import numpy as np
import os

# Paramètres
INPUT = "data/processed_data/IoT_Modbus_cleaned.csv"
X_OUT = "data/processed_data/X_windows.npy"
y_OUT = "data/processed_data/y_labels.npy"
WINDOW_SIZE = 30

# Vérification du fichier d'entrée
if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Fichier introuvable : {INPUT}")

# Chargement des données
df = pd.read_csv(INPUT, index_col=0, parse_dates=True)

# Création des fenêtres
X, y = [], []
for i in range(len(df) - WINDOW_SIZE):
    window = df.iloc[i:i + WINDOW_SIZE]
    X.append(window.drop(columns=['label']).values)
    y.append(df.iloc[i + WINDOW_SIZE]['label'])

# Conversion en numpy arrays
X = np.array(X)
y = np.array(y)

# Sauvegarde
os.makedirs("data/processed_data", exist_ok=True)
np.save(X_OUT, X)
np.save(y_OUT, y)

print("4_windowing terminé avec succès.")
