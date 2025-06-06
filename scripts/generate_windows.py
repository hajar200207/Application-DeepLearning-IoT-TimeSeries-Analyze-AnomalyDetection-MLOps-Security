import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Correction robuste du chemin
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "data", "processed_data", "IoT_Modbus_cleaned.csv")
df = pd.read_csv(csv_path)

# Traitement
df = df.select_dtypes(include=[np.number])
y = df['label'].values
X = df.drop(columns=['label']).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

window_size = 10
X_windows = []
y_labels = []

for i in range(len(X_scaled) - window_size):
    X_windows.append(X_scaled[i:i+window_size])
    y_labels.append(y[i + window_size])

X_windows = np.array(X_windows)
y_labels = np.array(y_labels)

# Sauvegarde
os.makedirs(os.path.join(base_dir, "..", "data", "processed_data"), exist_ok=True)
np.save(os.path.join(base_dir, "..", "data", "processed_data", "X_windows.npy"), X_windows)
np.save(os.path.join(base_dir, "..", "data", "processed_data", "y_labels.npy"), y_labels)

print("✅ Fichiers sauvegardés : X_windows.npy & y_labels.npy")
