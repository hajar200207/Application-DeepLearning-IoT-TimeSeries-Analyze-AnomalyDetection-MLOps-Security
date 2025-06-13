import sys
import os

# ✅ Ajouter le dossier parent au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.lstm_model import LSTMModel

# 📁 Corriger les chemins relatifs
model_path = os.path.join("..", "models", "LSTM.pt")
X_path = os.path.join("..", "data", "processed_data", "X_windows.npy")
y_path = os.path.join("..", "data", "processed_data", "y_labels.npy")

# 🔄 Charger les données
X = np.load(X_path)
y_true = np.load(y_path)

# 🔁 Recréer le modèle
model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=1)

# ✅ Chargement sécurisé des poids
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# 🔄 Conversion en tenseur
X_tensor = torch.tensor(X, dtype=torch.float32)

# 🔮 Prédiction
with torch.no_grad():
    outputs = model(X_tensor)
    predictions = torch.argmax(outputs, dim=1).numpy()

# 📊 Rapport de classification
print("📄 Classification Report:")
print(classification_report(y_true, predictions))

# 📉 Matrice de confusion
conf_matrix = confusion_matrix(y_true, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion - LSTM')
plt.xlabel('Prédit')
plt.ylabel('Réel')

# 📁 Sauvegarde de la figure
output_path = os.path.join("..", "data", "processed_data", "confusion_matrix_lstm.png")
plt.savefig(output_path)
plt.show()
