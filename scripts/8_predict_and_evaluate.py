import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Chemins vers les fichiers
model_path = "models/LSTM_jit.pt"
X_path = "data/processed_data/X_windows.npy"
y_path = "data/processed_data/y_labels.npy"

# 🔄 Charger les données
X = np.load(X_path)
y_true = np.load(y_path)

# ✅ Charger modèle TorchScript (JIT)
model = torch.jit.load(model_path, map_location=torch.device("cpu"))
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
plt.title('Matrice de confusion - LSTM JIT')
plt.xlabel('Prédit')
plt.ylabel('Réel')

# 📁 Sauvegarde
os.makedirs("data/processed_data", exist_ok=True)
plt.savefig("data/processed_data/confusion_matrix_lstm.png")
plt.show()
