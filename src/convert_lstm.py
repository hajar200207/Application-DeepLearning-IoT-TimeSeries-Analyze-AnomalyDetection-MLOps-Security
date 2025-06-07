import torch
import numpy as np
from src.models.lstm_model import LSTMModel  # ou adapte selon ton projet

# Charger les données juste pour connaître l'input_size
X = np.load("scripts/data/processed_data/X_windows_balanced.npy")

# Créer le modèle vide
model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=1)

# Charger les poids
state_dict = torch.load("models/LSTM.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

# Sauvegarde du modèle complet
torch.save(model, "models/LSTM.pt")
print("✅ Modèle complet sauvegardé avec succès.")
