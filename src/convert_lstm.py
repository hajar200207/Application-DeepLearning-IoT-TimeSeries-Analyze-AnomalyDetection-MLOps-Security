import torch
import numpy as np
from src.models.lstm_model import LSTMModel

# ✅ Exemple fictif de forme des données (remplace par tes vraies données si besoin)
X = np.load("data/processed_data/X_windows.npy")

# Créer le modèle vide
model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=1)

# Charger les poids (poids uniquement, pas un modèle complet !)
state_dict = torch.load("models/LSTM.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

# Convertir en TorchScript pour usage sécurisé
scripted_model = torch.jit.script(model)

# Sauvegarder en version JIT
torch.jit.save(scripted_model, "models/LSTM_jit.pt")
print("✅ Modèle TorchScript sauvegardé avec succès.")
