import torch
import os

# Liste des fichiers de modèles
model_paths = [
    "models/Autoencoder.pt",
    "models/CNN1D.pt",
    "models/GRU.pt",
    "models/Hybrid.pt",
    "models/LSTM.pt",
    "models/LSTM_AE.pt",
    "models/Transformer.pt"
]

output_path = "logs/evaluation/models_weights_dump.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for path in model_paths:
        f.write(f"\n--- {path} ---\n")
        try:
            model = torch.load(path, map_location='cpu')
            f.write(" Modèle chargé avec succès.\n Contenu du modèle :\n")
            if isinstance(model, dict) or isinstance(model, torch.nn.Module):
                state_dict = model if isinstance(model, dict) else model.state_dict()
                f.write(str(state_dict))
            else:
                f.write(str(model))
        except Exception as e:
            f.write(f"❌ Erreur de chargement : {e}\n")

print(f"✅ Résultats enregistrés dans {output_path}")
