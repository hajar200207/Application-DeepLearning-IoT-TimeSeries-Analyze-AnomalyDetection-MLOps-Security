from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import zipfile

app = FastAPI()

# ------------------------------
# CONFIGURATION
# ------------------------------
RUN_ID = "9933adcd4865449fae3a876b50747035"
EXPERIMENT_ID = "793373518782300742"
ARTIFACT_BASE = os.path.join("mlruns", EXPERIMENT_ID, RUN_ID, "artifacts")

AVAILABLE_MODELS = {
    "lstm": "LSTM.pt",
    "gru": "GRU.pt",
    "cnn1d": "CNN1D.pt",
    "autoencoder": "Autoencoder.pt",
    "transformer": "Transformer.pt",
    "hybrid": "Hybrid.pt"
}

# ------------------------------
# INPUT DATA FORMAT
# ------------------------------
class InputData(BaseModel):
    features: list[list[float]]  # (T, input_size)

# ------------------------------
# MODEL DEFINITIONS
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

class CNN1DModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, in_channels, seq_len]
        return self.net(x)

class AutoencoderModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size())

# Au lieu de l'importation depuis models.py, redéfinis-le ici ou dans models.py :
class SimpleTransformer(nn.Module):
    def __init__(self, input_size, seq_len=30, num_heads=4, num_layers=2, dim_feedforward=2048, num_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(seq_len * input_size, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        return self.classifier(x.view(x.size(0), -1))


class HybridModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# ------------------------------
# LOAD MODEL
# ------------------------------
def load_model(model_name, input_size, seq_len=30):
    path = os.path.join("models", AVAILABLE_MODELS[model_name])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        if model_name == "lstm":
            model = LSTMModel(input_size)
        elif model_name == "gru":
            model = GRUModel(input_size)
        elif model_name == "cnn1d":
            model = CNN1DModel(input_size=input_size)
        elif model_name == "autoencoder":
            model = AutoencoderModel(input_dim=input_size * seq_len)
        elif model_name == "transformer":
            if input_size % 4 != 0:
                raise HTTPException(status_code=500, detail=f"embed_dim ({input_size}) must be divisible by num_heads (4)")
            model = SimpleTransformer(input_size=input_size, seq_len=seq_len)
        elif model_name == "hybrid":
         model = HybridModel(input_size=input_size)  # ✅ Correct, dynamique
 # input_size fixé à 32 comme à l'entraînement

        else:
            raise HTTPException(status_code=400, detail="Unsupported model")

        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de chargement du modèle : {e}")


# === ROUTE PREDICT ===
@app.post("/predict/{model_name}")
def predict(model_name: str, data: InputData):
    model_name = model_name.lower()
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")

    try:
        x_np = np.array(data.features, dtype=np.float32)
        if len(x_np.shape) != 2:
            raise ValueError("Format des données invalide")
        input_size = x_np.shape[1]
        seq_len = x_np.shape[0]
        x_tensor = torch.tensor(x_np).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = load_model(model_name, input_size, seq_len)

    with torch.no_grad():
        output = model(x_tensor)
        if model_name == "autoencoder":
            reconstruction_error = torch.mean((output - x_tensor.view(output.size())) ** 2).item()
            prediction = "anomalie" if reconstruction_error > 0.05 else "normal"
            return {
                "model": model_name,
                "prediction": prediction,
                "reconstruction_error": round(reconstruction_error, 5)
            }
        
        else:
            score = torch.softmax(output, dim=1)[0, 1].item()
            prediction = "anomalie" if score > 0.5 else "normal"
            return {
                "model": model_name,
                "prediction": prediction,
                "score": round(score, 3)
            }
# ------------------------------
# PAGE HTML PRINCIPALE
# ------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Résultats Détection Anomalies IoT</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial; background: #f4f4f4; padding: 20px; }
            h1 { color: #333; }
            ul { line-height: 1.8; padding-left: 0; }
            li { list-style: none; margin-bottom: 10px; }
            a { text-decoration: none; color: #1a73e8; }
            .btn { padding: 10px 15px; background-color: #1a73e8; color: white; border-radius: 5px; }
            footer { margin-top: 40px; font-size: 0.9rem; color: #777; text-align: center; }
        </style>
    </head>
    <body>
        <h1>📊 Résultats - Anomalie Detection IoT</h1>
        <p><strong>Université Sultan Moulay Slimane</strong><br>
        Faculté Polydisciplinaire de Béni Mellal<br>
        <strong>PFE :</strong> Détection d’anomalies dans les séries temporelles IoT<br>
        <strong>Étudiante :</strong> Ouadi Hajar | <strong>Encadrant :</strong> Pr. Biniz</p>

        <ul>
            <li><a href="/artifact/figures/barplot_comparaison_models.png" target="_blank">📈 Barplot Comparatif</a></li>
            <li><a href="/artifact/figures/radar_comparaison_models.png" target="_blank">📊 Radar Comparatif</a></li>
            <li><a href="/artifact/rapports/resultats_detection_DL_IoT_Modbus.txt" target="_blank">📝 Résultats DL (txt)</a></li>
            <li><a href="/artifact/rapports/traditional_methods_report.txt" target="_blank">📝 Méthodes Traditionnelles (txt)</a></li>
            <li><a href="/artifact/comparatifs/comparatif_global_models.csv" target="_blank">📄 Comparatif Global (CSV)</a></li>
        </ul>
        <li><a href="http://localhost:5000/#/experiments/697220615460650424" target="_blank">🔬 Voir Expérience MLflow</a></li>
        <li><a href="http://localhost:5000/#/experiments/697220615460650424/runs/2a5dc03aae304d808632bf93033b9bde/model-metrics" target="_blank">📉 Métriques modèle</a></li>

        <a href="/download/all" class="btn">📦 Télécharger tous les résultats (.zip)</a>
        <footer>&copy; 2025 - PFE Ouadi Hajar</footer>
    </body>
    </html>
    """

# ------------------------------
# TÉLÉCHARGEMENT DE FICHIERS
# ------------------------------
@app.get("/artifact/{folder}/{filename}")
def get_artifact(folder: str, filename: str):
    path = os.path.join(ARTIFACT_BASE, folder, filename)
    if not os.path.isfile(path):
        return {"error": f"Fichier non trouvé : {path}"}
    return FileResponse(path)

@app.get("/download/all")
def download_all():
    zip_filename = "all_results.zip"
    zip_path = os.path.join("mlruns", zip_filename)
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(ARTIFACT_BASE):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, ARTIFACT_BASE)
                zipf.write(full_path, arcname)
    return FileResponse(zip_path, filename=zip_filename, media_type="application/zip")

# ------------------------------
# API INFÉRENCE TEMPS RÉEL
# ------------------------------
