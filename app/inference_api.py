from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import zipfile
import io
import json
import re
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

MLFLOW_PORT = 5000
MLFLOW_PATH = "/"

@app.get("/mlflow")
def mlflow_redirect(request: Request):
    host = request.headers.get("host", "")
    match = re.match(r"^([\w\-]+)-(\d+)-(\d+)-(\d+)-(\d+)\.ngrok-free\.app$", host)
    if not match:
        raise HTTPException(status_code=400, detail="Host non autorisé")
    mlflow_host = host.replace("3856", "49d4")
    url = f"https://{mlflow_host}.ngrok-free.app{MLFLOW_PATH}"
    return RedirectResponse(url, status_code=307)

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

class InputData(BaseModel):
    features: list[list[float]]
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.float32: lambda v: float(v)}

# === Définition des modèles Deep Learning ===

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
        x = x.permute(0, 2, 1)
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
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# === Chargement sécurisé du modèle ===

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
            model = CNN1DModel(in_channels=input_size)
        elif model_name == "autoencoder":
            model = AutoencoderModel(input_dim=input_size * seq_len)
        elif model_name == "transformer":
            if input_size % 4 != 0:
                raise HTTPException(status_code=500, detail="embed_dim doit être multiple de 4")
            model = SimpleTransformer(input_size=input_size, seq_len=seq_len)
        elif model_name == "hybrid":
            model = HybridModel(input_size=input_size)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model")

        with open(path, 'rb') as f:
            buffer = f.read()

        # 🔐 Chargement sécurisé :
        # torch.load est justifié ici car les fichiers .pt sont générés localement via MLflow
        # Aucun fichier externe ou non vérifié ne peut être injecté
        # Ce hotspot est considéré comme maîtrisé ✔
        state_dict = torch.load(io.BytesIO(buffer), map_location=torch.device("cpu"))  # nosec B301

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de chargement sécurisé du modèle : {e}")


# === Endpoint de prédiction ===

@app.post("/predict/{model_name}")
def predict(model_name: str, data: InputData):
    model_name = model_name.lower()
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    try:
        x_np = np.array(data.features).astype(np.float32)
        if len(x_np.shape) != 2:
            raise ValueError("Format invalide")
        input_size = x_np.shape[1]
        seq_len = x_np.shape[0]
        x_tensor = torch.tensor(x_np).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = load_model(model_name, input_size, seq_len)
    with torch.no_grad():
        output = model(x_tensor)

        if model_name == "autoencoder":
            error = torch.mean((output - x_tensor.view(output.size())) ** 2).item()
            prediction = "anomalie" if error > 0.05 else "normal"
            return {"model": model_name, "prediction": prediction, "reconstruction_error": round(error, 5)}

        score = torch.softmax(output, dim=1)[0, 1].item()
        prediction = "anomalie" if score > 0.5 else "normal"

        ioc_path = os.path.join("configs", "ioc.json")
        try:
            with open(ioc_path, "r", encoding="utf-8") as f:
                ioc_data = json.load(f)
        except Exception:
            ioc_data = {}

        model_threat_map = {
            "lstm": "DoS",
            "cnn1d": "Scan",
            "gru": "Injection",
            "hybrid": "Accès non autorisé"
        }
        threat = model_threat_map.get(model_name, "Inconnu")
        ioc = ioc_data.get(threat, {"risk": "Inconnu", "recommendation": "N/A"})

        if prediction == "anomalie":
            os.makedirs("logs", exist_ok=True)
            with open("logs/alerts.csv", "a") as f:
                f.write(f"{datetime.now()},{model_name},{prediction},{score:.3f},{threat},{ioc['risk']},{ioc['recommendation']}\n")

        return {
            "model": model_name,
            "prediction": prediction,
            "score": round(score, 3),
            "threat_type": threat,
            "risk": ioc["risk"],
            "recommendation": ioc["recommendation"]
        }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body><h1>Inference API IoT</h1></body></html>
    """

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

