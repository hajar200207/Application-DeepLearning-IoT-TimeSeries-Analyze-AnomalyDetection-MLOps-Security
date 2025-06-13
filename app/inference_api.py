from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
import torch
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
    "lstm": "LSTM_jit.pt",
    "gru": "GRU_jit.pt",
    "cnn1d": "CNN1D_jit.pt",
    "autoencoder": "Autoencoder_jit.pt",
    "transformer": "Transformer_jit.pt",
    "hybrid": "Hybrid_jit.pt"
}

class InputData(BaseModel):
    features: list[list[float]]
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.float32: lambda v: float(v)}

# === Chargement sécurisé des modèles via TorchScript ===
def load_model(model_name, input_size, seq_len=30):
    path = os.path.join("models", AVAILABLE_MODELS[model_name])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        model = torch.jit.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de chargement JIT : {e}")

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
            return {
                "model": model_name,
                "prediction": prediction,
                "reconstruction_error": round(error, 5)
            }

        score = torch.softmax(output, dim=1)[0, 1].item()
        prediction = "anomalie" if score > 0.5 else "normal"

        # Threat info
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

        # Log si anomalie
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
    return "<html><body><h1>Inference API IoT</h1></body></html>"

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
