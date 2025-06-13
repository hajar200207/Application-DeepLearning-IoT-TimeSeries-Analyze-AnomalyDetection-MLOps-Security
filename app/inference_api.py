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

# === Chargement sécurisé via TorchScript ===
def load_model(model_name: str):
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
        x_tensor = torch.tensor(x_np).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = load_model(model_name)
    with torch.no_grad():
        output = model(x_tensor)
        score = torch.softmax(output, dim=1)[0, 1].item()
        prediction = "anomalie" if score > 0.5 else "normal"
        return {
            "model": model_name,
            "prediction": prediction,
            "score": round(score, 3)
        }

@app.get("/", response_class=HTMLResponse)
def home():
    return "<html><body><h1>Inference API IoT</h1></body></html>"

# === Endpoint sécurisé pour récupérer un fichier d’artefact ===
@app.get("/artifact/{folder}/{filename}")
def get_artifact(folder: str, filename: str):
    if ".." in folder or ".." in filename or "/" in folder or "/" in filename:
        raise HTTPException(status_code=400, detail="Nom de fichier ou dossier invalide")

    safe_folder = os.path.basename(folder)
    safe_filename = os.path.basename(filename)
    path = os.path.join(ARTIFACT_BASE, safe_folder, safe_filename)

    if not os.path.isfile(path):
        return {"error": f"Fichier non trouvé : {safe_filename}"}
    return FileResponse(path)

# === Endpoint de téléchargement complet des artefacts ===
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
