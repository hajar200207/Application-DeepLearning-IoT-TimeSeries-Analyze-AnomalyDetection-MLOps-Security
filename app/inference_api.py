from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import zipfile
from prometheus_client import start_http_server, Summary, Counter
import time
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI
from fastapi import Request

app = FastAPI()

# Initialiser Prometheus après l’instanciation de app
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

from fastapi.responses import RedirectResponse

MLFLOW_PORT = 5000          # si tu gardes MLflow sur 5000
MLFLOW_PATH = "/"           # ou "/#/experiments/…"

import re

@app.get("/mlflow")
def mlflow_redirect(request: Request):
    """
    Redirige vers MLflow via Ngrok avec validation de sécurité.
    """
    host = request.headers.get("host", "")
    
    # ✅ Sécurisation : autoriser seulement certains formats
    match = re.match(r"^([\w\-]+)-(\d+)-(\d+)-(\d+)-(\d+)\.ngrok-free\.app$", host)
    if not match:
        raise HTTPException(status_code=400, detail="Host non autorisé")

    # Extraction et remplacement contrôlé
    mlflow_host = host.replace("3856", "49d4")  # ou extraire dynamiquement

    url = f"https://{mlflow_host}.ngrok-free.app{MLFLOW_PATH}"
    return RedirectResponse(url, status_code=307)


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
    features: list[list[float]]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.float32: lambda v: float(v)
        }

# (T, input_size)

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
            model = CNN1DModel(in_channels=input_size)

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
    import json
    from datetime import datetime

    model_name = model_name.lower()
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")

    try:
        # Conversion stricte en float32
        x_np = np.array(data.features).astype(np.float32)
        if len(x_np.shape) != 2:
            raise ValueError("Format des données invalide : attendu [T, input_size]")
        
        input_size = x_np.shape[1]
        seq_len = x_np.shape[0]
        x_tensor = torch.tensor(x_np).unsqueeze(0)  # Ajoute batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de traitement des données : {str(e)}")

    model = load_model(model_name, input_size, seq_len)
    model.eval()

    with torch.no_grad():
        output = model(x_tensor)

        # === Cas spécial autoencoder ===
        if model_name == "autoencoder":
            reconstruction_error = torch.mean((output - x_tensor.view(output.size())) ** 2).item()
            prediction = "anomalie" if reconstruction_error > 0.05 else "normal"
            return {
                "model": model_name,
                "prediction": prediction,
                "reconstruction_error": round(float(reconstruction_error), 5)
            }

        # === Cas général : classification (score softmax) ===
        score = torch.softmax(output, dim=1)[0, 1].item()
        prediction = "anomalie" if score > 0.5 else "normal"
        # === Partie IOC ===
        ioc_path = os.path.join("configs", "ioc.json")
        try:
            with open(ioc_path, "r", encoding="utf-8") as f:
                ioc_data = json.load(f)
        except:
            ioc_data = {}

        model_threat_map = {
            "lstm": "DoS",
            "cnn1d": "Scan",
            "gru": "Injection",
            "hybrid": "Accès non autorisé"
        }
        threat = model_threat_map.get(model_name, "Inconnu")
        ioc = ioc_data.get(threat, {"risk": "Inconnu", "recommendation": "N/A"})

        # === Log uniquement en cas d'anomalie détectée ===
        if prediction == "anomalie":
            os.makedirs("logs", exist_ok=True)
            with open("logs/alerts.csv", "a") as f:
                f.write(f"{datetime.now()},{model_name},{prediction},{score:.3f},{threat},{ioc['risk']},{ioc['recommendation']}\n")

        # === Résultat final (avec float pur) ===
        return {
            "model": model_name,
            "prediction": prediction,
            "score": round(float(score), 3),
            "threat_type": threat,
            "risk": ioc["risk"],
            "recommendation": ioc["recommendation"]
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
            <li><a href="/test" target="_blank">🧪 Test en ligne par l'encadrant</a></li>
        </ul>
        <li><a href="/mlflow" target="_blank">🔬 Voir MLflow (auto)</a></li>

        <li><a href="https://49d4-196-200-159-146.ngrok-free.app/#/experiments/697220615460650424/runs/2a5dc03aae304d808632bf93033b9bde/model-metrics" target="_blank">📉 Métriques modèle</a></li>

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
@app.get("/test", response_class=HTMLResponse)
def test_ui():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Tester un modèle DL (IoT Anomaly Detection)</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f6fa;
                margin: 0;
                padding: 40px;
                color: #333;
            }

            .container {
                max-width: 900px;
                margin: auto;
                background-color: #fff;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            }

            h2 {
                text-align: center;
                color: #2c3e50;
                margin-bottom: 20px;
            }

            label {
                font-weight: bold;
                display: block;
                margin-top: 20px;
                margin-bottom: 5px;
            }

            select, textarea, button {
                width: 100%;
                padding: 12px;
                border-radius: 5px;
                border: 1px solid #ccc;
                font-size: 14px;
            }

            textarea {
                resize: vertical;
            }

            button {
                margin-top: 20px;
                background-color: #3498db;
                color: white;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s;
            }

            button:hover {
                background-color: #2980b9;
            }

            #result {
                background-color: #ecf0f1;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin-top: 20px;
                white-space: pre-wrap;
                word-break: break-word;
            }
            ul li {
    display: inline-block;
    margin-right: 10px;
}

        </style>
    </head>
    <body>
        <div class="container">
            <h2>Test d'inférence en ligne (Détection d'anomalies IoT)</h2>
            <form id="predictForm">
                <label for="model">Choisir un modèle :</label>
                <select id="model">
                    <option value="lstm">LSTM</option>
                    <option value="gru">GRU</option>
                    <option value="cnn1d">CNN1D</option>
                    <option value="autoencoder">Autoencoder</option>
                    <option value="transformer">Transformer</option>
                    <option value="hybrid">Hybrid</option>
                </select>

                <label for="features">Données (JSON - tableau de vecteurs) :</label>
                <textarea id="features" rows="10">
 [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0]
  ]

                </textarea>

               <ul>
  <li><button type="submit">🚀 Lancer la prédiction</button></li>
  <li><button type="button" onclick="loadSampleData()">📥 Charger exemple dataset</button></li>
  <li><button type="button" onclick="window.location.href='/alerts'">📜 Voir alertes</button></li>
</ul>


            </form>

            <h3>Résultat :</h3>
            <pre id="result">En attente de résultat...</pre>
        </div>

        <script>
            document.getElementById("predictForm").onsubmit = async (e) => {
                e.preventDefault();
                const model = document.getElementById("model").value;
let features;
try {
    features = JSON.parse(document.getElementById("features").value);
} catch (err) {
    document.getElementById("result").textContent = "❌ Erreur de parsing JSON : " + err.message;
    return;
}

                const response = await fetch(`/predict/${model}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ features })
                });

                const data = await response.json();
                document.getElementById("result").textContent = JSON.stringify(data, null, 2);
            };
            function loadSampleData() {
    // Exemple de données simulées : 30 étapes temporelles, 4 capteurs
    const sample = [];
    for (let i = 0; i < 30; i++) {
        sample.push([
            Math.sin(i * 0.2),
            Math.cos(i * 0.2),
            0.5 + Math.random() * 0.1,
            1.0 + Math.random() * 0.1
        ]);
    }
    document.getElementById("features").value = JSON.stringify(sample, null, 2);
}

        </script>
    </body>
    </html>
    """

@app.get("/alerts", response_class=HTMLResponse)
def show_alerts():
    import csv
    from datetime import datetime

    alerts_path = "logs/alerts.csv"
    alerts_html = ""

    if not os.path.exists(alerts_path):
        alerts_html = "<p>Aucune alerte enregistrée pour le moment.</p>"
    else:
        with open(alerts_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)[-20:]  # Show last 20 alerts

        for row in rows[::-1]:  # reverse pour latest en haut
            timestamp, model, prediction, score, threat, risk, recommendation = row
            date_formatted = datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S").strftime("%d-%m-%Y %H:%M")
            alerts_html += f"""
            <li>📌 [{date_formatted}] Modèle: <strong>{model.upper()}</strong> | ⚠️ <strong>{threat}</strong> |
            🔥 Score: <code>{score}</code> | 🚨 Risque: <strong>{risk}</strong> | ✔️ Action: <em>{recommendation}</em></li>
            """

    return f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Alertes de sécurité IoT</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, sans-serif;
                background-color: #f9f9f9;
                padding: 30px;
                color: #333;
            }}
            h2 {{
                text-align: center;
                color: #c0392b;
            }}
            ul {{
                list-style: none;
                padding: 0;
            }}
            li {{
                background: #fff;
                margin-bottom: 15px;
                padding: 15px;
                border-left: 6px solid #e74c3c;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                border-radius: 6px;
            }}
            code {{
                background: #eee;
                padding: 2px 5px;
                border-radius: 4px;
            }}
            em {{
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <h2>🔐 Journal des Alertes Détection Anomalies IoT</h2>
        <ul>
            {alerts_html}
        </ul>
    </body>
    </html>
    """


REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNTER = Counter('http_requests_total', 'Total HTTP Requests')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQUEST_COUNTER.inc()
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_TIME.observe(duration)
    return response

# démarrer serveur Prometheus (en plus d'uvicorn)
start_http_server(8001)
