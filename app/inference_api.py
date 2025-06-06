""" from fastapi import FastAPI
from fastapi.responses import FileResponse
import mlflow
import os

# 🔧 Ajoute cette ligne pour indiquer à MLflow où chercher les artefacts :
mlflow.set_tracking_uri("file:///C:/Users/Lenovo/Desktop/aaa/Application-DeepLearning-IoT-TimeSeries-Analyze-AnomalyDetection-MLOps-Security/mlruns")
app = FastAPI()

# ID du run contenant les artefacts
RUN_ID = "9933adcd4865449fae3a876b50747035"
ARTIFACTS_DIR = "mlflow_downloads"

@app.get("/")
def welcome():
    return {"message": "API FastAPI de visualisation des artefacts MLflow"}

def download_artifact_from_mlflow(artifact_path: str, filename: str):
    dst_dir = os.path.join(ARTIFACTS_DIR, os.path.dirname(artifact_path))
    os.makedirs(dst_dir, exist_ok=True)

    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=RUN_ID,
        artifact_path=artifact_path,
        dst_path=dst_dir
    )

    if os.path.isfile(downloaded_path):
        return downloaded_path

    file_path = os.path.join(downloaded_path, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")

    return file_path

@app.get("/artifact/{file_type}/{filename}")
def get_artifact(file_type: str, filename: str):
    artifact_path = f"{file_type}/{filename}"
    local_file_path = download_artifact_from_mlflow(artifact_path, filename)
    return FileResponse(local_file_path)
 """
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
import os

app = FastAPI()

RUN_ID = "9933adcd4865449fae3a876b50747035"
EXPERIMENT_ID = "793373518782300742"

@app.get("/", response_class=HTMLResponse)
def homepage():
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Résultats Détection Anomalies IoT</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(to bottom right, #eef2f3, #cfd9df);
                padding: 30px;
                margin: 0;
            }
            .container {
                background: white;
                max-width: 700px;
                margin: auto;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #0a3d62;
                margin-bottom: 30px;
            }
            ul {
                list-style: none;
                padding: 0;
            }
            li {
                margin: 15px 0;
                font-size: 18px;
            }
            a {
                text-decoration: none;
                color: #0984e3;
                transition: 0.3s;
            }
            a:hover {
                text-decoration: underline;
                color: #2d3436;
            }
            .logo {
                display: block;
                margin: auto;
                width: 100px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <img class="logo" src="https://cdn-icons-png.flaticon.com/512/2103/2103616.png" alt="IoT Logo" />
            <h1>📊 Résultats - Détection Anomalies IoT</h1>
            <ul>
                <li>📈 <a href="/artifact/figures/barplot_comparaison_models.png" target="_blank">Barplot Comparatif</a></li>
                <li>📊 <a href="/artifact/figures/radar_comparaison_models.png" target="_blank">Radar Comparatif</a></li>
                <li>📝 <a href="/artifact/rapports/resultats_detection_DL_IoT_Modbus.txt" target="_blank">Résultats Deep Learning</a></li>
                <li>📄 <a href="/artifact/rapports/traditional_methods_report.txt" target="_blank">Méthodes Traditionnelles</a></li>
                <li>📋 <a href="/artifact/comparatifs/comparatif_global_models.csv" target="_blank">Comparatif Global (CSV)</a></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/artifact/{folder}/{filename}")
def get_artifact(folder: str, filename: str):
    artifact_path = os.path.join("mlruns", EXPERIMENT_ID, RUN_ID, "artifacts", folder, filename)

    if not os.path.isfile(artifact_path):
        return {"error": f"Fichier non trouvé : {artifact_path}"}

    return FileResponse(artifact_path)

