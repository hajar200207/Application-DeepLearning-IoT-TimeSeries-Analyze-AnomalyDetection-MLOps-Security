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
from fastapi.responses import FileResponse, HTMLResponse
import os
import zipfile

app = FastAPI()

RUN_ID = "9933adcd4865449fae3a876b50747035"
EXPERIMENT_ID = "793373518782300742"
ARTIFACT_BASE = os.path.join("mlruns", EXPERIMENT_ID, RUN_ID, "artifacts")

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
            body {
                font-family: Arial, sans-serif;
                background: #f4f4f4;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            ul {
                line-height: 1.8;
                padding-left: 0;
            }
            li {
                list-style: none;
                margin-bottom: 10px;
            }
            a {
                text-decoration: none;
                color: #1a73e8;
            }
            .btn {
                display: inline-block;
                padding: 10px 15px;
                margin-top: 20px;
                background-color: #1a73e8;
                color: white;
                border-radius: 5px;
            }
            footer {
                margin-top: 40px;
                font-size: 0.9rem;
                color: #777;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <h1>📊 Résultats - Anomalie Detection IoT</h1>
        <p><strong>Université Sultan Moulay Slimane</strong><br>
        Faculté Polydisciplinaire de Béni Mellal<br>
        Département Informatique<br>
        <strong>Projet de Fin d’Études (PFE)</strong><br>
        Détection d’anomalies basée sur les séries temporelles<br>
        Application du Deep Learning pour l’analyse des données IoT<br>
        <strong>Encadrant :</strong> Pr. Biniz<br>
        <strong>Réalisé par :</strong> Ouadi HAJAR</p>
        
        <ul>
            <li><a href="/artifact/figures/barplot_comparaison_models.png" target="_blank">📈 Barplot Comparatif</a></li>
            <li><a href="/artifact/figures/radar_comparaison_models.png" target="_blank">📊 Radar Comparatif</a></li>
            <li><a href="/artifact/rapports/resultats_detection_DL_IoT_Modbus.txt" target="_blank">📝 Résultats DL (txt)</a></li>
            <li><a href="/artifact/rapports/traditional_methods_report.txt" target="_blank">📝 Méthodes Traditionnelles (txt)</a></li>
            <li><a href="/artifact/comparatifs/comparatif_global_models.csv" target="_blank">📄 Comparatif Global (CSV)</a></li>
        </ul>
<li>
  <a href="http://localhost:5000/#/experiments/697220615460650424" target="_blank">
    🔬 Voir Expérience MLflow (locale)
  </a>
</li>
<li>
  <a href="http://localhost:5000/#/experiments/697220615460650424/runs/2a5dc03aae304d808632bf93033b9bde/model-metrics" target="_blank">
    📉 Traces et métriques du modèle
  </a>
</li>

        <a href="/download/all" class="btn">📦 Télécharger tous les résultats (.zip)</a>

        <footer>
            &copy; 2025 - Projet de Fin d'Études de Ouadi Hajar
        </footer>
    </body>
    </html>
    """

@app.get("/artifact/{folder}/{filename}")
def get_artifact(folder: str, filename: str):
    artifact_path = os.path.join(ARTIFACT_BASE, folder, filename)
    if not os.path.isfile(artifact_path):
        return {"error": f"Fichier non trouvé : {artifact_path}"}
    return FileResponse(artifact_path)

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
