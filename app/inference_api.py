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
from fastapi.responses import FileResponse
import os

app = FastAPI()

# ID du run dans MLflow
RUN_ID = "9933adcd4865449fae3a876b50747035"

@app.get("/")
def welcome():
    return {"message": "API FastAPI de visualisation des artefacts MLflow"}

@app.get("/artifact/{folder}/{filename}")
def get_artifact(folder: str, filename: str):
    # Chemin vers l'artefact dans le volume Docker
    artifact_path = os.path.join("mlruns", "793373518782300742", RUN_ID, "artifacts", folder, filename)

    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(f"Fichier non trouvé : {artifact_path}")

    return FileResponse(artifact_path)
