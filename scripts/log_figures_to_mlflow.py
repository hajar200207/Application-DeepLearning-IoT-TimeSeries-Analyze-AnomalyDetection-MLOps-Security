import mlflow
import os

# Dossier des figures
figures_dir = "results/comparatifs/figures"

# Démarrer une expérience MLflow dédiée
mlflow.set_experiment("AnomalyDetection_Comparatif_Figures")

with mlflow.start_run(run_name="Figures_Comparatives"):
    mlflow.log_param("Type", "Figures de comparaison")

    # Lister les fichiers .png dans le dossier
    for filename in os.listdir(figures_dir):
        if filename.endswith(".png"):
            fig_path = os.path.join(figures_dir, filename)
            mlflow.log_artifact(fig_path, artifact_path="figures")

print("✅ Figures loggées dans MLflow.")
