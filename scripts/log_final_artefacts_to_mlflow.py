import mlflow
import os

# Définir l'expérience
mlflow.set_experiment("AnomalyDetection_Final_Artefacts")

# Démarrer un run
with mlflow.start_run(run_name="Figures_rapports_tableau"):

    # === Chemins des fichiers ===
    path_fig1 = "results/figures/barplot_comparaison_models.png"
    path_fig2 = "results/figures/radar_comparaison_models.png"
    path_txt1 = "results/resultats_detection_DL_IoT_Modbus.txt"
    path_txt2 = "results/traditional_methods_report.txt"
    path_csv  = "results/comparatifs/comparatif_global_models.csv"

    # === Logging des artefacts ===
    mlflow.log_artifact(path_fig1, artifact_path="figures")
    mlflow.log_artifact(path_fig2, artifact_path="figures")

    mlflow.log_artifact(path_txt1, artifact_path="rapports")
    mlflow.log_artifact(path_txt2, artifact_path="rapports")

    mlflow.log_artifact(path_csv, artifact_path="comparatifs")

    print("✅ Figures, rapports et tableau CSV loggés avec succès dans MLflow.")
