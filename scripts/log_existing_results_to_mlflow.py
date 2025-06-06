import mlflow

# Définir le nom de l'expérience
mlflow.set_experiment("AnomalyDetection_ExistingResults")

def log_metrics(model_name, accuracy, auc, precision, recall, f1):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("Model", model_name)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-score", f1)

def extract_metrics_from_block(block):
    lines = block.strip().splitlines()
    
    # Extraire accuracy et AUC
    acc_line = next((l for l in lines if "Accuracy" in l), None)
    auc_line = next((l for l in lines if "AUC-ROC" in l), None)
    accuracy = float(acc_line.split(":")[1].strip()) if acc_line else 0.0
    auc = float(auc_line.split(":")[1].strip()) if auc_line else 0.0

    # Trouver la ligne avec "macro avg"
    for l in lines:
        if "macro avg" in l:
            parts = l.strip().split()
            try:
                # Les 3 dernières colonnes = precision, recall, f1-score
                precision = float(parts[-4])
                recall = float(parts[-3])
                f1 = float(parts[-2])
                return accuracy, auc, precision, recall, f1
            except ValueError:
                continue

    # Si pas trouvé
    return accuracy, auc, 0.0, 0.0, 0.0



# Charger les deux fichiers
with open("results/resultats_detection_DL_IoT_Modbus.txt", "r", encoding="utf-8") as f:
    dl_content = f.read()

with open("results/traditional_methods_report.txt", "r", encoding="utf-8") as f:
    trad_content = f.read()

# Traiter les blocs Deep Learning
for block in dl_content.split("🔍 Évaluation :")[1:]:
    lines = block.strip().splitlines()
    model_name = lines[0].strip()
    acc, auc, precision, recall, f1 = extract_metrics_from_block(block)
    log_metrics(model_name, acc, auc, precision, recall, f1)

# Traiter les blocs traditionnels
for block in trad_content.split("🔍 Évaluation :")[1:]:
    lines = block.strip().splitlines()
    model_name = lines[0].strip()
    acc, auc, precision, recall, f1 = extract_metrics_from_block(block)
    log_metrics(model_name, acc, auc, precision, recall, f1)
