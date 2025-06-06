import subprocess
import os

# Liste ordonnée des scripts de preprocessing et visualisation
scripts = [
    "1_load_and_explore.py",          # Chargement et exploration initiale
    "2_filter_timestamp.py",          # Filtrage timestamps si besoin
    "3_clean_and_select.py",          # Nettoyage + sélection features
    "6_data_augmentation.py",         # Jittering / augmentation (sur clean data)
    "4_windowing.py",                 # Windowing sur les données jittered
    "5_normalization.py",             # Normalisation (sur les fenêtres)
    "7_smote_balance.py",             # SMOTE si besoin pour équilibrer classes
    "eda_visuals.py",                 # Visualisation et analyse exploratoire
    "save_final_dataset.py",          # Sauvegarde finale (npys, csv)
    "evaluate_and_save_anomalies.py", # Évaluation modèle et sauvegarde anomalies
    "plot_time_anomalies.py"          # Graphiques anomalies dans le temps
]


# Création du répertoire pour les logs s’il n’existe pas
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Boucle d’exécution
for script in scripts:
    script_path = script
    print(f"[🚀] Exécution : {script}")
    log_path = os.path.join(log_dir, script.replace(".py", ".log"))

    with open(log_path, "w", encoding="utf-8") as logfile:
        result = subprocess.run(["python", script_path], stdout=logfile, stderr=logfile)

    if result.returncode == 0:
        print(f"[✅] Terminé : {script}")
    else:
        print(f"[❌] Erreur dans : {script} → voir {log_path}")
