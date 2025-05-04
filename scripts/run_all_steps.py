import subprocess
import os

scripts = [
    "1_load_and_explore.py",
    "2_filter_timestamp.py",
    "3_clean_and_select.py",
    "4_windowing.py",
    "5_normalization.py",
    "6_data_augmentation.py",
    "7_smote_balance.py",
    "eda_visuals.py",
    "save_final_dataset.py",
    "evaluate_and_save_anomalies.py",
    "plot_time_anomalies.py"
]

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

for script in scripts:
    script_path = script  # <- correction ici
    print(f" Exécution : {script}")
    with open(os.path.join(log_dir, script.replace(".py", ".log")), "w", encoding="utf-8") as logfile:
        result = subprocess.run(["python", script_path], stdout=logfile, stderr=logfile)
        if result.returncode == 0:
            print(f"Terminé : {script}")
        else:
            print(f" Erreur dans : {script} (voir logs)")
