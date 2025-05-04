import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import ast
import time

start_time = time.time()

# Charger les données
df = pd.read_csv("data/processed_data/final_dataset_for_review.csv")

# Convertir les colonnes f0 à f3
for col in ["f0", "f1", "f2", "f3"]:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Vectorisation
expanded_cols = []
for col in ["f0", "f1", "f2", "f3"]:
    vectors = pd.DataFrame(df[col].tolist())
    vectors.columns = [f"{col}_{i}" for i in range(vectors.shape[1])]
    expanded_cols.append(vectors)

X = pd.concat(expanded_cols, axis=1)
print("Forme finale des features:", X.shape)

# Isolation Forest
iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df["iso_pred"] = iso_model.fit_predict(X)

# Z-Score
df["zscore_mean"] = X.apply(zscore).abs().mean(axis=1)
df["zscore_anomaly"] = df["zscore_mean"] > 3

# Export
df.to_csv("data/processed_data/anomaly_detection_results.csv", index=False)
print(" Résultats enregistrés dans 'data/processed_data/anomaly_detection_results.csv'")
print("⏱ Temps d'exécution total : %.2f secondes" % (time.time() - start_time))
