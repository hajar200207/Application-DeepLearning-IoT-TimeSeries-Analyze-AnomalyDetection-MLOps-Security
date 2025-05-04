import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# 1. Charger le dataset
df = pd.read_csv("data/processed_data/final_dataset_for_review.csv")

# 2. Convertir les colonnes vectorisées
for col in ["f0", "f1", "f2", "f3"]:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 3. Étaler les colonnes
X = pd.concat([pd.DataFrame(df[col].tolist()).add_prefix(f"{col}_") for col in ["f0", "f1", "f2", "f3"]], axis=1)

# 4. Isolation Forest
iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df["iso_pred"] = iso_model.fit_predict(X).astype(int)
df["iso_pred"] = df["iso_pred"].map({1: 0, -1: 1})

# 5. Z-Score
df["zscore_mean"] = X.apply(zscore).abs().mean(axis=1)
df["zscore_pred"] = (df["zscore_mean"] > 3).astype(int)

# 6. Ground Truth
y_true = df["label"]

# 7. Sauvegarder les rapports de classification
with open("data/processed_data/evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write("Isolation Forest:\n")
    print(classification_report(y_true, df["iso_pred"], zero_division=0))
    f.write("\n\n Z-Score:\n")
    f.write(classification_report(y_true, df["zscore_pred"]))

# 8. Sauvegarder les données avec prédictions
df.to_csv("data/processed_data/detection_with_predictions.csv", index=False)

# 9. Sauvegarder les matrices de confusion
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_true, df["iso_pred"]), annot=True, fmt="d", cmap="Blues")
plt.title("Matrice Confusion - Isolation Forest")
plt.xlabel("Prédit")
plt.ylabel("Réel")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_true, df["zscore_pred"]), annot=True, fmt="d", cmap="Greens")
plt.title("Matrice Confusion - Z-Score")
plt.xlabel("Prédit")
plt.ylabel("Réel")

plt.tight_layout()
plt.savefig("data/processed_data/confusion_matrices.png")

print("Évaluation enregistrée dans :")
print(" - detection_with_predictions.csv")
print(" - evaluation_report.txt")
print(" - confusion_matrices.png")
