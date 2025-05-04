
import numpy as np
import pandas as pd
import random

df = pd.read_csv("data/processed_data/IoT_Modbus_cleaned.csv", index_col=0)

# Jittering: ajoute un bruit gaussien aux valeurs numériques
def jitter(data, noise_level=0.01):
    return data + np.random.normal(loc=0.0, scale=noise_level, size=data.shape)

# Sauvegarde d’un exemple de Jittering
jittered_df = df.copy()
features = jittered_df.drop(columns=["label"])
jittered_df[features.columns] = jitter(features)
jittered_df.to_csv("data/processed_data/IoT_Modbus_jittered.csv")
print("Fichier jittered sauvegardé.")
