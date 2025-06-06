import pandas as pd

INPUT = "data/processed_data/IoT_Modbus_timestamped.csv"
OUTPUT = "data/processed_data/IoT_Modbus_cleaned.csv"

df = pd.read_csv(INPUT, index_col=0, parse_dates=True)

# Supprimer colonnes inutiles (il n'y en a pas ici mais à adapter)
# Sélectionner colonnes numériques + label
columns_to_keep = ['FC1_Read_Input_Register', 'FC2_Read_Discrete_Value','FC3_Read_Holding_Register', 'FC4_Read_Coil', 'label']
df = df[columns_to_keep]

df.to_csv(OUTPUT)
print("3_clean_and_select terminé")