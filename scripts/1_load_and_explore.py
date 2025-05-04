import pandas as pd
import os

INPUT_PATH = "../data/raw_data/IoT_Modbus.csv"
OUTPUT_PATH = "data/processed_data/IoT_Modbus_loaded.csv"

os.makedirs("data/processed_data", exist_ok=True)

# Charger et fusionner date/time en timestamp
df = pd.read_csv(INPUT_PATH)
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df.drop(columns=['date', 'time'], inplace=True)
df = df.sort_values('timestamp')
df.set_index('timestamp', inplace=True)
df.to_csv(OUTPUT_PATH)
print(" 1_load_and_explore terminé")
