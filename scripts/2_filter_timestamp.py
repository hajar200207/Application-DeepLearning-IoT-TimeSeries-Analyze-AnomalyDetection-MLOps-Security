import pandas as pd

INPUT = "data/processed_data/IoT_Modbus_loaded.csv"
OUTPUT = "data/processed_data/IoT_Modbus_timestamped.csv"

df = pd.read_csv(INPUT, index_col=0, parse_dates=True)

# Filtrage simple si besoin (par date par exemple)
df = df[df.index.notnull()]

df.to_csv(OUTPUT)
print(" 2_filter_timestamp terminé")
