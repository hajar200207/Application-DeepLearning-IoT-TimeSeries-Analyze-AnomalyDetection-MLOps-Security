
# import pandas as pd

# def load_and_process_data():
#     # Charger les données brutes
#     df_temp = pd.read_csv('data/raw_data/temperature.csv')
#     df_window = pd.read_csv('data/raw_data/window_opened_closed.csv')

#     # Convertir en datetime
#     df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
#     df_window['datetime'] = pd.to_datetime(df_window['datetime'])

#     # Trier
#     df_temp.sort_values('datetime', inplace=True)
#     df_window.sort_values('datetime', inplace=True)

#     # Ajout des statistiques
#     df_temp['rolling_mean'] = df_temp['temperature'].rolling(window=10).mean()
#     df_temp['rolling_std'] = df_temp['temperature'].rolling(window=10).std()
#     df_temp['z_score'] = (df_temp['temperature'] - df_temp['rolling_mean']) / df_temp['rolling_std']
#     df_temp['temp_max'] = df_temp['temperature'].rolling('1h', on='datetime').max()
#     df_temp['temp_min'] = df_temp['temperature'].rolling('1h', on='datetime').min()

#     # Fusion
#     df_merged = pd.merge_asof(df_temp, df_window, on='datetime', direction='nearest')

#     # Export
#     os.makedirs('data/processed_data', exist_ok=True)
#     df_temp.to_csv('data/processed_data/temperature_processed.csv', index=False)
#     df_merged.to_csv('data/processed_data/merged_data.csv', index=False)
#     print("✅ Données traitées et sauvegardées dans data/processed_data/")

# if __name__ == '__main__':
#     load_and_process_data()







import pandas as pd
import os

def preprocess():
    df = pd.read_csv('data/raw_data/temperature.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    # Rolling features
    df['rolling_mean'] = df['temperature'].rolling(10).mean()
    df['rolling_std'] = df['temperature'].rolling(10).std()
    df['z_score'] = (df['temperature'] - df['rolling_mean']) / df['rolling_std']

    os.makedirs('data/processed_data', exist_ok=True)
    df.to_csv('data/processed_data/temperature_processed.csv', index=False)

if __name__ == '__main__':
    preprocess()
