from fastapi import FastAPI
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/lstm_iot.h5')
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API IoT Anomaly Detection"}

@app.post("/predict")
def predict(data: list):
    df = pd.DataFrame(data)
    X = df[['temperature', 'rolling_mean', 'z_score']].fillna(0).values
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    predictions = model.predict(X)
    return {"anomalies": (predictions > 0.5).astype(int).tolist()}
