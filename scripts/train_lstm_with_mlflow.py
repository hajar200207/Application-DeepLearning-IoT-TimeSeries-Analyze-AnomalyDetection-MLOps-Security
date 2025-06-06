import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# === Données ===
X = np.load("scripts/data/processed_data/X_windows_balanced.npy")
y = np.load("scripts/data/processed_data/y_labels_balanced.npy")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === Modèle LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# === MLflow ===
mlflow.set_experiment("AnomalyDetection_LSTM")

with mlflow.start_run(run_name="LSTM_Model"):

    mlflow.log_param("model", "LSTM")
    mlflow.log_param("hidden_size", 64)
    mlflow.log_param("num_layers", 1)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("batch_size", 64)

    # Entraînement
    model.train()
    for epoch in range(5):
        running_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        mlflow.log_metric("loss", avg_loss, step=epoch)
        print(f"📘 Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # Évaluation
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            prob = torch.softmax(out, dim=1)
            preds = torch.argmax(prob, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(prob[:, 1].cpu().numpy())
            all_labels.extend(yb.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    report = classification_report(all_labels, all_preds, digits=4)

    # Logging des résultats
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc", auc)
    mlflow.log_text(report, "classification_report.txt")
    mlflow.pytorch.log_model(model, "lstm_model")

    # Sauvegarde en local
    torch.save(model.state_dict(), "models/LSTM.pt")
    print(f"✅ Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print("✅ Rapport sauvegardé, modèle tracé avec MLflow.")
