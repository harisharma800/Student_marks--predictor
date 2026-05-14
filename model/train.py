import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("Training shuru!")

X = np.array([
    [6.0, 85.0, 72.0],
    [8.0, 92.0, 88.0],
    [3.0, 65.0, 55.0],
    [7.0, 90.0, 82.0],
    [5.0, 75.0, 68.0],
    [9.0, 95.0, 90.0],
    [4.0, 70.0, 60.0],
    [7.5, 88.0, 79.0],
    [2.0, 55.0, 45.0],
    [8.5, 94.0, 87.0],
], dtype=np.float32)

y = np.array([
    78.0, 91.0, 58.0, 85.0, 70.0,
    93.0, 63.0, 83.0, 48.0, 90.0,
], dtype=np.float32).reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

class MarksPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.network(x)

model     = MarksPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/1000 | Loss: {loss.item():.4f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/model.pth")
with open("model/scalers.pkl", "wb") as f:
    pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)

print("Done! model/model.pth ban gaya!")