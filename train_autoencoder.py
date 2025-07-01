import torch
import torch.nn as nn
import torch.optim as optim
from maintenance_model import SensorAutoencoder
from data_loader import get_dataloaders
import os

# ---- Hyperparameters ----
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
MODEL_SAVE_PATH = "autoencoder_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Data ----
train_loader, _ = get_dataloaders(windows_path=r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\imu_windows.npy", batch_size=BATCH_SIZE)

# ---- Initialize Model ----
model = SensorAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_inputs in train_loader:
        batch_inputs = batch_inputs.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_inputs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

# ---- Save Trained Model ----
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Autoencoder model saved to {MODEL_SAVE_PATH}")