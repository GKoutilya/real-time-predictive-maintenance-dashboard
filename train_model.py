import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from maintenance_model import AnomalyDetectionModel, train_model
from data_preprocessor import load_processed_data

# Hyperparameters
input_dim = 3  # since windows have axes ax,ay,az
hidden_dim = 64
num_layers = 2
batch_size = 32
epochs = 20
lr = 1e-3
train_ratio = 0.8

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load data with correct label loading
inputs, labels = load_processed_data(windows_path="imu_windows.npy", labels_path="imu_labels.npy")

# Dataset & split
dataset = TensorDataset(inputs, labels)
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss, optimizer
model = AnomalyDetectionModel(input_dim, hidden_dim, num_layers).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# If your train_model function expects inputs on device, verify it or wrap data here
# Assuming train_model handles device internally; if not, we'd have to modify it.

# Train model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

# Save the trained model
model_save_path = "models/anomaly_model.pth"
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
