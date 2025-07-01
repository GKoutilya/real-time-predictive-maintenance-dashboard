"""
maintenance_model.py

Contains the machine learning models for predictive maintenance:
- Anomaly detection
- Fault prediction

Includes:
- Model architecture definitions
- Training utilities
- Inference functions
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class AnomalyDetectionModel(nn.Module):
    """
    LSTM-based Anomaly Detection Model for time-series sensor data.

    Args:
        input_dim (int): Number of features per timestep (e.g., 3 for x,y,z accel).
        hidden_dim (int): Number of hidden units in LSTM layers.
        num_layers (int): Number of stacked LSTM layers.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Forward Output:
        torch.Tensor: Output anomaly score logits of shape (batch_size).
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2):
        super(AnomalyDetectionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]  # Take last timestep output
        out = self.fc(out)   # Linear layer to 1 output
        return out.squeeze() # (batch,)


class SensorAutoencoder(nn.Module):
    """
    Autoencoder model for unsupervised anomaly detection in sensor data sequences.

    Args:
        input_dim (int): Number of features per timestep.
        latent_dim (int): Dimensionality of latent representation.
        seq_len (int): Length of input sequences.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Forward Output:
        torch.Tensor: Reconstructed input tensor of shape (batch_size, seq_len, input_dim).
    """
    def __init__(self, input_dim=3, latent_dim=16, seq_len=50):
        super(SensorAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim * seq_len),
            nn.Sigmoid()  # Assuming normalized input between 0 and 1
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Flatten (batch, seq_len * input_dim)
        latent = self.encoder(x_flat)
        recon = self.decoder(latent)
        recon = recon.view(batch_size, self.seq_len, self.input_dim)
        return recon


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                epochs: int, device: str = 'cpu') -> nn.Module:
    """
    Trains the PyTorch model for binary anomaly detection.

    Args:
        model (nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for training data, yields (inputs, labels).
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epochs (int): Number of training epochs.
        device (str): Device to run training on ('cpu' or 'cuda').

    Returns:
        nn.Module: The trained model.
    """
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.squeeze().float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.squeeze().float())
                val_loss += loss.item() * inputs.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} "
              f"- Acc: {acc:.4f} - Prec: {prec:.4f} - Rec: {rec:.4f} - F1: {f1:.4f}")

    return model


def infer(model, input_data):
    """
    Runs inference on input sensor data to detect anomalies.

    Args:
        model (nn.Module): Trained model
        input_data (torch.Tensor): Input data tensor

    Returns:
        torch.Tensor: Anomaly scores or predictions
    """
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output


if __name__ == "__main__":
    print("This module contains ML model classes and training/inference functions for maintenance.")