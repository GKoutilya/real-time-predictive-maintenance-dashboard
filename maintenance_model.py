"""
maintenance_model.py

Contains the machine learning model(s) for predictive maintenance:
- Anomaly detection
- Fault prediction (if applicable)

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
    Neural network model for anomaly detection in sensor data.

    Modify or extend this class based on your architecture
    (e.g., LSTM, CNN, autoencoder).
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(AnomalyDetectionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output: anomaly score or binary class

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Output tensor with anomaly score or classification result
        """
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)

        return out.squeeze(-1)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu'):
    """
    Trains the anomaly detection model.

    Args:
        model (nn.Module): The PyTorch model to train
        train_loader (DataLoader): Training dataset loader
        val_loader (DataLoader): Validation dataset loader
        criterion: Loss function
        optimizer: Optimizer for model parameters
        epochs (int): Number of epochs to train
        device (str): 'cpu' or 'cuda'

    Returns:
        model: Trained model
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