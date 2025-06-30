import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from maintenance_model import AnomalyDetectionModel
from data_preprocessor import normalize_imu_data, create_sliding_windows, inject_anomalies

# Config
MODEL_PATH = "models/anomaly_model.pth"
VAL_ACCEL_CSV = "rgbd_dataset_freiburg2_desk/resampled_accel.csv"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_validation_data():
    df = pd.read_csv(VAL_ACCEL_CSV)
    df = df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
    df = normalize_imu_data(df)

    # Inject anomalies in the dataframe BEFORE sliding windows
    df_with_anomalies, labels_per_sample = inject_anomalies(df, num_anomalies=int(0.1 * len(df)))

    # Now create sliding windows from the modified df
    windows = create_sliding_windows(df_with_anomalies, window_size=50, step_size=10)

    # You need to create window-level labels from sample-level labels
    labels = create_window_labels(labels_per_sample, window_size=50, step_size=10)

    return windows, labels


def create_window_labels(sample_labels, window_size, step_size):
    """
    Converts sample-level anomaly labels into window-level labels.
    If any sample in the window is anomalous (label==1), label the window as anomalous.

    Args:
        sample_labels (np.ndarray): Array of labels per sample (0 or 1).
        window_size (int): Size of each sliding window.
        step_size (int): Step size between windows.

    Returns:
        np.ndarray: Array of labels per window.
    """
    window_labels = []
    for start in range(0, len(sample_labels) - window_size + 1, step_size):
        window = sample_labels[start:start + window_size]
        window_labels.append(1 if np.any(window == 1) else 0)
    return np.array(window_labels)

def prepare_dataloader(windows, labels):
    # Convert numpy arrays to PyTorch tensors
    X = torch.tensor(windows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # Load model
    model = AnomalyDetectionModel(input_dim=3, hidden_dim=64, num_layers=2)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)

    # Load validation data with injected anomalies and labels
    windows, labels = load_validation_data()
    val_loader = prepare_dataloader(windows, labels)
    evaluate(model, val_loader)