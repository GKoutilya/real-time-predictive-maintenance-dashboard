import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt
from maintenance_model import AnomalyDetectionModel
from inference_autoencoder import load_autoencoder_model, compute_reconstruction_error
from data_preprocessor import normalize_imu_data, create_sliding_windows

# Config
LSTM_MODEL_PATH = "models/anomaly_model.pth"
AE_MODEL_PATH = "autoencoder_model.pth"
LABELED_CSV_PATH = "rgbd_dataset_freiburg2_desk/labeled_accel.csv"

BATCH_SIZE = 32
WINDOW_SIZE = 50
STEP_SIZE = 10
THRESHOLD_LSTM = 0.5
THRESHOLD_AE = None  # We'll compute it dynamically or set manually
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_window_labels(sample_labels, window_size, step_size):
    window_labels = []
    for start in range(0, len(sample_labels) - window_size + 1, step_size):
        window = sample_labels[start : start + window_size]
        window_labels.append(1 if np.any(window == 1) else 0)
    return np.array(window_labels)


def prepare_dataloader(windows):
    X = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_model(model, dataloader, threshold, model_name="LSTM"):
    model.eval()
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for inputs_tuple in dataloader:
            inputs = inputs_tuple[0].to(DEVICE)
            outputs = model(inputs).squeeze()
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = scores > threshold
            all_scores.extend(scores)
            all_preds.extend(preds)

    return np.array(all_scores), np.array(all_preds)


def evaluate_autoencoder(model, windows, threshold=None):
    scores = []
    for window in windows:
        score = compute_reconstruction_error(model, window)[0]
        scores.append(score)
    scores = np.array(scores)

    if threshold is None:
        # Set threshold as mean + 3*std dev on scores from windows labeled normal (0)
        # For simplicity, assume the first 100 windows are normal (adjust as needed)
        normal_scores = scores[:100]
        threshold = normal_scores.mean() + 3 * normal_scores.std()
        print(f"Autoencoder threshold set dynamically to {threshold:.4f}")

    preds = scores > threshold
    return scores, preds, threshold


def print_metrics(true_labels, preds, model_name="Model"):
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")


def plot_roc_pr(true_labels, scores, model_name="Model"):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(true_labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR curve (area = {pr_auc:.2f})")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{model_name}_roc_pr_curves.png")
    plt.close()


if __name__ == "__main__":
    print("Loading labeled dataset...")
    df = pd.read_csv(LABELED_CSV_PATH)
    df = df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
    df = normalize_imu_data(df)

    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column with 0/1 anomaly labels.")

    sample_labels = df["label"].values

    print("Creating sliding windows...")
    windows = create_sliding_windows(df, WINDOW_SIZE, STEP_SIZE)
    window_labels = create_window_labels(sample_labels, WINDOW_SIZE, STEP_SIZE)

    # Prepare DataLoader for LSTM
    val_loader = prepare_dataloader(windows)

    print("Loading LSTM model...")
    lstm_model = AnomalyDetectionModel(input_dim=3, hidden_dim=64, num_layers=2)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    lstm_model.to(DEVICE)

    print("Evaluating LSTM model...")
    lstm_scores, lstm_preds = evaluate_model(lstm_model, val_loader, THRESHOLD_LSTM, "LSTM")
    print_metrics(window_labels, lstm_preds, "LSTM")

    print("\nLoading Autoencoder model...")
    ae_model = load_autoencoder_model(AE_MODEL_PATH)
    ae_model.to(DEVICE)

    print("Evaluating Autoencoder model...")
    ae_scores, ae_preds, ae_threshold = evaluate_autoencoder(ae_model, windows, threshold=THRESHOLD_AE)
    print_metrics(window_labels, ae_preds, "Autoencoder")

    print("\nSaving detailed results to 'evaluation_results.csv'...")
    results_df = pd.DataFrame({
        "window_start_idx": np.arange(len(window_labels)) * STEP_SIZE,
        "label": window_labels,
        "lstm_score": lstm_scores,
        "lstm_pred": lstm_preds.astype(int),
        "ae_score": ae_scores,
        "ae_pred": ae_preds.astype(int),
    })
    results_df.to_csv("evaluation_results.csv", index=False)

    print("Plotting LSTM metrics...")
    plot_roc_pr(window_labels, lstm_scores, "LSTM")

    print("Plotting Autoencoder metrics...")
    plot_roc_pr(window_labels, ae_scores, "Autoencoder")

    print("\nDone evaluation.")
