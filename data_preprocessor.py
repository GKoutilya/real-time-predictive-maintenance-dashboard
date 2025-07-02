import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def preprocess_accelerometer(input_path: str, output_path: str) -> None:
    """
    Reads raw accelerometer data from a text file, normalizes and resamples it at 10 Hz,
    and saves the processed data as a CSV file.
    """
    if os.path.exists(output_path):
        print(f'File already exists: {output_path}')
        print("Skipping preprocessing and resampling duplication.")
        return

    with open(input_path, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 4:
            timestamp, ax, ay, az = parts
            data.append({
                "timestamp": float(timestamp),
                "accel_x": float(ax),
                "accel_y": float(ay),
                "accel_z": float(az)
            })

    df = pd.DataFrame(data)

    for axis in ["accel_x", "accel_y", "accel_z"]:
        mean = df[axis].mean()
        std = df[axis].std()
        df[axis] = (df[axis] - mean) / std

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    df_resampled = df.resample('100ms').mean().interpolate()

    df_resampled.reset_index(inplace=True)
    df_resampled['timestamp'] = df_resampled["timestamp"].astype('int64') / 1e9

    df_resampled.to_csv(output_path, index=False)
    print(f"Saved normalized and resampled accelerometer data to {output_path}")

def normalize_imu_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df[['ax', 'ay', 'az']] = scaler.fit_transform(df[['ax', 'ay', 'az']])
    return df

def create_sliding_windows(df: pd.DataFrame, window_size: int = 50, step_size: int = 10):
    data = df[['ax', 'ay', 'az']].values
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    return np.array(windows)

def sync_rgb_and_accel(rgb_file: str, accel_file: str, output_file: str) -> None:
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        print("Skipping RGB + accelerometer sync to avoid duplication")
        return

    rgb_timestamps = []
    rgb_filenames = []

    with open(rgb_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                timestamp, filename = parts
                rgb_timestamps.append(float(timestamp))
                rgb_filenames.append(filename)

    accel_df = pd.read_csv(accel_file)

    synced_data = []
    for ts, filename in zip(rgb_timestamps, rgb_filenames):
        closest_idx = (accel_df['timestamp'] - ts).abs().idxmin()
        closest_row = accel_df.loc[closest_idx]
        synced_data.append({
            "rgb_timestamp": ts,
            "rgb_file": filename,
            "accel_timestamp": closest_row["timestamp"],
            "accel_x": closest_row["accel_x"],
            "accel_y": closest_row["accel_y"],
            "accel_z": closest_row["accel_z"]
        })

    synced_df = pd.DataFrame(synced_data)
    synced_df.to_csv(output_file, index=False)
    print(f"Saved synced RGB + accelerometer data to {output_file}")

def load_processed_data(windows_path="imu_windows.npy", labels_path="imu_labels.npy"):
    """
    Loads preprocessed sliding windows and their true labels.

    Returns:
        inputs: torch.Tensor of shape (num_windows, window_size, feature_dim)
        labels: torch.Tensor of shape (num_windows, 1) with anomaly labels
    """
    windows = np.load(windows_path)
    labels = np.load(labels_path)

    inputs = torch.tensor(windows, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    return inputs, labels

if __name__ == "__main__":
    dataset_dir = "rgbd_dataset_freiburg2_desk"
    raw_accel_file = os.path.join(dataset_dir, "accelerometer.txt")
    preprocessed_accel_file = os.path.join(dataset_dir, "resampled_accel.csv")
    rgb_file = os.path.join(dataset_dir, "rgb.txt")
    synced_output_file = os.path.join(dataset_dir, "synced_rgb_accel.csv")

    preprocess_accelerometer(raw_accel_file, preprocessed_accel_file)

    imu_df = pd.read_csv(preprocessed_accel_file)
    imu_df = imu_df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
    imu_df = normalize_imu_data(imu_df)

    windows = create_sliding_windows(imu_df, window_size=50, step_size=10)
    print(f"Created {windows.shape[0]} windows of shape {windows.shape[1:]}")

    np.save("imu_windows.npy", windows)

    sync_rgb_and_accel(rgb_file, preprocessed_accel_file, synced_output_file)