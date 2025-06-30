import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_accelerometer(input_path: str, output_path: str) -> None:
    """
    Reads raw accelerometer data from a text file, normalizes and resamples it at 10 Hz,
    and saves the processed data as a CSV file.

    Args:
        input_path (str): Path to the raw accelerometer data file (e.g., accelerometer.txt).
        output_path (str): Path where the processed CSV file will be saved.

    Returns:
        None. Writes processed data to output_path.
    """
    if os.path.exists(output_path):
        print(f'File already exists: {output_path}')
        print("Skipping preprocessing and resampling duplication.")
        return
    
    # Read raw accelerometer data line-by-line
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

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data)

    # Normalize accelerometer data (zero mean, unit variance)
    for axis in ["accel_x", "accel_y", "accel_z"]:
        mean = df[axis].mean()
        std = df[axis].std()
        df[axis] = (df[axis] - mean) / std
    
    # Convert timestamp to datetime for resampling
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    # Resample at 10 Hz (every 100ms), interpolate missing values
    df_resampled = df.resample('100ms').mean().interpolate()

    # Reset index and convert timestamp back to float seconds
    df_resampled.reset_index(inplace=True)
    df_resampled['timestamp'] = df_resampled["timestamp"].astype('int64') / 1e9

    # Save preprocessed accelerometer data
    df_resampled.to_csv(output_path, index=False)
    print(f"Saved normalized and resampled accelerometer data to {output_path}")


def normalize_imu_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes IMU accelerometer data columns ('ax', 'ay', 'az') to zero mean and unit variance.

    Args:
        df (pd.DataFrame): DataFrame containing 'ax', 'ay', 'az' columns.

    Returns:
        pd.DataFrame: DataFrame with normalized accelerometer columns.
    """
    scaler = StandardScaler()
    df[['ax', 'ay', 'az']] = scaler.fit_transform(df[['ax', 'ay', 'az']])
    return df


def create_sliding_windows(df: pd.DataFrame, window_size: int = 50, step_size: int = 10) -> np.ndarray:
    """
    Creates overlapping sliding windows from normalized IMU data for LSTM model input.

    Args:
        df (pd.DataFrame): DataFrame with normalized IMU data columns 'ax', 'ay', 'az'.
        window_size (int): Number of samples in each window.
        step_size (int): Step size to move the window (controls overlap).

    Returns:
        np.ndarray: 3D numpy array with shape (num_windows, window_size, 3).
    """
    data = df[['ax', 'ay', 'az']].values
    windows = []

    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)

    return np.array(windows)


def sync_rgb_and_accel(rgb_file: str, accel_file: str, output_file: str) -> None:
    """
    Syncs RGB camera timestamps with accelerometer data by finding the closest accel timestamp
    for each RGB frame timestamp. Saves the synced data as a CSV file.

    Args:
        rgb_file (str): Path to the RGB timestamps file (e.g., rgb.txt).
        accel_file (str): Path to the preprocessed accelerometer CSV file.
        output_file (str): Path where synced RGB + accelerometer CSV will be saved.

    Returns:
        None. Writes synced data to output_file.
    """
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        print("Skipping RGB + accelerometer sync to avoid duplication")
        return

    # Load RGB timestamps and filenames
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

    # Load accelerometer data as DataFrame
    accel_df = pd.read_csv(accel_file)

    # For each RGB timestamp, find the closest accelerometer timestamp
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

    # Convert to DataFrame and save synced CSV
    synced_df = pd.DataFrame(synced_data)
    synced_df.to_csv(output_file, index=False)
    print(f"Saved synced RGB + accelerometer data to {output_file}")

def load_processed_data(windows_path="imu_windows.npy", label_path=None):
    """
    Loads preprocessed sliding windows and returns:
        inputs: torch.Tensor of shape (num_windows, window_size, feature_dim)
        labels: torch.Tensor of shape (num_windows, 1) with anomaly labels
    """
    import torch

    windows = np.load(windows_path)  # shape: (N, window_size, 3)

    # Demo: assign random labels — replace with real logic or from files
    labels = np.random.randint(0, 2, size=(windows.shape[0], 1))

    inputs = torch.tensor(windows, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    return inputs, labels

def inject_anomalies(df, num_anomalies=50, anomaly_type="spike"):
    """
    Injects synthetic anomalies into accelerometer data.
    Returns:
        df (pd.DataFrame): Modified DataFrame with injected anomalies.
        labels (np.ndarray): Array of anomaly labels (1 = anomaly, 0 = normal).
    """
    df = df.copy()
    labels = np.zeros(len(df))

    for _ in range(num_anomalies):
        idx = np.random.randint(0, len(df))
        labels[idx] = 1

        if anomaly_type == "spike":
            df.loc[idx, ["ax", "ay", "az"]] += np.random.normal(5, 2, size=3)
        elif anomaly_type == "flatline":
            if idx + 5 < len(df):
                df.loc[idx:idx+5, ["ax", "ay", "az"]] = df.loc[idx, ["ax", "ay", "az"]].values
                labels[idx:idx+5] = 1

    return df, labels


if __name__ == "__main__":
    # Define file paths
    dataset_dir = "rgbd_dataset_freiburg2_desk"
    raw_accel_file = os.path.join(dataset_dir, "accelerometer.txt")
    preprocessed_accel_file = os.path.join(dataset_dir, "resampled_accel.csv")
    rgb_file = os.path.join(dataset_dir, "rgb.txt")
    synced_output_file = os.path.join(dataset_dir, "synced_rgb_accel.csv")

    # Step 1: Preprocess accelerometer data
    preprocess_accelerometer(raw_accel_file, preprocessed_accel_file)

    # Step 2: Load the preprocessed accelerometer CSV for windowing and ML input (optional here)
    imu_df = pd.read_csv(preprocessed_accel_file)

    # Normalize columns for ML model input
    imu_df = imu_df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
    imu_df = normalize_imu_data(imu_df)

    # Create sliding windows (optional: if you want to save for model training)
    windows = create_sliding_windows(imu_df, window_size=50, step_size=10)
    print(f"Created {windows.shape[0]} windows of shape {windows.shape[1:]}")

    # Save windows as .npy file for training use
    np.save("imu_windows.npy", windows)

    # Step 3: Sync RGB timestamps with accelerometer data
    sync_rgb_and_accel(rgb_file, preprocessed_accel_file, synced_output_file)