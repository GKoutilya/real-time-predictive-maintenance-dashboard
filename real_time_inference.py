import torch
import pandas as pd
import numpy as np
import time
from collections import deque
from maintenance_model import AnomalyDetectionModel
from data_preprocessor import normalize_imu_data

# Config
MODEL_PATH = "models/anomaly_model.pth"
VAL_ACCEL_CSV = "rgbd_dataset_freiburg2_desk/resampled_accel.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 50
THRESHOLD = 0.5  # Adjust based on how sensitive you want detection to be

# Load model
model = AnomalyDetectionModel(input_dim=3, hidden_dim=64, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

# Load and normalize data
df = pd.read_csv(VAL_ACCEL_CSV)
df = df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
df = normalize_imu_data(df)

# Simulate real-time stream
stream_buffer = deque(maxlen=WINDOW_SIZE)

print("🔍 Starting real-time anomaly detection...")

for idx, row in df.iterrows():
    accel = [row["ax"], row["ay"], row["az"]]
    stream_buffer.append(accel)

    if len(stream_buffer) == WINDOW_SIZE:
        window_np = np.array(stream_buffer, dtype=np.float32)
        input_tensor = torch.tensor(window_np).unsqueeze(0).to(DEVICE)  # (1, 50, 3)
        with torch.no_grad():
            output = model(input_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            is_anomaly = prob > THRESHOLD

        print(f"[{idx}] Anomaly Score: {prob:.4f} --> {'Anomaly Detected!' if is_anomaly else 'Normal'}")

    time.sleep(0.05)  # Simulate time delay between readings (20 Hz stream)