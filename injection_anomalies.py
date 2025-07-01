import pandas as pd
import numpy as np
import random

# --- Configuration ---
DATA_PATH = r"rgbd_dataset_freiburg2_desk/resampled_accel.csv"
OUTPUT_PATH = r"rgbd_dataset_freiburg2_desk/labeled_accel.csv"
SEQ_LEN = 50
NUM_ANOMALIES = 8  # Total injected events
ANOMALY_TYPES = ['spike', 'drift', 'dropout', 'noise']

# --- Load real IMU data ---
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
df['label'] = 0  # Start with all normal

# --- Helper Functions ---
def inject_spike(data, col):
    spike_mag = np.random.uniform(3, 5)
    direction = np.random.choice([-1, 1])
    data[col] += direction * spike_mag
    return data

def inject_drift(data, col):
    drift = np.linspace(0, np.random.uniform(1, 2), SEQ_LEN)
    data[col] += drift
    return data

def inject_dropout(data, col):
    constant = data[col].iloc[0]
    data[col] = constant
    return data

def inject_noise(data, col):
    noise = np.random.normal(0, 1.0, SEQ_LEN)
    data[col] += noise
    return data

# --- Injection Loop ---
used_indices = set()
for i in range(NUM_ANOMALIES):
    start_idx = random.randint(0, len(df) - SEQ_LEN - 1)
    while any((start_idx <= idx < start_idx + SEQ_LEN) for idx in used_indices):
        start_idx = random.randint(0, len(df) - SEQ_LEN - 1)

    anomaly_type = random.choice(ANOMALY_TYPES)
    col = random.choice(['ax', 'ay', 'az'])

    window = df.iloc[start_idx : start_idx + SEQ_LEN].copy()

    if anomaly_type == 'spike':
        window = inject_spike(window, col)
    elif anomaly_type == 'drift':
        window = inject_drift(window, col)
    elif anomaly_type == 'dropout':
        window = inject_dropout(window, col)
    elif anomaly_type == 'noise':
        window = inject_noise(window, col)

    window['label'] = 1
    df.iloc[start_idx : start_idx + SEQ_LEN] = window

    used_indices.update(range(start_idx, start_idx + SEQ_LEN))
    print(f"Injected {anomaly_type.upper()} on {col} at index {start_idx}")

# --- Save ---
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved labeled data with {NUM_ANOMALIES} anomalies → {OUTPUT_PATH}")
