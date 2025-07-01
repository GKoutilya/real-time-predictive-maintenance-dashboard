import streamlit as st
from streamlit_autorefresh import st_autorefresh
import torch
import numpy as np
import pandas as pd
from collections import deque
from maintenance_model import AnomalyDetectionModel
from data_preprocessor import normalize_imu_data
from inference_autoencoder import load_autoencoder_model, compute_reconstruction_error, is_anomalous
from plyer import notification
import os

# Load models
AUTOENCODER_MODEL_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\autoencoder_model.pth"
autoencoder_model = load_autoencoder_model(AUTOENCODER_MODEL_PATH)

st.title("Real-Time Predictive Maintenance Dashboard with LSTM + Autoencoder")

# Model params
INPUT_DIM = 3
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQ_LEN = 50
THRESHOLD = 0.5

# Severity classification thresholds (tune as needed)
SEVERE_THRESHOLD_LSTM = 0.9
SEVERE_THRESHOLD_AE = 0.15

# Paths
MODEL_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\models\anomaly_model.pth"
DATA_CSV_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\rgbd_dataset_freiburg2_desk\resampled_accel.csv"

if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
if not os.path.isfile(DATA_CSV_PATH):
    st.error(f"Data CSV file not found at {DATA_CSV_PATH}")

@st.cache_resource
def load_model():
    model = AnomalyDetectionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_CSV_PATH)
    df = df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
    df = normalize_imu_data(df)
    return df

data = load_data()

# Initialize session state buffers and idx
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=SEQ_LEN)
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# Initialize rolling score buffers for visualization (last 100 points)
if 'lstm_scores' not in st.session_state:
    st.session_state.lstm_scores = deque(maxlen=100)
if 'ae_scores' not in st.session_state:
    st.session_state.ae_scores = deque(maxlen=100)
if 'alerts' not in st.session_state:
    st.session_state.alerts = deque(maxlen=100)

# --- UI: Model Selection ---
model_option = st.selectbox(
    "Choose anomaly detection mode",
    options=["LSTM only", "Autoencoder only", "Both (OR)", "Both (AND)"],
    index=2
)

# --- UI: Desktop Notification Toggle ---
enable_notifications = st.checkbox("Enable desktop notifications for severe anomalies", value=False)
st.session_state.enable_notifications = enable_notifications

# Stream next data point
if st.session_state.current_idx < len(data):
    next_reading = data.iloc[st.session_state.current_idx][["ax", "ay", "az"]].values
    st.session_state.data_buffer.append(next_reading)
    st.session_state.current_idx += 1
else:
    st.markdown("End of dataset reached.")
    st.stop()

st.markdown(f"### Latest Sensor Reading: {next_reading}")

if len(st.session_state.data_buffer) == SEQ_LEN:
    input_window = np.array(st.session_state.data_buffer)

    # LSTM inference
    input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor).squeeze()
        lstm_score = torch.sigmoid(output).item()
        lstm_anomaly = lstm_score > THRESHOLD

    # Autoencoder inference
    ae_score = compute_reconstruction_error(autoencoder_model, input_window)[0]
    ae_anomaly = is_anomalous(ae_score)

    # Save scores for plotting
    st.session_state.lstm_scores.append(lstm_score)
    st.session_state.ae_scores.append(ae_score)

    # Determine alert based on selected ensemble logic
    if model_option == "LSTM only":
        alert = lstm_anomaly
    elif model_option == "Autoencoder only":
        alert = ae_anomaly
    elif model_option == "Both (OR)":
        alert = lstm_anomaly or ae_anomaly
    else:  # Both (AND)
        alert = lstm_anomaly and ae_anomaly

    st.session_state.alerts.append(int(alert))

    # Display scores and alert
    st.markdown(f"### LSTM Anomaly Score: {lstm_score:.4f}")
    st.markdown(f"### AE Reconstruction Error: {ae_score:.4f}")

    # --- Severity Classification & Notification ---
    severity = None
    if alert:
        if lstm_score > SEVERE_THRESHOLD_LSTM or ae_score > SEVERE_THRESHOLD_AE:
            severity = "SEVERE"
        elif lstm_score > 0.7 or ae_score > 0.1:
            severity = "Moderate"
        else:
            severity = "Low"

        st.markdown(f"<span style='color:red;font-weight:bold'>Anomaly Detected! Severity: {severity}</span>", unsafe_allow_html=True)
        st.markdown(f"Triggered by: {model_option}")

        if st.session_state.enable_notifications:
            from plyer import notification
            notification.notify(
                title="Severe Anomaly Detected!",
                message=f"Triggered by: {model_option}",
                timeout=5
            )

        if severity == "SEVERE":
            notification.notify(
                title="🚨 Severe Anomaly Detected",
                message=f"{model_option} triggered: LSTM={lstm_score:.4f}, AE={ae_score:.4f}",
                timeout=5
            )
    else:
        st.markdown("Normal operation")

    # Plot rolling scores
    st.line_chart({
        "LSTM Score": list(st.session_state.lstm_scores),
        "Autoencoder Score": list(st.session_state.ae_scores),
        "Alert": list(st.session_state.alerts),
    })
else:
    st.markdown(f"Waiting for {SEQ_LEN - len(st.session_state.data_buffer)} more data points to start anomaly detection...")

# Auto-refresh every 1 second
st_autorefresh(interval=1000, limit=None, key="datarefresh")
