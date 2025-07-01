import streamlit as st
from streamlit_autorefresh import st_autorefresh
import torch
import numpy as np
import pandas as pd
from collections import deque
from maintenance_model import AnomalyDetectionModel
from data_preprocessor import normalize_imu_data
import os

st.title("Real-Time Predictive Maintenance Dashboard with LSTM Model")

# Model parameters
INPUT_DIM = 3
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQ_LEN = 50
THRESHOLD = 0.5

# Paths (use raw strings or double backslashes)
MODEL_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\models\anomaly_model.pth"
DATA_CSV_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\rgbd_dataset_freiburg2_desk\resampled_accel.csv"

# Check file existence for easier debugging
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

# Initialize session state
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=SEQ_LEN)
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

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
    input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor).squeeze()
        anomaly_score = torch.sigmoid(output).item()
        is_anomaly = anomaly_score > THRESHOLD

    st.markdown(f"### Anomaly Score: {anomaly_score:.4f}")
    if is_anomaly:
        st.markdown("<span style='color:red;font-weight:bold'>Anomaly Detected!</span>", unsafe_allow_html=True)
    else:
        st.markdown("Normal operation")
else:
    st.markdown(f"Waiting for {SEQ_LEN - len(st.session_state.data_buffer)} more data points to start anomaly detection...")

# Auto-refresh every 1 second
st_autorefresh(interval=1000, limit=None, key="datarefresh")