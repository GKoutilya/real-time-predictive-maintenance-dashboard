import streamlit as st
import torch
import numpy as np
from collections import deque
from maintenance_model import AnomalyDetectionModel

st.title("Real-Time Predictive Maintenance Dashboard with LSTM Model")

# Model parameters - match your trained model architecture
INPUT_DIM = 3  # e.g., accelerometer axes
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQ_LEN = 50  # window size your model expects

# Load model once on app startup
@st.cache_resource  # Cache model to avoid reload on rerun
def load_model():
    model = AnomalyDetectionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(torch.load('models/anomaly_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Buffer to hold recent sensor data windows for LSTM input
if 'data_buffer' not in st.session_state:
    # Deque with maxlen = SEQ_LEN, storing each data point as [ax, ay, az]
    st.session_state.data_buffer = deque(maxlen=SEQ_LEN)

# Simulate or load your incoming sensor reading here:
def get_new_sensor_reading():
    # Replace with real sensor data streaming logic
    return np.random.normal(size=(INPUT_DIM,))  # dummy 3-axis reading

# Add new reading to buffer
new_reading = get_new_sensor_reading()
st.session_state.data_buffer.append(new_reading)

st.markdown(f"### Latest Sensor Reading: {new_reading}")

# Only run inference if buffer is full (enough time steps)
if len(st.session_state.data_buffer) == SEQ_LEN:
    # Prepare input tensor shape: (1, seq_len, input_dim)
    input_window = np.array(st.session_state.data_buffer)
    input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor).squeeze()
        anomaly_score = torch.sigmoid(output).item()
        is_anomaly = anomaly_score > 0.5  # threshold can be tuned

    st.markdown(f"### Anomaly Score: {anomaly_score:.4f}")
    if is_anomaly:
        st.markdown("**Anomaly Detected!**", unsafe_allow_html=True)
    else:
        st.markdown("Normal operation")
else:
    st.markdown(f"Waiting for {SEQ_LEN - len(st.session_state.data_buffer)} more data points to start anomaly detection...")

# Auto-refresh every 500ms to simulate real-time streaming
st_autorefresh = st.experimental_rerun