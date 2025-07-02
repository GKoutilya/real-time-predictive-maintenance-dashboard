import streamlit as st
from streamlit_autorefresh import st_autorefresh
import torch
import numpy as np
import pandas as pd
from collections import deque
from maintenance_model import AnomalyDetectionModel
from data_preprocessor import normalize_imu_data
from inference_autoencoder import load_autoencoder_model, compute_reconstruction_error, is_anomalous
import os
import io
import plotly.graph_objects as go

# Stream generator setup
INPUT_CSV_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\rgbd_dataset_freiburg2_desk\resampled_accel.csv"

def stream_data():
    if not os.path.exists(INPUT_CSV_PATH):
        st.error(f"Data file not found: {INPUT_CSV_PATH}")
        return
    df = pd.read_csv(INPUT_CSV_PATH)
    df = df.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az"})
    df = normalize_imu_data(df)
    for _, row in df.iterrows():
        timestamp = row["timestamp"] if "timestamp" in row else None
        data = [row["ax"], row["ay"], row["az"]]
        yield (timestamp, data)

# Load models
AUTOENCODER_MODEL_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\autoencoder_model.pth"
autoencoder_model = load_autoencoder_model(AUTOENCODER_MODEL_PATH)

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🛠️ Real-Time Predictive Maintenance Dashboard")

# Model params
INPUT_DIM = 3
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQ_LEN = 50
THRESHOLD = 0.5
SEVERE_THRESHOLD_LSTM = 0.9
SEVERE_THRESHOLD_AE = 0.15
MODEL_PATH = r"C:\Users\kouti\Python\(4) Real-Time Predictive Maintenance Dashboard\models\anomaly_model.pth"

if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")

@st.cache_resource
def load_model():
    model = AnomalyDetectionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Initialize session state
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=SEQ_LEN)
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'lstm_scores' not in st.session_state:
    st.session_state.lstm_scores = deque(maxlen=100)
if 'ae_scores' not in st.session_state:
    st.session_state.ae_scores = deque(maxlen=100)
if 'alerts' not in st.session_state:
    st.session_state.alerts = deque(maxlen=100)
if 'anomaly_log' not in st.session_state:
    st.session_state.anomaly_log = []
if 'data_stream_gen' not in st.session_state:
    st.session_state.data_stream_gen = stream_data()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    model_option = st.selectbox(
        "Detection Mode",
        options=["LSTM only", "Autoencoder only", "Both (OR)", "Both (AND)"],
        index=2
    )
    enable_notifications = st.checkbox("Desktop Alerts (Severe only)", value=False)
    st.progress(st.session_state.current_idx / 1000.0)  # Approximate until known dataset length

# Stream next data point
try:
    timestamp, next_reading = next(st.session_state.data_stream_gen)
    st.session_state.data_buffer.append(next_reading)
    st.session_state.current_idx += 1
except StopIteration:
    st.success("✅ End of data stream reached.")
    st.stop()

# Layout
col1, col2 = st.columns(2)
col1.metric("Current Reading - ax", f"{next_reading[0]:.3f}")
col1.metric("Current Reading - ay", f"{next_reading[1]:.3f}")
col1.metric("Current Reading - az", f"{next_reading[2]:.3f}")

# Inference
if len(st.session_state.data_buffer) == SEQ_LEN:
    input_window = np.array(st.session_state.data_buffer)
    input_tensor = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor).squeeze()
        lstm_score = torch.sigmoid(output).item()
        lstm_anomaly = lstm_score > THRESHOLD

    ae_score = compute_reconstruction_error(autoencoder_model, input_window)[0]
    ae_anomaly = is_anomalous(ae_score)

    st.session_state.lstm_scores.append(lstm_score)
    st.session_state.ae_scores.append(ae_score)

    if model_option == "LSTM only":
        alert = lstm_anomaly
    elif model_option == "Autoencoder only":
        alert = ae_anomaly
    elif model_option == "Both (OR)":
        alert = lstm_anomaly or ae_anomaly
    else:
        alert = lstm_anomaly and ae_anomaly

    st.session_state.alerts.append(int(alert))

    severity = None
    if alert:
        if lstm_score > SEVERE_THRESHOLD_LSTM or ae_score > SEVERE_THRESHOLD_AE:
            severity = "SEVERE"
        elif lstm_score > 0.7 or ae_score > 0.1:
            severity = "Moderate"
        else:
            severity = "Low"

        col2.markdown(f"### ⚠️ <span style='color:red'>Anomaly Detected!</span> Severity: **{severity}**", unsafe_allow_html=True)
        col2.markdown(f"Triggered by: **{model_option}**")

        st.session_state.anomaly_log.append({
            "timestamp": timestamp,
            "lstm_score": lstm_score,
            "ae_score": ae_score,
            "triggered_by": model_option,
            "severity": severity
        })

        if enable_notifications and severity == "SEVERE":
            try:
                from plyer import notification
                notification.notify(
                    title="🚨 Severe Anomaly Detected",
                    message=f"{model_option}: LSTM={lstm_score:.4f}, AE={ae_score:.4f}",
                    timeout=5
                )
            except Exception:
                st.warning("Desktop notifications not available.")
    else:
        col2.markdown("### ✅ Normal operation")

    with st.expander("📊 Live Anomaly Scores"):
        st.line_chart({
            "LSTM Score": list(st.session_state.lstm_scores),
            "Autoencoder Score": list(st.session_state.ae_scores),
            "Alert": list(st.session_state.alerts),
        })
else:
    st.info(f"Waiting for {SEQ_LEN - len(st.session_state.data_buffer)} more samples to begin detection...")

# Summary & CSV Export
st.markdown("---")
st.subheader("📈 Anomaly Summary")

if len(st.session_state.anomaly_log) > 0:
    df_summary = pd.DataFrame(st.session_state.anomaly_log)
    st.markdown(f"**Total Anomalies Detected:** {len(df_summary)}")
    st.dataframe(df_summary.tail(5), use_container_width=True)
    csv = df_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Anomaly Log as CSV",
        data=csv,
        file_name="anomaly_log.csv",
        mime="text/csv",
    )

    time_axis = list(range(st.session_state.current_idx - len(st.session_state.lstm_scores), st.session_state.current_idx))
    colors = []
    for i in range(len(st.session_state.alerts)):
        if st.session_state.alerts[i] == 0:
            colors.append('green')
        else:
            lstm_s = st.session_state.lstm_scores[i]
            ae_s = st.session_state.ae_scores[i]
            if lstm_s > SEVERE_THRESHOLD_LSTM or ae_s > SEVERE_THRESHOLD_AE:
                colors.append('red')
            elif lstm_s > 0.7 or ae_s > 0.1:
                colors.append('orange')
            else:
                colors.append('yellow')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=list(st.session_state.lstm_scores), mode='lines+markers', name='LSTM Score', line=dict(color='blue'), marker=dict(size=6, color=colors)))
    fig.add_trace(go.Scatter(x=time_axis, y=list(st.session_state.ae_scores), mode='lines+markers', name='Autoencoder Score', line=dict(color='purple'), marker=dict(size=6, color=colors, symbol='circle-open')))
    fig.update_layout(title="Anomaly Scores Over Time", xaxis_title="Window Index (Time)", yaxis_title="Anomaly Score", legend_title="Models", hovermode='x unified', height=400)
    st.plotly_chart(fig, use_container_width=True)

st_autorefresh(interval=1000, limit=None, key="datarefresh")