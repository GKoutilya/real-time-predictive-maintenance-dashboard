import streamlit as st
import numpy as np
import random
from streamlit_autorefresh import st_autorefresh
from sensor_streamer import generate_sensor_reading
from anomaly_detector import RunningStats, detect_anomalies

st.title("Real-Time Predictive Maintenance Dashboard")

# Auto-refresh every 500 ms, limit to 100 refreshes
count = st_autorefresh(interval=500, limit=100, key="datarefresh")

# Initialize running stats once and keep in session state
if 'running_stats' not in st.session_state:
    st.session_state.running_stats = RunningStats(num_sensors=5)
running_stats = st.session_state.running_stats

# Generate sensor reading
reading = generate_sensor_reading()

# Inject anomaly with 5% chance before updating stats
if random.random() < 0.05:
    anomaly_index = random.randint(0, 4)
    anomaly_value = random.uniform(10, 15)
    reading[anomaly_index] = anomaly_value

# Update running stats with this reading (including anomaly)
running_stats.update(reading)

# Detect anomalies
anomaly_flags = detect_anomalies(reading, running_stats)

st.markdown(f"**Reading #{count}**")

# Display each sensor reading, coloring anomalous ones red
for i, value in enumerate(reading):
    if anomaly_flags[i]:
        st.markdown(f"Sensor {i}: <span style='color:red'>{value:.2f} ⚠️ Anomaly</span>", unsafe_allow_html=True)
    else:
        st.write(f"Sensor {i}: {value:.2f}")
