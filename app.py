import streamlit as st
import numpy as np
import random
from streamlit_autorefresh import st_autorefresh

st.title("Real-Time Predictive Maintenance Dashboard")

# Auto-refresh every 500 ms (0.5 seconds), limit to 100 refreshes
count = st_autorefresh(interval=500, limit=100, key="datarefresh")

placeholder = st.empty()

def generate_sensor_reading(num_sensors=5):
    readings = np.random.normal(loc=0, scale=1, size=num_sensors)
    return readings.tolist()

reading = generate_sensor_reading()

# Inject anomaly with 5% chance
if random.random() < 0.05:
    anomaly_index = random.randint(0, 4)
    anomaly_value = random.uniform(10, 15)
    reading[anomaly_index] = anomaly_value
    anomaly_msg = f"*** Anomaly at sensor {anomaly_index} with value {anomaly_value:.2f} ***"
else:
    anomaly_msg = ""

placeholder.markdown(f"**Reading #{count}**")
placeholder.write(reading)

if anomaly_msg:
    placeholder.markdown(f"<span style='color:red'>{anomaly_msg}</span>", unsafe_allow_html=True)
