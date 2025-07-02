# Real-Time Predictive Maintenance Dashboard

---

## Overview

Welcome to the **Real-Time Predictive Maintenance Dashboard**, a cutting-edge tool designed to empower industrial IoT and smart manufacturing by detecting anomalies in streaming sensor data — specifically accelerometer signals — using advanced deep learning techniques. This dashboard harnesses the power of **hybrid anomaly detection models**, combining the temporal strength of Long Short-Term Memory (LSTM) networks with the reconstruction capabilities of Autoencoders to identify early signs of machine failure.

By proactively detecting abnormal behavior in equipment, this project aims to minimize downtime, optimize maintenance schedules, and ultimately save operational costs.

---

## Key Features

- **Real-Time Sensor Data Simulation**  
  Generate realistic, continuous accelerometer data streams mimicking actual machine sensor output.

- **Sliding Window Preprocessing**  
  Efficiently window and normalize time-series data for seamless model input.

- **Hybrid Anomaly Detection System**  
  - *LSTM Model*: Captures temporal dependencies to identify anomalies in sequences.  
  - *Autoencoder*: Detects anomalies via reconstruction errors, complementing LSTM insights.

- **Synthetic Anomaly Injection**  
  Inject controlled, realistic anomalies into validation data, enabling robust model evaluation and stress testing.

- **Comprehensive Evaluation Metrics**  
  Analyze precision, recall, F1-score, ROC, and Precision-Recall curves to ensure model reliability.

- **Interactive Streamlit Dashboard**  
  Visualize live sensor streams, anomaly scores, and receive instant alerts—all in a user-friendly interface.

- **Smart Alerting Mechanism**  
  Configurable desktop notifications and optional SMS alerts for critical anomalies, balancing promptness and intrusiveness.

- **Multi-Modal Data Synchronization**  
  Align accelerometer readings with RGB camera data for enriched context and fault diagnosis.

---

## Technology Stack

- **Programming:** Python 3.10+  
- **Deep Learning:** PyTorch  
- **Web UI:** Streamlit  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Evaluation:** Scikit-learn  
- **Utilities:** Welford’s algorithm, SMS APIs (optional)  

---

## Getting Started

### Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/predictive-maintenance-dashboard.git
cd predictive-maintenance-dashboard
````

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the Dashboard

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL in your browser to explore live sensor data visualizations, model predictions, and alert notifications.

---

## Configuration

* **Sliding Window Size:** Default 50 timesteps (adjustable in `data_preprocessor.py`)
* **Anomaly Thresholds:** Tune detection sensitivity in `config.py`
* **Alert Preferences:** Enable/disable desktop or SMS alerts as needed
* **Data Synchronization:** Configure timing offsets for accelerometer and RGB data alignment

---

## Dashboard Highlights

* **Dynamic Time-Series Plots:** Real-time visualization of sensor data streams.
* **Anomaly Score Overlays:** Separate displays for LSTM and Autoencoder confidence levels.
* **Interactive Controls:** Pause/resume streaming, adjust thresholds, and switch between models seamlessly.
* **Non-Intrusive Alerts:** Smart notifications keep you informed without disruption.

---

## Project Structure

```
predictive-maintenance-dashboard/
├── app.py                      # Streamlit dashboard application  
├── data_preprocessor.py        # Sensor data ingestion, preprocessing, and synchronization  
├── maintenance_model.py        # LSTM and Autoencoder model architectures and inference  
├── anomaly_injection.py        # Synthetic anomaly insertion for evaluation  
├── evaluate_model.py           # Scripts for model validation and metrics computation  
├── utils.py                    # Helper utilities including alerting and plotting  
├── requirements.txt            # Python dependencies  
├── README.md                   # Project documentation  
└── data/                       # Dataset storage (raw and processed)  
```

---

## Evaluation & Performance

Both models have undergone rigorous training and validation on historical accelerometer datasets. Key evaluation results include:

* Confusion matrices illustrating true vs. predicted anomalies
* Precision, Recall, and F1-Score metrics for balanced performance insights
* ROC and Precision-Recall curves showcasing model discrimination power
* Enhanced validation realism via injected synthetic anomalies

Results are logged and visualized through the dashboard and detailed CSV reports, supporting continuous improvement.

---

## Future Directions

* **LLM-Powered Chatbot Integration:** Intelligent fault diagnosis and maintenance guidance.
* **Expanded Alerting Channels:** Email and SMS escalation workflows.
* **Multi-Sensor Fusion:** Incorporate vibration, temperature, and other sensor modalities.
* **Explainability:** Model interpretability techniques tailored for sequential data.
* **Edge Deployment:** Optimize models for IoT device inference.
* **API Development:** RESTful endpoints for system interoperability.
* **Advanced Analytics:** Historical data trends and customizable alert rules for proactive maintenance.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

**Author:** Koutilya Ganapathiraju

**Email:** [gkoutilyaraju@gmail.com]

**GitHub:** [https://github.com/GKoutilya]

**LinkedIn:** [www.linkedin.com/in/koutilya-ganapathiraju-0a3350182]

---

Thank you for exploring the Real-Time Predictive Maintenance Dashboard. Your feedback and contributions are welcome!