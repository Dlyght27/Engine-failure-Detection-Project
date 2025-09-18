🚗 Engine Failure Detection System (Predictive Maintenance)

A real-time predictive maintenance application that monitors automobile engine health and predicts potential faults before they escalate.
Built using Python, Streamlit, FastAPI, scikit-learn, and Plotly.

📌 Features

Live Telemetry API (FastAPI): Simulates or fetches real engine data (RPM, temperature, torque, vibrations, etc.).

Streamlit Dashboard: Interactive UI for real-time monitoring, predictions, and log inspection.

Machine Learning Model: Predicts engine fault conditions (Normal → Critical).

Visualization: Trend charts, fault distribution plots, and dataset row inspection.

Logging: Stores all predictions with timestamps for later analysis.

⚙️ Tech Stack

Frontend/UI: Streamlit + Plotly

Backend/API: FastAPI + Uvicorn

ML/Analytics: scikit-learn, pandas, numpy

Persistence: CSV log files

Deployment: Localhost (can be extended to cloud)

🚀 Getting Started
1. Clone Repo
git clone https://github.com/yourusername/predictive-maintenance-app.git
cd predictive-maintenance-app

2. Install Dependencies
pip install -r requirements.txt

3. Run App
streamlit run predictive_maintenance_app.py


The FastAPI service runs in the background (http://127.0.0.1:8000/telemetry).

📂 Project Structure
├── predictive_maintenance_app.py   # Main Streamlit + FastAPI app
├── Engine_model.pkl                # Trained ML model
├── Engine_failure_features_name.pkl # Feature names used in training
├── engine_failure_features.csv     # Dataset
├── maintenance_log.csv             # Auto-generated logs
└── README.md                       # Project documentation

📊 Usage Modes

Manual Input: Enter temperature, RPM, torque, etc. → model predicts fault level.

Simulation Mode: Uses API to generate live telemetry data.

View Logs: Inspect logs, visualize fault distribution, and download records.

⚠️ Limitations

Dataset was limited, with few fault condition classes.

Accuracy of the ML model wasn’t strong despite feature engineering, cross-validation, and hyperparameter tuning.

Model generalization is limited; requires a richer dataset for production use.

📚 Learning Curve

Learned the importance of high-quality, diverse data in ML projects.

Gained hands-on experience with APIs for real-time data streaming.

Discovered how feature relationships from EDA influence model performance.

Understood the role of stratified sampling in handling imbalanced datasets.

🔮 Future Improvements

Collect larger and more diverse real-world datasets (via OBD-II car sensors).

Explore deep learning for multivariate time series fault detection.

Deploy as a cloud-based monitoring system with alert notifications (SMS/Email).

🏷️ Tags

#PredictiveMaintenance #MachineLearning #IoT #FastAPI #Streamlit #DataScience
