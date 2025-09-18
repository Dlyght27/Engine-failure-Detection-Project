# 🚗 Engine Failure Detection System (Predictive Maintenance)

A real-time **predictive maintenance application** that monitors automobile engine health and predicts potential faults before they escalate.  
Built using **Python, Streamlit, FastAPI, scikit-learn, and Plotly**.

---

## 📌 Features
- **Live Telemetry API (FastAPI):** Simulates or fetches real engine data (RPM, temperature, torque, vibrations, etc.).  
- **Streamlit Dashboard:** Interactive UI for real-time monitoring, predictions, and log inspection.  
- **Machine Learning Model:** Predicts engine fault conditions (Normal → Critical).  
- **Visualization:** Trend charts, fault distribution plots, and dataset row inspection.  
- **Logging:** Stores all predictions with timestamps for later analysis.  

---

## ⚙️ Tech Stack
- **Frontend/UI:** Streamlit + Plotly  
- **Backend/API:** FastAPI + Uvicorn  
- **ML/Analytics:** scikit-learn, pandas, numpy  
- **Persistence:** CSV log files  
- **Deployment:** Localhost (can be extended to cloud)  

---

├── predictive_maintenance_app.py    # Main Streamlit + FastAPI app
├── Engine_model.pkl                 # Trained ML model
├── Engine_failure_features_name.pkl # Feature names used in training
├── engine_failure_features.csv      # Dataset
├── maintenance_log.csv              # Auto-generated logs
└── README.md                        # Project documentation
