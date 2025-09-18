# predictive_maintenance_app.py
import random
import math
import threading
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
from sklearn.metrics.pairwise import euclidean_distances
from fastapi import FastAPI
import uvicorn
import plotly.express as px

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(page_title="Engine Failure Detection System", layout="centered", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        .block-container { max-width: 900px; padding-top: 1rem; padding-bottom: 1rem; }
        h1 { font-size: 1.8rem; }
        h2 { font-size: 1.4rem; }
        h3 { font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

# ==============================
# FASTAPI MOCK ENGINE TELEMETRY API
# ==============================
api = FastAPI()
engine_state = {"time": 0, "temperature": 25.0, "rpm": 800, "mode": "Idle"}

@api.get("/telemetry")
def get_telemetry():
    engine_state["time"] += 1
    if engine_state["time"] % 20 == 0:
        engine_state["mode"] = random.choice(["Idle", "Cruising", "Heavy Load"])
    mode = engine_state["mode"]

    if mode == "Idle":
        rpm = 800 + random.randint(-50, 50)
    elif mode == "Cruising":
        rpm = 2000 + int(200 * math.sin(engine_state["time"] / 5)) + random.randint(-50, 50)
    else:
        rpm = 3000 + int(300 * math.sin(engine_state["time"] / 4)) + random.randint(-100, 100)
    engine_state["rpm"] = max(700, rpm)

    if engine_state["temperature"] < 90:
        engine_state["temperature"] += 0.5
    else:
        engine_state["temperature"] += random.uniform(-0.2, 0.2)

    if mode == "Idle":
        fuel_eff = random.uniform(10, 15)
    elif mode == "Cruising":
        fuel_eff = random.uniform(15, 20)
    else:
        fuel_eff = random.uniform(5, 10)

    torque = round(engine_state["rpm"] * 0.1 + random.uniform(-10, 10), 2)
    power = round(torque * engine_state["rpm"] / 9550, 2)

    if mode == "Idle":
        vib_x, vib_y, vib_z = random.uniform(0.1, 0.3), random.uniform(0.1, 0.3), random.uniform(0.2, 0.4)
    elif mode == "Cruising":
        vib_x, vib_y, vib_z = random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.3, 0.6)
    else:
        vib_x, vib_y, vib_z = random.uniform(0.4, 0.8), random.uniform(0.4, 0.8), random.uniform(0.5, 1.2)

    return {
        "Temperature": round(engine_state["temperature"], 2),
        "RPM": engine_state["rpm"],
        "Fuel_Efficiency": round(fuel_eff, 2),
        "Torque": torque,
        "Power_Output": power,
        "Vibration_X": round(vib_x, 3),
        "Vibration_Y": round(vib_y, 3),
        "Vibration_Z": round(vib_z, 3),
        "Operational_Mode": mode
    }

def run_api():
    uvicorn.run(api, host="127.0.0.1", port=8000, log_level="error")

threading.Thread(target=run_api, daemon=True).start()

# ==============================
# LOAD MODEL & DATA
# ==============================
# ==============================
# APP HEADER & LANDING SECTION (TEXT FIRST, WHITE FOR DARK THEME)
# ==============================
st.markdown(
    """
    <h1 style='text-align: center; color: #ffffff;'>🚗 Predictive Maintenance App</h1>
    <h4 style='text-align: center; color: #ffffff;'>
        Monitor engine <span style='color:#1f77b4; font-weight:bold;'>real-time</span> telemetry, 
        detect <span style='color:#ff7f0e; font-weight:bold;'>faults early</span>, 
        and take <span style='color:#2ca02c; font-weight:bold;'>preventive actions</span> 
        to keep your vehicle running smoothly.
    </h4>
    <p style='text-align: center; color: #dddddd; font-size: 0.9rem;'>
        Track <span style='font-weight:bold;'>temperature</span>, 
        <span style='font-weight:bold;'>RPM</span>, 
        <span style='font-weight:bold;'>torque</span>, 
        <span style='font-weight:bold;'>power output</span>, 
        and <span style='font-weight:bold;'>operational modes</span> 
        while receiving actionable maintenance insights.
    </p>
    """, unsafe_allow_html=True
)

st.image(
    'https://imgs.search.brave.com/pQHLIuNTynwY6ih87welTwyY_Z-AnxA2TIoIf0cfAiY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9yaXNs/b25lLmNvbS93cC1j/b250ZW50L3VwbG9h/ZHMvMjAyMy8wNC9I/UEMxMDAtYmxvZy1m/ZWF0dXJlZC1pbWFn/ZS53ZWJw',
    caption="Automobile Engine",
    use_container_width=True
)


model = joblib.load("Engine_model.pkl")
feature_names = joblib.load("Engine_failure_features_name.pkl")
dataset = pd.read_csv("engine_failure_features.csv")
log_file = "maintenance_log.csv"

# ==============================
# FAULT LEVELS & STATUS
# ==============================
fault_levels = {
    0: {"msg": "✅ Engine operating normally.", "type": "success"},
    1: {"msg": "⚠️ Minor fault detected. Please monitor engine closely.", "type": "warning"},
    2: {"msg": "🚨 Major fault detected! Schedule maintenance soon.", "type": "error"},
    3: {"msg": "❌ CRITICAL FAULT! Stop engine immediately!", "type": "error"}
}

status_labels = {0: "Normal", 1: "Minor Fault", 2: "Major Fault", 3: "Critical Fault"}
fault_colors = {0: "green", 1: "orange", 2: "red", 3: "darkred"}

# ==============================
# DYNAMIC FAULT MESSAGE GENERATOR
# ==============================
def generate_fault_message(prediction, temp, vibration, torque=None, power_output=None):
    msg = ""
    if prediction == 0:
        msg = "✅ Engine operating normally."
    elif prediction == 1:
        msg = "⚠️ Minor fault detected."
        if temp > 95: msg += f" Engine temperature slightly high ({temp}°C)."
        if vibration > 0.5: msg += f" Vibration slightly elevated ({vibration})."
    elif prediction == 2:
        msg = "🚨 Major fault detected!"
        if temp > 100: msg += f" Engine temperature critical ({temp}°C)."
        if vibration > 0.7: msg += f" Vibration above normal ({vibration})."
        if torque and power_output: msg += f" Torque/power below expected (Torque={torque}, Power={power_output})."
    elif prediction == 3:
        msg = "❌ CRITICAL FAULT! Stop engine immediately!"
        if temp > 110: msg += f" Engine severely overheated ({temp}°C)."
        if vibration > 1.0: msg += f" Extreme vibration detected ({vibration})."
        if torque and power_output: msg += f" Severe power loss (Torque={torque}, Power={power_output})."
    return msg

# ==============================
# DISPLAY FUNCTIONS
# ==============================
def display_fault(prediction, temp, vibration, torque, power_output):
    dynamic_msg = generate_fault_message(prediction, temp, vibration, torque, power_output)
    msg_type = fault_levels.get(prediction, {"type": "info"})["type"]
    getattr(st, msg_type)(dynamic_msg)  # only display the generated message

def display_fault_badge(prediction):
    status = status_labels.get(prediction, "Unknown")
    color = fault_colors.get(prediction, "gray")
    st.markdown(f"<h3 style='color:{color};'>{status}</h3>", unsafe_allow_html=True)

# ==============================
# LOGGING FUNCTION
# ==============================
COLUMNS = [
    "Temperature (°C)", "RPM", "Fuel_Efficiency",
    "Vibration_X", "Vibration_Y", "Vibration_Z",
    "Torque", "Power_Output (kW)", "Operational_Mode",
    "Prediction", "Dataset_Row_Index", "Timestamp"
]

def save_log(temp, rpm, fuel_eff, vib_x, vib_y, vib_z, torque, power_output,
             op_mode, prediction, source_idx=None):
    new_entry = pd.DataFrame([{
        "Temperature (°C)": round(temp, 2), "RPM": rpm, "Fuel_Efficiency": round(fuel_eff, 2),
        "Vibration_X": round(vib_x, 2), "Vibration_Y": round(vib_y, 2), "Vibration_Z": round(vib_z, 2),
        "Torque": round(torque, 2), "Power_Output (kW)": round(power_output, 2),
        "Operational_Mode": op_mode, "Prediction": int(prediction),
        "Dataset_Row_Index": source_idx, "Timestamp": pd.Timestamp.now()
    }], columns=COLUMNS)
    if os.path.exists(log_file):
        new_entry.to_csv(log_file, mode="a", header=False, index=False)
    else:
        new_entry.to_csv(log_file, mode="w", header=True, index=False)

# ==============================
# MANUAL INPUT MODE
# ==============================
def manual_input():
    st.subheader("✍️ Enter Engine Data")
    temp = st.number_input("Engine Temperature (°C)", 17.0, 120.0, 90.0, step=10.0)
    rpm = st.number_input("Engine RPM", 800.000, 4000.000, 2500.000, step=100.000)
    fuel_eff = st.number_input("Fuel Efficiency (km/l)", 4.0, 30.0, 22.0, step=2.0)
    torque = st.number_input("Torque (Nm)", 40.0, 200.0, 120.0, step=20.0)
    power_output = st.number_input("Power Output (kW)", 20.0, 100.0, 50.0, step=10.0)
    op_mode = st.selectbox("Operational Mode", ["Idle", "Cruising", "Heavy Load"])

    if st.button("🔍 Predict Fault Condition"):
        features = dataset[["Temperature (°C)", "RPM", "Fuel_Efficiency", "Torque", "Power_Output (kW)"]]
        distances = euclidean_distances(features, [[temp, rpm, fuel_eff, torque, power_output]]).flatten()
        row_idx = np.argmin(distances)
        vib_x, vib_y, vib_z = dataset.iloc[row_idx][["Vibration_X", "Vibration_Y", "Vibration_Z"]]

        input_data = pd.DataFrame([[temp, rpm, fuel_eff, vib_x, vib_y, vib_z, torque, power_output,
                                    1 if op_mode == "Cruising" else 0,
                                    1 if op_mode == "Heavy Load" else 0,
                                    1 if op_mode == "Idle" else 0]], columns=feature_names)

        prediction = model.predict(input_data)[0]
        vibration_avg = (vib_x + vib_y + vib_z)/3
        display_fault_badge(prediction)
        display_fault(prediction, temp, vibration_avg, torque, power_output)
        save_log(temp, rpm, fuel_eff, vib_x, vib_y, vib_z, torque, power_output, op_mode, prediction, row_idx)

# ==============================
# SIMULATION MODE
# ==============================
def simulation_mode():
    st.subheader("📡 Real-Time Data Simulation")
    run_simulation = st.checkbox("Start Simulation")

    if "simulation_active" not in st.session_state:
        st.session_state.simulation_active = False
    if "stop_reason" not in st.session_state:
        st.session_state.stop_reason = ""
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.stop_reason:
        st.info(st.session_state.stop_reason)
        if st.button("🔄 Reset Simulation"):
            st.session_state.simulation_active = True
            st.session_state.stop_reason = ""
            st.session_state.history = []
    else:
        st.session_state.simulation_active = run_simulation

    if st.session_state.simulation_active:
        placeholder = st.empty()
        for _ in range(1000):
            try:
                data = requests.get("http://127.0.0.1:8000/telemetry", timeout=2).json()
            except:
                data = {"Temperature": np.random.uniform(70,120),
                        "RPM": np.random.randint(800,4000),
                        "Fuel_Efficiency": np.random.uniform(5,20),
                        "Torque": np.random.uniform(100,400),
                        "Power_Output": np.random.uniform(50,300),
                        "Vibration_X": np.random.uniform(0.1,1.0),
                        "Vibration_Y": np.random.uniform(0.1,1.0),
                        "Vibration_Z": np.random.uniform(0.1,1.5),
                        "Operational_Mode": random.choice(["Idle","Cruising","Heavy Load"])}

            temp, rpm, fuel_eff, torque, power_output = data["Temperature"], data["RPM"], data["Fuel_Efficiency"], data["Torque"], data["Power_Output"]
            vib_x, vib_y, vib_z, op_mode = data["Vibration_X"], data["Vibration_Y"], data["Vibration_Z"], data["Operational_Mode"]
            vibration_avg = (vib_x + vib_y + vib_z)/3

            # Correctly map operational mode to feature columns
            op_dict = {"Idle":0,"Cruising":0,"Heavy Load":0}
            op_dict[op_mode] = 1
            input_data = pd.DataFrame([[temp,rpm,fuel_eff,vib_x,vib_y,vib_z,torque,power_output,
                                        op_dict.get("Cruising",0),
                                        op_dict.get("Heavy Load",0),
                                        op_dict.get("Idle",0)]],
                                      columns=feature_names)
            prediction = model.predict(input_data)[0]

            save_log(temp,rpm,fuel_eff,vib_x,vib_y,vib_z,torque,power_output,op_mode,prediction)

            st.session_state.history.append({"Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                                             "Temp (°C)": round(temp,2),"RPM": rpm,"Mode": op_mode,
                                             "Prediction": status_labels.get(prediction,"Other")})
            st.session_state.history = st.session_state.history[-5:]

            with placeholder.container():
                st.metric("🌡️ Temperature (°C)", round(temp,2))
                st.metric("⚙️ RPM", rpm)
                st.metric("⛽ Fuel Efficiency (km/l)", round(fuel_eff,2))
                st.metric("🔧 Torque (Nm)", round(torque,2))
                st.metric("⚡ Power Output (kW)", round(power_output,2))
                st.write(f"🚗 Operational Mode: **{op_mode}**")
                display_fault_badge(prediction)
                display_fault(prediction,temp,vibration_avg,torque,power_output)
                st.subheader("🕒 Recent Simulation History")
                st.table(pd.DataFrame(st.session_state.history))

            if prediction == 3:
                st.warning("Simulation stopped due to CRITICAL FAULT! 🚨")
                st.session_state.simulation_active = False
                st.session_state.stop_reason = "Simulation stopped due to CRITICAL FAULT. Press reset to continue."
                break

            time.sleep(2)

# ==============================
# VIEW LOGS
# ==============================
def view_logs():
    st.subheader("📊 Prediction Logs")
    if os.path.exists(log_file):
        try:
            log_data = pd.read_csv(log_file)
            if "Prediction" in log_data.columns:
                log_data["Prediction"]=pd.to_numeric(log_data["Prediction"],errors="coerce")
                log_data=log_data.dropna(subset=["Prediction"])
                log_data["Prediction"]=log_data["Prediction"].astype(int)
                log_data["Status"]=log_data["Prediction"].map(status_labels)
            if log_data.empty:
                st.info("ℹ️ No status records available yet. Run predictions or simulations first.")
            else:
                st.dataframe(log_data.tail(50))

                # Status distribution
                if "Status" in log_data.columns:
                    status_counts=log_data["Status"].value_counts().reset_index()
                    status_counts.columns=["Status","Count"]
                    fig=px.bar(status_counts,x="Status",y="Count",text="Count",color="Status",title="Status Distribution",template="plotly_white")
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig,use_container_width=True)

                # Telemetry trends
                if "Timestamp" in log_data.columns:
                    numeric_cols = log_data.select_dtypes(include=["number"]).columns.tolist()
                    if numeric_cols:
                        selected_feature = st.selectbox("Choose a feature to visualize",numeric_cols)
                        fig2=px.line(log_data,x="Timestamp",y=selected_feature,title=f"{selected_feature} Over Time",template="plotly_white")
                        st.plotly_chart(fig2,use_container_width=True)

                # ✅ Dataset Row Inspection & Download
                st.subheader("🔎 Inspect Original Dataset Row")
                if "Dataset_Row_Index" in log_data.columns:
                    available_indices = log_data["Dataset_Row_Index"].dropna().unique()
                    if len(available_indices) > 0:
                        selected_idx = st.selectbox("Select Dataset Row Index", available_indices)
                        row_data = dataset.iloc[int(selected_idx)].to_dict()
                        st.json(row_data)
                        st.write("📋 Dataset Row as Table")
                        st.dataframe(pd.DataFrame([row_data]))
                    else:
                        st.info("No dataset rows used yet.")
                else:
                    st.info("Dataset row index not found in logs.")

                st.download_button(
                    "📥 Download Logs",
                    data=log_data.to_csv(index=False),
                    file_name="maintenance_log.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error reading log file: {e}")
    else:
        st.info("ℹ️ No log file found yet.")

# ==============================
# MAIN
# ==============================
mode=st.sidebar.radio("Select Mode",["Manual Input","Simulation Mode","View Logs"])
if mode=="Manual Input": manual_input()
elif mode=="Simulation Mode": simulation_mode()
elif mode=="View Logs": view_logs()
