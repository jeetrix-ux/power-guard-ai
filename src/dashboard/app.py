import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os
import plotly.graph_objects as go
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.simulation.buck_converter import BuckConverterSimulator
from src.models.inference import HealthMonitor

# Page Config
st.set_page_config(
    page_title="PowerGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Industrial/Cyberpunk" look
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #00ADB5;
    }
    .metric-label {
        color: #AAAAAA;
        font-size: 0.9em;
    }
    .status-healthy {
        background-color: rgba(0, 255, 0, 0.1);
        border: 2px solid #00FF00;
        color: #00FF00;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .status-fault {
        background-color: rgba(255, 0, 0, 0.1);
        border: 2px solid #FF0000;
        color: #FF0000;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# Application State
if 'sim' not in st.session_state:
    st.session_state.sim = BuckConverterSimulator()
if 'monitor' not in st.session_state:
    st.session_state.monitor = HealthMonitor()
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame()
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

# Sidebar Controls
st.sidebar.title("⚡ System Controls")

st.sidebar.subheader("Operating Conditions")
v_in_set = st.sidebar.slider("Input Voltage (V)", 15.0, 30.0, 24.0)
duty_set = st.sidebar.slider("Duty Cycle", 0.1, 0.9, 0.5)
load_set = st.sidebar.slider("Load Resistance (Ω)", 1.0, 20.0, 10.0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Fault Injection")
fault_type = st.sidebar.selectbox("Inject Fault", ["None", "Capacitor Aging (ESR High)", "MOSFET Wear (Rds High)", "Thermal Cooling Fail"])

# Apply Faults
esr_mult = 1.0
rds_mult = 1.0
c_mult = 1.0
cooling_fail = False

if fault_type == "Capacitor Aging (ESR High)":
    esr_mult = 5.0
    c_mult = 0.7
elif fault_type == "MOSFET Wear (Rds High)":
    rds_mult = 3.0
elif fault_type == "Thermal Cooling Fail":
    cooling_fail = True

# Update Sim State
st.session_state.sim.set_degradation(esr_mult, c_mult, rds_mult)
if cooling_fail:
    # Hack to simulate cooling fail: Increase thermal resistance dynamically
    st.session_state.sim.params.R_thermal = 20.0 # High resistance
else:
    st.session_state.sim.params.R_thermal = 5.0 # Normal

# Main Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.title("PowerGuard AI™ Monitor")
    st.markdown("_Predictive Maintenance for Power Electronics_")

with col2:
    if st.button("START / STOP Simulation", type="primary"):
        st.session_state.simulation_running = not st.session_state.simulation_running

# Live Dashboard
placeholder = st.empty()

while st.session_state.simulation_running:
    # Run 1 Step (representing 0.1s of data for UI smoothness)
    # We step 10 times internally to get 10 datapoints
    batch = []
    for _ in range(10):
        data = st.session_state.sim.step(v_in_set, duty_set, load_set, 0.01) # 100Hz
        batch.append(data)
    
    # Process the last one for "Live" reading, but push all to buffer
    latest_data = batch[-1]
    
    # Run Inference on the stream
    ml_result = {"status": "Initializing...", "color": "gray"}
    for d in batch:
        res = st.session_state.monitor.process_stream(d)
        if res:
            ml_result = res
            
    # Update History
    new_rows = pd.DataFrame(batch)
    st.session_state.history = pd.concat([st.session_state.history, new_rows]).tail(200)
    
    # Render UI
    with placeholder.container():
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-value'>{latest_data['V_out']:.2f}V</div><div class='metric-label'>Output</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-value'>{latest_data['I_load']:.2f}A</div><div class='metric-label'>Current</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-value'>{latest_data['Temp_heatsink']:.1f}°C</div><div class='metric-label'>Temp</div></div>", unsafe_allow_html=True)
        
        # Health Status
        status_color = "status-healthy" if ml_result['color'] == 'green' else "status-fault"
        status_text = ml_result['status']
        if ml_result.get('fault_type') and ml_result['fault_type'] != 'None':
            status_text += f": {ml_result['fault_type']}"
            
        m4.markdown(f"<div class='{status_color}'>{status_text} <br> <small>{ml_result.get('confidence', '')}</small></div>", unsafe_allow_html=True)

        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Voltage Output & Ripple")
            chart_data = st.session_state.history[['timestamp', 'V_out', 'V_in']]
            st.line_chart(chart_data.set_index('timestamp'))
            
        with chart_col2:
            st.subheader("Thermal Profile")
            st.line_chart(st.session_state.history[['timestamp', 'Temp_heatsink']].set_index('timestamp'))
            
        # Log
        if ml_result['color'] == 'red':
            st.error(f"ALERT: {ml_result['fault_type']} detected! Anomaly Score: {ml_result['anomaly_score']}")
            
    time.sleep(0.1)

if not st.session_state.simulation_running:
    st.info("Simulation Paused. Press START to begin.")
