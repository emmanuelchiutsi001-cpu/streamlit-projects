# app.py - Complete Working System with Dark Theme
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import os
import random

# Page config with dark theme
st.set_page_config(
    page_title="Industrial Predictive Maintenance System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS with EXTREMELY BLACK sidebar text
st.markdown("""
<style>
    /* Dark theme background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* Main container */
    .main {
        background: rgba(0,0,0,0.3);
        padding: 1rem;
        border-radius: 10px;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Metric cards */
    .metric-card {
        background: rgba(30, 30, 46, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        border-left: 5px solid #ff0000;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        animation: pulse 1.5s infinite;
        color: white;
        box-shadow: 0 4px 15px rgba(255,0,0,0.3);
    }

    .alert-warning {
        background: linear-gradient(135deg, #ffa500 0%, #ff6b35 100%);
        border-left: 5px solid #ff8c00;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(255,165,0,0.3);
    }

    .alert-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-left: 5px solid #00f2fe;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        color: white;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); box-shadow: 0 6px 20px rgba(255,0,0,0.5); }
        100% { transform: scale(1); }
    }

    /* Recommendation cards */
    .recommendation-card {
        background: rgba(40, 40, 58, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }

    .recommendation-card:hover {
        background: rgba(50, 50, 70, 0.9);
        transform: translateX(5px);
    }

    /* Status indicators */
    .status-healthy {
        color: #4caf50;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(76,175,80,0.5);
    }

    .status-warning {
        color: #ff9800;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255,152,0,0.5);
    }

    .status-critical {
        color: #f44336;
        font-weight: bold;
        animation: blink 1s infinite;
        text-shadow: 0 0 10px rgba(244,67,54,0.5);
    }

    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }

    /* STOP MACHINE BUTTON STYLING - RED EMERGENCY */
    .stButton > button:has(.stop-button-text) {
        background: linear-gradient(135deg, #ff0000 0%, #8b0000 100%);
        font-size: 1.2rem;
        padding: 0.8rem 1.5rem;
        animation: pulse-red 1s infinite;
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255,0,0,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255,0,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,0,0,0); }
    }

    /* Dataframe styling */
    .dataframe {
        background: rgba(0,0,0,0.5);
        color: white;
        border-radius: 10px;
    }

    /* Sidebar styling - EXTREMELY BLACK */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #000000 !important;
        background-color: #000000 !important;
    }

    /* Sidebar text - EXTREMELY BLACK background with white text for contrast */
    .css-1d391kg .stMarkdown, 
    .css-1d391kg label, 
    .css-1d391kg p, 
    .css-1d391kg h1, 
    .css-1d391kg h2, 
    .css-1d391kg h3,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        background: #000000 !important;
    }

    /* Sidebar radio buttons */
    .css-1d391kg .stRadio > div, 
    [data-testid="stSidebar"] .stRadio > div {
        background: #000000 !important;
    }

    .css-1d391kg .stRadio label, 
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        background: #000000 !important;
    }

    /* Sidebar selectbox */
    .css-1d391kg .stSelectbox > div, 
    [data-testid="stSidebar"] .stSelectbox > div {
        background: #1a1a1a !important;
    }

    .css-1d391kg .stSelectbox label, 
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
    }

    /* Sidebar expander */
    .css-1d391kg .streamlit-expanderHeader,
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #ffffff !important;
        background: #000000 !important;
    }

    /* Sidebar number input */
    .css-1d391kg .stNumberInput input,
    [data-testid="stSidebar"] .stNumberInput input {
        background: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* Text styling */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ffffff !important;
    }

    /* Input fields */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background: rgba(30, 30, 46, 0.9);
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(30, 30, 46, 0.9);
        color: white;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-size: 2rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }

    /* Alert overlay for critical state */
    .critical-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(244, 67, 54, 0.1);
        pointer-events: none;
        z-index: 999;
        animation: pulse-bg 2s infinite;
    }

    @keyframes pulse-bg {
        0% { background: rgba(244, 67, 54, 0); }
        50% { background: rgba(244, 67, 54, 0.2); }
        100% { background: rgba(244, 67, 54, 0); }
    }

    /* Machine stopped overlay */
    .machine-stopped-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(8px);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }

    .machine-stopped-content {
        text-align: center;
        background: linear-gradient(135deg, #8b0000 0%, #ff0000 100%);
        padding: 3rem;
        border-radius: 20px;
        border: 3px solid #ff6666;
        animation: pulse-border 1s infinite;
    }

    @keyframes pulse-border {
        0% { border-color: #ff6666; box-shadow: 0 0 0 0 rgba(255,102,102,0.7); }
        70% { border-color: #ff0000; box-shadow: 0 0 0 20px rgba(255,0,0,0); }
        100% { border-color: #ff6666; box-shadow: 0 0 0 0 rgba(255,102,102,0); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'critical_mode' not in st.session_state:
    st.session_state.critical_mode = False
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = None
if 'email_sent_for_risk' not in st.session_state:
    st.session_state.email_sent_for_risk = False
if 'email_config' not in st.session_state:
    st.session_state.email_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender': 'emmanuelchiutsi001@gmail.com',
        'recipient': 'emmanuelchiutsi001@gmail.com',
        'password': 'twml rauq erkv xark'
    }
if 'machine_stopped' not in st.session_state:
    st.session_state.machine_stopped = False
if 'stop_machine_time' not in st.session_state:
    st.session_state.stop_machine_time = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'custom_dataset_loaded' not in st.session_state:
    st.session_state.custom_dataset_loaded = False


# ============================================================================
# DATABASE SETUP
# ============================================================================
def init_database():
    """Initialize SQLite database for storing sensor data and alerts"""
    conn = sqlite3.connect('industrial_monitoring.db')
    c = conn.cursor()

    # Sensor data table
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  temperature REAL,
                  vibration REAL,
                  pressure REAL,
                  rpm REAL,
                  current REAL,
                  voltage REAL,
                  health_score REAL,
                  predicted_ttf REAL,
                  failure_probability REAL,
                  alert_triggered BOOLEAN)''')

    # Alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  alert_type TEXT,
                  severity TEXT,
                  parameter TEXT,
                  value REAL,
                  threshold REAL,
                  message TEXT,
                  acknowledged BOOLEAN)''')

    # Maintenance schedule table
    c.execute('''CREATE TABLE IF NOT EXISTS maintenance_schedule
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  task TEXT,
                  priority TEXT,
                  scheduled_date TEXT,
                  email_sent BOOLEAN,
                  completed BOOLEAN)''')

    # Machine stop log table
    c.execute('''CREATE TABLE IF NOT EXISTS machine_stop_log
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  reason TEXT,
                  stopped_by TEXT,
                  restored_at TEXT)''')

    conn.commit()
    conn.close()


# ============================================================================
# GENERATE SAMPLE DATASET
# ============================================================================
def generate_sample_dataset():
    """Generate comprehensive sample dataset with normal and failure conditions"""

    np.random.seed(42)
    n_samples = 5000

    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i * 10) for i in range(n_samples)]

    # Normal operating parameters
    temperature_normal = np.random.normal(65, 3, n_samples)
    vibration_normal = np.random.normal(4.5, 0.8, n_samples)
    pressure_normal = np.random.normal(70, 5, n_samples)
    rpm_normal = np.random.normal(3500, 200, n_samples)
    current_normal = np.random.normal(200, 15, n_samples)
    voltage_normal = np.random.normal(400, 8, n_samples)

    # Initialize arrays
    temperature = temperature_normal.copy()
    vibration = vibration_normal.copy()
    pressure = pressure_normal.copy()
    rpm = rpm_normal.copy()
    current = current_normal.copy()
    voltage = voltage_normal.copy()

    # Create failure conditions
    failure_indices = []
    fault_types = []

    # Type 1: Overheating failure (last 10% of samples)
    n_failures = int(n_samples * 0.15)
    overheating_idx = np.random.choice(range(int(n_samples * 0.7), n_samples),
                                       int(n_failures * 0.4), replace=False)
    for idx in overheating_idx:
        temperature[idx] += np.random.uniform(15, 35)
        failure_indices.append(idx)
        fault_types.append('overheating')

    # Type 2: Bearing wear failure
    bearing_idx = np.random.choice([i for i in range(int(n_samples * 0.8), n_samples)
                                    if i not in failure_indices],
                                   int(n_failures * 0.3), replace=False)
    for idx in bearing_idx:
        vibration[idx] += np.random.uniform(4, 9)
        temperature[idx] += np.random.uniform(5, 15)
        failure_indices.append(idx)
        fault_types.append('bearing_wear')

    # Type 3: Pressure failure
    pressure_idx = np.random.choice([i for i in range(int(n_samples * 0.85), n_samples)
                                     if i not in failure_indices],
                                    int(n_failures * 0.3), replace=False)
    for idx in pressure_idx:
        pressure[idx] += np.random.uniform(20, 40)
        failure_indices.append(idx)
        fault_types.append('pressure_spike')

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.clip(temperature, 20, 120),
        'vibration': np.clip(vibration, 0, 15),
        'pressure': np.clip(pressure, 40, 120),
        'rpm': np.clip(rpm, 2500, 6000),
        'current': np.clip(current, 150, 400),
        'voltage': np.clip(voltage, 350, 480),
        'fault_type': ['normal'] * n_samples
    })

    # Add fault types
    for idx, ftype in zip(failure_indices, fault_types):
        df.at[idx, 'fault_type'] = ftype

    # Add failure flag
    df['failure'] = (df['fault_type'] != 'normal').astype(int)

    # Add time to failure (in hours)
    df['time_to_failure'] = 500
    for idx in failure_indices:
        # Closer to failure = smaller TTF
        position_in_failure = idx / n_samples
        df.at[idx, 'time_to_failure'] = np.random.exponential(scale=50) * (1 - position_in_failure)

    # Add health score
    df['health_score'] = 100 - (df['temperature'] - 65).clip(lower=0) * 1.5
    df['health_score'] -= (df['vibration'] - 4.5).clip(lower=0) * 3
    df['health_score'] = df['health_score'].clip(0, 100)

    # Save to file
    file_path = r"C:\Users\emmanuel chiutsi\Documents\Industrial_fault_detection.csv"
    df.to_csv(file_path, index=False)

    return df, file_path


# ============================================================================
# LOAD AND PREPARE DATA FOR TRAINING
# ============================================================================
def load_and_prepare_data():
    """Load data from database or uploaded file for training"""
    conn = sqlite3.connect('industrial_monitoring.db')

    # First try to get data from database
    try:
        db_data = pd.read_sql_query("SELECT * FROM sensor_data ORDER BY timestamp DESC", conn)
        # Convert timestamp string back to datetime if needed
        if len(db_data) > 0 and 'timestamp' in db_data.columns:
            db_data['timestamp'] = pd.to_datetime(db_data['timestamp'])
    except Exception as e:
        print(f"Error loading database data: {e}")
        db_data = pd.DataFrame()

    conn.close()

    # If we have uploaded custom dataset, use that instead or merge
    if st.session_state.uploaded_data is not None and st.session_state.custom_dataset_loaded:
        uploaded_df = st.session_state.uploaded_data.copy()

        # Ensure required columns exist
        required_cols = ['temperature', 'vibration', 'pressure', 'rpm', 'current', 'voltage']
        if all(col in uploaded_df.columns for col in required_cols):
            # Add timestamp if not present
            if 'timestamp' not in uploaded_df.columns:
                uploaded_df['timestamp'] = datetime.now()

            # Convert timestamp to datetime if it's string
            if 'timestamp' in uploaded_df.columns:
                uploaded_df['timestamp'] = pd.to_datetime(uploaded_df['timestamp'])

            # Calculate health score if not present
            if 'health_score' not in uploaded_df.columns:
                sensor = SensorDataSensing()
                health_scores = []
                for _, row in uploaded_df.iterrows():
                    data_dict = row.to_dict()
                    health_scores.append(sensor.calculate_health_score(data_dict))
                uploaded_df['health_score'] = health_scores

            # Add time_to_failure if not present (simulate based on health score)
            if 'time_to_failure' not in uploaded_df.columns:
                uploaded_df['time_to_failure'] = uploaded_df['health_score'].apply(
                    lambda x: max(0.5, (x / 100) * 500) if x > 0 else 500
                )

            # Drop any rows with NaN in critical columns
            critical_cols = ['temperature', 'vibration', 'pressure', 'rpm', 'current', 'voltage', 'health_score']
            uploaded_df = uploaded_df.dropna(subset=critical_cols)

            # If we have database data, combine them
            if len(db_data) > 0:
                combined_data = pd.concat([db_data, uploaded_df], ignore_index=True)
                return combined_data
            else:
                return uploaded_df

    # Drop any rows with NaN in critical columns from database data
    if len(db_data) > 0:
        critical_cols = ['temperature', 'vibration', 'pressure', 'rpm', 'current', 'voltage']
        db_data = db_data.dropna(subset=critical_cols)

    return db_data


# ============================================================================
# ADD SENSOR DATA TO DATABASE
# ============================================================================
def add_sensor_data_to_db(data_row):
    """Add a single sensor reading to database"""
    conn = sqlite3.connect('industrial_monitoring.db')
    c = conn.cursor()

    # Convert timestamp to string to avoid type issues
    timestamp_val = data_row.get('timestamp', datetime.now())
    if isinstance(timestamp_val, datetime):
        timestamp_str = timestamp_val.strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp_str = str(timestamp_val)

    c.execute("""INSERT INTO sensor_data 
                 (timestamp, temperature, vibration, pressure, rpm, current, voltage, health_score)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp_str,
               float(data_row.get('temperature', 0)),
               float(data_row.get('vibration', 0)),
               float(data_row.get('pressure', 0)),
               float(data_row.get('rpm', 0)),
               float(data_row.get('current', 0)),
               float(data_row.get('voltage', 0)),
               float(data_row.get('health_score', 50))))
    conn.commit()
    conn.close()


# ============================================================================
# SENSOR DATA SENSING MODULE
# ============================================================================
class SensorDataSensing:
    """Sense machine operational data based on parameters of interest"""

    def __init__(self):
        self.parameters_of_interest = [
            'temperature', 'vibration', 'pressure', 'rpm', 'current', 'voltage'
        ]

    def read_sensors(self, manual_input=None):
        """Read sensor data (simulated or manual)"""
        if manual_input:
            return manual_input

        # If machine is stopped, return zero readings
        if st.session_state.machine_stopped:
            return {
                'timestamp': datetime.now(),
                'temperature': 0,
                'vibration': 0,
                'pressure': 0,
                'rpm': 0,
                'current': 0,
                'voltage': 0
            }

        # Simulate realistic sensor readings with slight variations
        base_readings = {
            'timestamp': datetime.now(),
            'temperature': np.random.normal(65, 5),
            'vibration': np.random.normal(4.5, 1),
            'pressure': np.random.normal(70, 8),
            'rpm': np.random.normal(3500, 300),
            'current': np.random.normal(200, 20),
            'voltage': np.random.normal(400, 10)
        }

        # Add realistic constraints
        base_readings['temperature'] = np.clip(base_readings['temperature'], 20, 120)
        base_readings['vibration'] = np.clip(base_readings['vibration'], 0, 15)
        base_readings['pressure'] = np.clip(base_readings['pressure'], 0, 120)
        base_readings['rpm'] = np.clip(base_readings['rpm'], 0, 6000)

        return base_readings

    def calculate_health_score(self, data):
        """Calculate equipment health score (0-100)"""
        # If machine is stopped, health score is 0 (not operational)
        if st.session_state.machine_stopped:
            return 0

        health = 100

        # Temperature impact
        if data['temperature'] > 85:
            health -= (data['temperature'] - 85) * 2
        elif data['temperature'] > 75:
            health -= (data['temperature'] - 75) * 1

        # Vibration impact
        if data['vibration'] > 8.5:
            health -= (data['vibration'] - 8.5) * 5
        elif data['vibration'] > 6.5:
            health -= (data['vibration'] - 6.5) * 3

        # Pressure impact
        if data['pressure'] > 95:
            health -= (data['pressure'] - 95) * 2
        elif data['pressure'] > 85:
            health -= (data['pressure'] - 85) * 1

        return max(0, min(100, health))


# ============================================================================
# PATTERN ANALYZER
# ============================================================================
class PatternAnalyzer:
    """Analyse machine data to identify underlying patterns"""

    def detect_anomalies(self, data):
        """Detect anomalies in current readings"""
        # If machine is stopped, no anomalies
        if st.session_state.machine_stopped:
            return ["Machine is stopped - No active operation"]

        anomalies = []

        if data['temperature'] > 85:
            anomalies.append(f"CRITICAL: Temperature at {data['temperature']:.1f}°C")
        elif data['temperature'] > 75:
            anomalies.append(f"WARNING: Elevated temperature at {data['temperature']:.1f}°C")

        if data['vibration'] > 8.5:
            anomalies.append(f"CRITICAL: Vibration at {data['vibration']:.2f} mm/s")
        elif data['vibration'] > 6.5:
            anomalies.append(f"WARNING: Elevated vibration at {data['vibration']:.2f} mm/s")

        if data['pressure'] > 95:
            anomalies.append(f"CRITICAL: Pressure at {data['pressure']:.1f} PSI")
        elif data['pressure'] > 85:
            anomalies.append(f"WARNING: Elevated pressure at {data['pressure']:.1f} PSI")

        if data['rpm'] > 5500:
            anomalies.append(f"CRITICAL: RPM at {data['rpm']:.0f}")
        elif data['rpm'] > 5000:
            anomalies.append(f"WARNING: High RPM at {data['rpm']:.0f}")

        return anomalies

    def identify_degradation_pattern(self, historical_data):
        """Identify degradation patterns from historical data"""
        if len(historical_data) < 10:
            return "Insufficient data for pattern analysis. Need at least 10 data points."

        # Calculate recent trends
        recent_temp = historical_data['temperature'].tail(min(50, len(historical_data))).values
        recent_vib = historical_data['vibration'].tail(min(50, len(historical_data))).values

        if len(recent_temp) > 5:
            temp_trend = np.polyfit(range(len(recent_temp)), recent_temp, 1)[0]
            vib_trend = np.polyfit(range(len(recent_vib)), recent_vib, 1)[0]

            if temp_trend > 0.1 and vib_trend > 0.05:
                return "Accelerating degradation - Bearing wear pattern detected"
            elif temp_trend > 0.15:
                return "Thermal degradation pattern - Cooling system issue"
            elif vib_trend > 0.1:
                return "Mechanical degradation - Imbalance or misalignment"
            else:
                return "Normal operation - No significant degradation"
        else:
            return "Collecting more data for trend analysis..."


# ============================================================================
# FAILURE PREDICTOR
# ============================================================================
class FailurePredictor:
    """Prediction of Expected Time of Failure"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def train_model(self, historical_data):
        """Train prediction model"""
        if len(historical_data) < 10:
            return None

        feature_cols = ['temperature', 'vibration', 'pressure', 'rpm', 'current', 'voltage']

        # Prepare features
        X = historical_data[feature_cols].values

        # Create synthetic time to failure based on health score if not present
        if 'time_to_failure' in historical_data.columns:
            y = historical_data['time_to_failure'].values
        else:
            # Generate synthetic TTF based on health score
            if 'health_score' in historical_data.columns:
                y = historical_data['health_score'].apply(lambda x: max(0.5, (x / 100) * 500) if x > 0 else 500).values
            else:
                y = np.ones(len(historical_data)) * 500

        # Handle NaN values - replace with mean or default
        y = np.nan_to_num(y, nan=500.0)

        # Check for any remaining NaN or inf
        if np.isnan(y).any() or np.isinf(y).any():
            y = np.nan_to_num(y, nan=500.0, posinf=500.0, neginf=500.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)

        return self.model

    def predict_ttf(self, current_data):
        """Predict time to failure"""
        # If machine is stopped, no failure prediction
        if st.session_state.machine_stopped:
            return {
                'predicted_ttf': 999999,
                'confidence': 1.0,
                'severity_factor': 0
            }

        if self.model is None:
            return None

        feature_cols = ['temperature', 'vibration', 'pressure', 'rpm', 'current', 'voltage']
        current_features = np.array([[float(current_data[col]) for col in feature_cols]])
        current_scaled = self.scaler.transform(current_features)

        ttf = self.model.predict(current_scaled)[0]

        # Adjust based on current severity
        severity_factor = 1.0
        if current_data['temperature'] > 85:
            severity_factor *= 0.5
        if current_data['vibration'] > 8.5:
            severity_factor *= 0.6

        adjusted_ttf = ttf * severity_factor

        return {
            'predicted_ttf': max(0.5, adjusted_ttf),
            'confidence': 0.85 if adjusted_ttf > 24 else 0.7,
            'severity_factor': severity_factor
        }

    def map_historical_profiles(self, current_data):
        """Map current degradation against historical failure profiles"""
        if st.session_state.machine_stopped:
            return []

        profiles = []

        if current_data['temperature'] > 80 and current_data['vibration'] > 7:
            profiles.append({
                'type': 'Critical Bearing Failure',
                'probability': 0.92,
                'pattern': 'Simultaneous temperature and vibration increase',
                'typical_ttf': 12
            })
        elif current_data['temperature'] > 80:
            profiles.append({
                'type': 'Thermal Failure',
                'probability': 0.85,
                'pattern': 'Rapid temperature increase',
                'typical_ttf': 24
            })
        elif current_data['vibration'] > 7.5:
            profiles.append({
                'type': 'Mechanical Failure',
                'probability': 0.75,
                'pattern': 'Increasing vibration amplitude',
                'typical_ttf': 72
            })

        return profiles

    def calculate_failure_probability(self, current_data, historical_data):
        """Calculate accurate failure probability based on current data and historical patterns"""
        if st.session_state.machine_stopped:
            return 0

        if len(historical_data) < 10:
            return 0.15

        # Get current parameter values
        temp = current_data['temperature']
        vib = current_data['vibration']
        press = current_data['pressure']
        rpm_val = current_data['rpm']

        # Calculate individual risk contributions
        temp_risk = 0
        if temp > 85:
            temp_risk = min(0.95, (temp - 85) / 20)
        elif temp > 75:
            temp_risk = min(0.6, (temp - 75) / 15)

        vib_risk = 0
        if vib > 8.5:
            vib_risk = min(0.95, (vib - 8.5) / 5)
        elif vib > 6.5:
            vib_risk = min(0.5, (vib - 6.5) / 4)

        press_risk = 0
        if press > 95:
            press_risk = min(0.85, (press - 95) / 20)
        elif press > 85:
            press_risk = min(0.4, (press - 85) / 15)

        rpm_risk = 0
        if rpm_val > 5500:
            rpm_risk = min(0.75, (rpm_val - 5500) / 500)
        elif rpm_val > 5000:
            rpm_risk = min(0.3, (rpm_val - 5000) / 500)

        # Combine risks with weights
        total_risk = (temp_risk * 0.4) + (vib_risk * 0.35) + (press_risk * 0.15) + (rpm_risk * 0.1)

        # Get historical failure rate for similar conditions
        if 'health_score' in historical_data.columns:
            recent_failures = historical_data[historical_data['health_score'] < 40].shape[0]
            historical_risk = min(0.5, recent_failures / len(historical_data))
        else:
            historical_risk = 0.1

        # Final probability
        final_probability = min(0.98, total_risk * 0.7 + historical_risk * 0.3)

        return final_probability

    def predict_failure_details(self, current_data, historical_data):
        """Generate detailed failure prediction including type and severity"""
        if st.session_state.machine_stopped:
            return {
                'predicted_failure_type': 'Machine Stopped',
                'probability': 0,
                'timeframe': 'Not applicable - Machine is stopped',
                'severity': 'SAFE',
                'affected_components': ['None - Machine not operational'],
                'current_health_score': 0,
                'recommended_action': 'Machine is safely stopped. No immediate action needed.',
                'predicted_ttf_hours': 999999,
                'predicted_failure_datetime': None,
                'prediction_timestamp': datetime.now()
            }

        if len(historical_data) < 10:
            return {
                'predicted_failure_type': 'Insufficient Data',
                'probability': 0.15,
                'timeframe': 'Unknown - Need more data',
                'severity': 'Unknown',
                'affected_components': ['Unable to determine - collect more sensor data'],
                'predicted_failure_datetime': None,
                'recommended_action': 'Continue collecting sensor data for accurate predictions'
            }

        # Get TTF prediction first
        ttf_prediction = self.predict_ttf(current_data)
        predicted_ttf_hours = ttf_prediction['predicted_ttf'] if ttf_prediction else 72

        # Calculate predicted failure date and time
        current_time = datetime.now()
        predicted_failure_datetime = current_time + timedelta(hours=predicted_ttf_hours)

        # Determine failure type based on patterns
        temp = current_data['temperature']
        vib = current_data['vibration']
        press = current_data['pressure']

        failure_type = "General Degradation"
        affected_parts = ["General wear"]

        if temp > 85 and vib > 7:
            failure_type = "Critical Bearing Failure with Thermal Overload"
            affected_parts = ["Bearings", "Shaft Assembly", "Lubrication System"]
        elif temp > 80:
            failure_type = "Thermal Overload Failure"
            affected_parts = ["Cooling System", "Motor Windings", "Insulation"]
        elif vib > 8:
            failure_type = "Severe Mechanical Imbalance/Bearing Failure"
            affected_parts = ["Bearings", "Rotor Assembly", "Mounting Structure"]
        elif vib > 6.5:
            failure_type = "Developing Bearing Wear/Misalignment"
            affected_parts = ["Bearings", "Coupling", "Alignment"]
        elif press > 90:
            failure_type = "Pressure System Failure"
            affected_parts = ["Seals", "Valves", "Pressure Vessel"]

        # Calculate probability
        probability = self.calculate_failure_probability(current_data, historical_data)

        # Determine timeframe with exact date
        if predicted_ttf_hours < 24:
            timeframe = f"0-24 hours (by {predicted_failure_datetime.strftime('%Y-%m-%d %H:%M')}) - IMMINENT"
            severity = "CRITICAL"
        elif predicted_ttf_hours < 72:
            timeframe = f"24-72 hours (by {predicted_failure_datetime.strftime('%Y-%m-%d %H:%M')})"
            severity = "HIGH"
        elif predicted_ttf_hours < 168:
            timeframe = f"3-7 days (by {predicted_failure_datetime.strftime('%Y-%m-%d %H:%M')})"
            severity = "MEDIUM"
        else:
            timeframe = f"> 7 days (approx {predicted_failure_datetime.strftime('%Y-%m-%d %H:%M')})"
            severity = "LOW"

        # Get current health score
        sensor = SensorDataSensing()
        health_score = sensor.calculate_health_score(current_data)

        return {
            'predicted_failure_type': failure_type,
            'probability': probability,
            'timeframe': timeframe,
            'severity': severity,
            'affected_components': affected_parts,
            'current_health_score': health_score,
            'recommended_action': self._get_recommended_action(probability, failure_type),
            'predicted_ttf_hours': predicted_ttf_hours,
            'predicted_failure_datetime': predicted_failure_datetime,
            'prediction_timestamp': current_time
        }

    def _get_recommended_action(self, probability, failure_type):
        if probability > 0.7:
            return "IMMEDIATE SHUTDOWN - Emergency maintenance required"
        elif probability > 0.4:
            return "Schedule maintenance within 24-48 hours"
        elif probability > 0.2:
            return "Plan maintenance for next scheduled downtime"
        else:
            return "Continue monitoring - No immediate action needed"


# ============================================================================
# STATISTICAL THRESHOLDS
# ============================================================================
class StatisticalThresholds:
    """Create Statistical Likelihood Thresholds"""

    def __init__(self):
        self.thresholds = {
            'temperature': {'critical': 85, 'warning': 75, 'weight': 0.35},
            'vibration': {'critical': 8.5, 'warning': 6.5, 'weight': 0.30},
            'pressure': {'critical': 95, 'warning': 85, 'weight': 0.20},
            'rpm': {'critical': 5500, 'warning': 5000, 'weight': 0.15}
        }

    def calculate_likelihood(self, data):
        """Calculate failure likelihood based on thresholds"""
        # If machine is stopped, likelihood is 0
        if st.session_state.machine_stopped:
            return {
                'likelihood': 0,
                'risk_level': 'NORMAL',
                'score': 0
            }

        likelihood_scores = []
        total_weight = 0

        for param, value in data.items():
            if param in self.thresholds:
                thresh = self.thresholds[param]

                if value >= thresh['critical']:
                    likelihood = 0.9 + min(0.1, (value - thresh['critical']) / thresh['critical'])
                elif value >= thresh['warning']:
                    ratio = (value - thresh['warning']) / (thresh['critical'] - thresh['warning'])
                    likelihood = 0.4 + ratio * 0.5
                else:
                    likelihood = 0.1

                likelihood_scores.append(likelihood * thresh['weight'])
                total_weight += thresh['weight']

        overall = sum(likelihood_scores) / total_weight if total_weight > 0 else 0

        # Determine risk level
        if overall >= 0.7:
            risk = "CRITICAL"
        elif overall >= 0.4:
            risk = "WARNING"
        else:
            risk = "NORMAL"

        return {
            'likelihood': overall,
            'risk_level': risk,
            'score': overall * 100
        }


# ============================================================================
# ALERT GENERATOR
# ============================================================================
class AlertGenerator:
    """Generate automated early warning alerts"""

    def __init__(self):
        self.email_config = st.session_state.email_config

    def update_email_config(self, config):
        """Update email configuration"""
        self.email_config = config
        st.session_state.email_config = config

    def generate_alerts(self, data, likelihood, predictions, health_score):
        """Generate alerts based on current state"""
        alerts = []

        # If machine is stopped, show stopped alert
        if st.session_state.machine_stopped:
            alerts.append({
                'type': 'info',
                'severity': 'INFO',
                'message': '🛑 MACHINE STOPPED - System is in emergency stop mode. No active monitoring.',
                'action': 'Click "Restart Machine" to resume operations'
            })
            st.session_state.critical_mode = False
            return alerts

        # Critical alerts
        if likelihood['risk_level'] == 'CRITICAL':
            alerts.append({
                'type': 'critical',
                'severity': 'CRITICAL',
                'message': f"🔴 CRITICAL ALERT! System failure probability at {likelihood['likelihood']:.1%}. Immediate action required!",
                'action': 'IMMEDIATE SHUTDOWN AND INSPECTION'
            })
            st.session_state.critical_mode = True
        elif likelihood['risk_level'] == 'WARNING':
            alerts.append({
                'type': 'warning',
                'severity': 'WARNING',
                'message': f"⚠️ WARNING: Elevated failure risk ({likelihood['likelihood']:.1%}). Schedule maintenance within 24 hours.",
                'action': 'SCHEDULE INSPECTION'
            })
            st.session_state.critical_mode = True
        else:
            # System is NORMAL - stop flashing red light
            st.session_state.critical_mode = False
            st.session_state.email_sent_for_risk = False

        # TTF based alerts
        if predictions and predictions['predicted_ttf'] < 24:
            alerts.append({
                'type': 'critical',
                'severity': 'CRITICAL',
                'message': f"⏰ FAILURE IMMINENT! Predicted failure in {predictions['predicted_ttf']:.1f} hours!",
                'action': 'EMERGENCY MAINTENANCE REQUIRED'
            })
        elif predictions and predictions['predicted_ttf'] < 72:
            alerts.append({
                'type': 'warning',
                'severity': 'WARNING',
                'message': f"⚠️ Failure predicted in {predictions['predicted_ttf']:.1f} hours. Prepare for maintenance.",
                'action': 'SCHEDULE MAINTENANCE WITHIN 3 DAYS'
            })

        # Parameter-specific alerts
        if data['temperature'] > 85:
            alerts.append({
                'type': 'critical',
                'severity': 'CRITICAL',
                'message': f"🌡️ CRITICAL: Temperature at {data['temperature']:.1f}°C exceeds safety limit!",
                'action': 'CHECK COOLING SYSTEM IMMEDIATELY'
            })

        if data['vibration'] > 8.5:
            alerts.append({
                'type': 'critical',
                'severity': 'CRITICAL',
                'message': f"📳 CRITICAL: Vibration at {data['vibration']:.2f} mm/s indicates bearing failure risk!",
                'action': 'INSPECT BEARINGS AND ALIGNMENT'
            })

        return alerts

    def send_email(self, alert, current_data):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = self.email_config['recipient']
            msg['Subject'] = f"URGENT: {alert['severity']} Alert - Industrial System"

            body = f"""
            ALERT DETAILS:
            Time: {datetime.now()}
            Severity: {alert['severity']}
            Message: {alert['message']}

            CURRENT READINGS:
            Temperature: {current_data['temperature']:.1f}°C
            Vibration: {current_data['vibration']:.2f} mm/s
            Pressure: {current_data['pressure']:.1f} PSI
            RPM: {current_data['rpm']:.0f}
            Current: {current_data['current']:.1f} A
            Voltage: {current_data['voltage']:.1f} V

            RECOMMENDED ACTION:
            {alert['action']}

            This is an automated alert from Industrial Predictive Maintenance System.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender'], self.email_config['password'])
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            print(f"Email error: {e}")
            return False

    def send_risk_email(self, risk_level, likelihood_score, current_data, failure_details=None):
        """Send email when system is declared urgent/high priority or not normal risk"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = self.email_config['recipient']
            msg['Subject'] = f"URGENT: {risk_level} RISK ALERT - Industrial System Requires Attention"

            body = f"""
            RISK ALERT DETAILS:
            Time: {datetime.now()}
            Risk Level: {risk_level}
            Failure Probability: {likelihood_score:.1%}

            CURRENT READINGS:
            Temperature: {current_data['temperature']:.1f}°C
            Vibration: {current_data['vibration']:.2f} mm/s
            Pressure: {current_data['pressure']:.1f} PSI
            RPM: {current_data['rpm']:.0f}
            Current: {current_data['current']:.1f} A
            Voltage: {current_data['voltage']:.1f} V
            Health Score: {current_data.get('health_score', 'N/A')}
            """

            if failure_details:
                body += f"""

            FAILURE PREDICTION DETAILS:
            Predicted Failure Type: {failure_details.get('predicted_failure_type', 'Unknown')}
            Failure Probability: {failure_details.get('probability', 0):.1%}
            Expected Timeframe: {failure_details.get('timeframe', 'Unknown')}
            Predicted Failure Date & Time: {failure_details.get('predicted_failure_datetime', 'Unknown')}
            Severity: {failure_details.get('severity', 'Unknown')}
            Affected Components: {', '.join(failure_details.get('affected_components', ['Unknown']))}
            Recommended Action: {failure_details.get('recommended_action', 'Inspect system immediately')}
            """

            body += """

            This is an automated alert from Industrial Predictive Maintenance System.
            Immediate attention required to prevent equipment failure.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender'], self.email_config['password'])
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            print(f"Risk email error: {e}")
            return False

    def send_maintenance_email(self, task, priority, scheduled_date, current_data):
        """Send email when maintenance is scheduled"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = self.email_config['recipient']
            msg['Subject'] = f"Maintenance Scheduled: {priority} Priority - {task}"

            body = f"""
            MAINTENANCE SCHEDULED:
            Scheduled By: Industrial Predictive Maintenance System
            Time Scheduled: {datetime.now()}
            Task: {task}
            Priority: {priority}
            Scheduled Date: {scheduled_date}

            CURRENT SYSTEM STATUS:
            Temperature: {current_data['temperature']:.1f}°C
            Vibration: {current_data['vibration']:.2f} mm/s
            Pressure: {current_data['pressure']:.1f} PSI
            RPM: {current_data['rpm']:.0f}
            Health Score: {current_data.get('health_score', 'N/A')}

            Please ensure maintenance is performed as scheduled.

            This is an automated notification from Industrial Predictive Maintenance System.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender'], self.email_config['password'])
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            print(f"Maintenance email error: {e}")
            return False


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================
class RecommendationEngine:
    """Provide prescriptive recommendations as interventions"""

    def generate_recommendations(self, alerts, current_data, predictions):
        """Generate prescriptive recommendations"""
        recommendations = []

        for alert in alerts:
            if 'temperature' in alert['message'] and alert['type'] == 'critical':
                recommendations.append({
                    'priority': 'URGENT',
                    'issue': 'Critical Overheating Detected',
                    'actions': [
                        '🛑 IMMEDIATELY stop machine operation',
                        '❄️ Check cooling system for failure',
                        '🔧 Inspect for blocked ventilation',
                        '💨 Verify all cooling fans operational',
                        '📊 Monitor temperature every 5 minutes'
                    ],
                    'downtime': '2-4 hours'
                })

            elif 'vibration' in alert['message'] and alert['type'] == 'critical':
                recommendations.append({
                    'priority': 'URGENT',
                    'issue': 'Critical Vibration - Bearing Failure Risk',
                    'actions': [
                        '🛑 Stop machine immediately',
                        '🔍 Inspect bearings for damage',
                        '⚙️ Check shaft alignment',
                        '🔩 Verify mounting bolts tightness',
                        '📈 Perform vibration analysis'
                    ],
                    'downtime': '4-8 hours'
                })

            elif alert['type'] == 'warning':
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': 'Elevated Risk Detected',
                    'actions': [
                        '📊 Increase monitoring frequency',
                        '🔧 Schedule inspection within 24 hours',
                        '📝 Review recent maintenance logs',
                        '🔄 Prepare replacement parts',
                        '👥 Alert maintenance team'
                    ],
                    'downtime': '1-2 hours (scheduled)'
                })

        # Predictive recommendations
        if predictions and predictions['predicted_ttf'] < 48:
            recommendations.append({
                'priority': 'URGENT',
                'issue': f"Failure Predicted in {predictions['predicted_ttf']:.1f} Hours",
                'actions': [
                    '📦 Order replacement parts immediately',
                    '👥 Schedule maintenance crew',
                    '📋 Prepare maintenance checklist',
                    '⏰ Plan production downtime',
                    '🚨 Activate emergency response plan'
                ],
                'downtime': 'As scheduled'
            })

        return recommendations


# ============================================================================
# EMAIL CONFIGURATION CONTROL PANEL
# ============================================================================
def email_configuration_panel():
    """Email configuration control panel in sidebar"""
    with st.sidebar.expander("📧 Email Configuration", expanded=False):
        st.markdown("### Configure Email Settings")

        # SMTP Settings
        smtp_server = st.text_input("SMTP Server", value=st.session_state.email_config['smtp_server'])
        smtp_port = st.number_input("SMTP Port", value=st.session_state.email_config['smtp_port'])

        # Email Credentials
        sender_email = st.text_input("Sender Email", value=st.session_state.email_config['sender'])
        recipient_email = st.text_input("Recipient Email", value=st.session_state.email_config['recipient'])
        password = st.text_input("App Password", value=st.session_state.email_config['password'], type="password")

        # Test Email Button
        if st.button("📧 Test Email Configuration", key="test_email_btn"):
            try:
                test_msg = MIMEMultipart()
                test_msg['From'] = sender_email
                test_msg['To'] = recipient_email
                test_msg['Subject'] = "Test Email - Industrial Predictive Maintenance System"
                test_body = """
                This is a test email from your Industrial Predictive Maintenance System.

                Email configuration is working correctly.

                Time: {}
                """.format(datetime.now())
                test_msg.attach(MIMEText(test_body, 'plain'))

                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender_email, password)
                server.send_message(test_msg)
                server.quit()

                st.success("✅ Test email sent successfully!")

                # Update configuration if test successful
                st.session_state.email_config = {
                    'smtp_server': smtp_server,
                    'smtp_port': smtp_port,
                    'sender': sender_email,
                    'recipient': recipient_email,
                    'password': password
                }

            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")

        st.markdown("---")
        st.caption("Note: For Gmail, use an App Password (not your regular password)")


# ============================================================================
# DATA UPLOAD PANEL
# ============================================================================
def data_upload_panel():
    """Data upload panel in sidebar for adding custom datasets"""
    with st.sidebar.expander("📂 Data Management", expanded=False):
        st.markdown("### Add Sensor Data for Training")

        # Option to generate sample data
        if st.button("🔄 Generate Sample Dataset (5000 records)", use_container_width=True):
            with st.spinner("Generating sample dataset..."):
                df, filepath = generate_sample_dataset()
                st.success(f"✅ Sample dataset generated with {len(df)} records!")

                # Load the sample data into session for training
                st.session_state.uploaded_data = df
                st.session_state.custom_dataset_loaded = True

                # Also add to database - convert timestamp to string
                for idx, row in df.head(200).iterrows():  # Add first 200 to avoid overwhelming
                    row_dict = row.to_dict()
                    # Convert timestamp to string
                    if isinstance(row_dict.get('timestamp'), datetime):
                        row_dict['timestamp'] = row_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    add_sensor_data_to_db(row_dict)
                st.info("Sample data loaded for training predictions!")
                st.rerun()

        st.markdown("---")
        st.markdown("### Upload Your Own Dataset")
        st.caption(
            "Upload CSV with columns: temperature, vibration, pressure, rpm, current, voltage (optional: timestamp, health_score)")

        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded {len(df)} records")

                # Show preview
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(5), use_container_width=True)

                # Check required columns
                required_cols = ['temperature', 'vibration', 'pressure', 'rpm', 'current', 'voltage']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.info(
                        "Please ensure your CSV has these columns: temperature, vibration, pressure, rpm, current, voltage")
                else:
                    if st.button("📥 Load Dataset for Training", use_container_width=True):
                        # Clean the data - handle NaN values
                        for col in required_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                        st.session_state.uploaded_data = df
                        st.session_state.custom_dataset_loaded = True

                        # Add data to database
                        with st.spinner("Adding data to database..."):
                            count = 0
                            for _, row in df.iterrows():
                                row_dict = row.to_dict()
                                # Ensure timestamp is string
                                if 'timestamp' in row_dict:
                                    if isinstance(row_dict['timestamp'], datetime):
                                        row_dict['timestamp'] = row_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                                    elif pd.isna(row_dict['timestamp']):
                                        row_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    else:
                                        row_dict['timestamp'] = str(row_dict['timestamp'])
                                else:
                                    row_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                                add_sensor_data_to_db(row_dict)
                                count += 1
                            st.success(f"✅ Added {count} records to database for training!")

                        st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

        # Show current data status
        st.markdown("---")
        st.markdown("### Current Data Status")

        conn = sqlite3.connect('industrial_monitoring.db')
        try:
            db_count = pd.read_sql_query("SELECT COUNT(*) as count FROM sensor_data", conn).iloc[0]['count']
        except:
            db_count = 0
        conn.close()

        st.metric("Records in Database", db_count)

        if st.session_state.custom_dataset_loaded:
            st.success("✅ Custom dataset loaded and ready for predictions!")
        else:
            if db_count > 0:
                st.info(f"ℹ️ {db_count} records available. Upload more data for better predictions.")
            else:
                st.warning("⚠️ No data available. Generate sample data or upload your own to enable predictions.")

        if st.button("🔄 Refresh Data Status", use_container_width=True):
            st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application"""

    # Initialize database
    init_database()

    # Initialize modules
    sensor = SensorDataSensing()
    pattern_analyzer = PatternAnalyzer()
    predictor = FailurePredictor()
    thresholds = StatisticalThresholds()
    alert_gen = AlertGenerator()
    recommender = RecommendationEngine()

    # Update alert_gen with current email config
    alert_gen.update_email_config(st.session_state.email_config)

    # Generate initial sample data if database is empty
    conn_check = sqlite3.connect('industrial_monitoring.db')
    try:
        db_count = pd.read_sql_query("SELECT COUNT(*) as count FROM sensor_data", conn_check).iloc[0]['count']
    except:
        db_count = 0
    conn_check.close()

    if db_count == 0 and not st.session_state.custom_dataset_loaded:
        with st.spinner("Generating initial sample dataset for predictions..."):
            df, filepath = generate_sample_dataset()
            # Add first 200 records to database with proper timestamp formatting
            for idx, row in df.head(200).iterrows():
                row_dict = row.to_dict()
                # Convert timestamp to string
                if isinstance(row_dict.get('timestamp'), datetime):
                    row_dict['timestamp'] = row_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                add_sensor_data_to_db(row_dict)
            st.success("✅ Initial sample data loaded for predictions!")

    # Connect to database
    conn = sqlite3.connect('industrial_monitoring.db')

    # Sidebar navigation
    st.sidebar.markdown("## 🏭 Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["🏠 Live Dashboard", "🔍 Pattern Analysis", "⚠️ Failure Prediction",
         "📈 Risk Analysis", "🚨 Alerts & Actions", "📊 Reports"]
    )

    # Email Configuration Panel
    email_configuration_panel()

    # Data Upload Panel
    data_upload_panel()

    # Stop Machine Button in Sidebar - EMERGENCY STOP
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 EMERGENCY CONTROLS")

    if not st.session_state.machine_stopped:
        if st.sidebar.button("🛑 EMERGENCY STOP MACHINE", key="stop_machine_btn", use_container_width=True):
            st.session_state.machine_stopped = True
            st.session_state.stop_machine_time = datetime.now()
            st.session_state.critical_mode = True

            # Log the stop
            c = conn.cursor()
            c.execute("""INSERT INTO machine_stop_log (timestamp, reason, stopped_by)
                         VALUES (?, ?, ?)""",
                      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Emergency stop triggered by operator",
                       "System Operator"))
            conn.commit()

            # Send email about emergency stop
            emergency_alert = {
                'type': 'critical',
                'severity': 'CRITICAL',
                'message': 'EMERGENCY STOP ACTIVATED - Machine has been stopped by operator',
                'action': 'Inspect machine before restarting'
            }
            current_data = sensor.read_sensors()
            alert_gen.send_email(emergency_alert, current_data)

            st.sidebar.success("✅ Machine STOPPED! Emergency mode activated.")
            st.rerun()
    else:
        st.sidebar.error("🔴 MACHINE STOPPED")
        if st.sidebar.button("🟢 RESTART MACHINE", key="restart_machine_btn", use_container_width=True):
            st.session_state.machine_stopped = False
            st.session_state.critical_mode = False
            st.session_state.stop_machine_time = None

            # Log the restart
            c = conn.cursor()
            c.execute("UPDATE machine_stop_log SET restored_at = ? WHERE restored_at IS NULL",
                      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))
            conn.commit()

            st.sidebar.success("✅ Machine restarted! Normal operations resumed.")
            st.rerun()

    # Critical mode overlay (only show when critical_mode is True)
    if st.session_state.critical_mode and not st.session_state.machine_stopped:
        st.markdown('<div class="critical-overlay"></div>', unsafe_allow_html=True)

    # Machine stopped overlay
    if st.session_state.machine_stopped:
        st.markdown("""
        <div class="machine-stopped-overlay">
            <div class="machine-stopped-content">
                <h1 style="color: white; font-size: 3rem;">🛑 MACHINE STOPPED 🛑</h1>
                <p style="color: white; font-size: 1.5rem;">Emergency Stop Activated</p>
                <p style="color: white;">Please inspect the machine before restarting</p>
                <p style="color: #ff6666;">Use the "RESTART MACHINE" button in the sidebar to resume operations</p>
                <p style="color: white; margin-top: 1rem;">Stop time: """ + (
            st.session_state.stop_machine_time.strftime(
                '%Y-%m-%d %H:%M:%S') if st.session_state.stop_machine_time else "") + """</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🏭 Industrial Predictive Maintenance System</h1>
        <p>AI-Powered Machine Monitoring & Failure Prevention Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Display machine status
    if st.session_state.machine_stopped:
        st.error("🔴 **EMERGENCY MODE: MACHINE IS STOPPED** 🔴\n\nPlease restart from sidebar to resume monitoring.")

    # ========================================================================
    # LIVE DASHBOARD PAGE
    # ========================================================================
    if page == "🏠 Live Dashboard":
        st.header("📊 Real-Time Equipment Monitoring")

        # Input method
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Current Sensor Readings")

            use_manual = st.checkbox("Enter Manual Readings", value=False)

            if use_manual:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    temp = st.number_input("Temperature (°C)", 0.0, 120.0, 72.5)
                    vib = st.number_input("Vibration (mm/s)", 0.0, 15.0, 5.8)
                with col_b:
                    press = st.number_input("Pressure (PSI)", 0.0, 120.0, 78.3)
                    rpm = st.number_input("RPM", 0, 6000, 4200)
                with col_c:
                    current = st.number_input("Current (A)", 0.0, 500.0, 245.0)
                    voltage = st.number_input("Voltage (V)", 0.0, 500.0, 415.0)

                current_data = {
                    'timestamp': datetime.now(),
                    'temperature': temp,
                    'vibration': vib,
                    'pressure': press,
                    'rpm': rpm,
                    'current': current,
                    'voltage': voltage
                }
            else:
                # Auto-simulate with occasional critical conditions for testing (only if machine not stopped)
                if not st.session_state.machine_stopped:
                    if random.random() < 0.1:  # 10% chance of critical condition for testing
                        current_data = {
                            'timestamp': datetime.now(),
                            'temperature': random.uniform(88, 105),
                            'vibration': random.uniform(8.8, 12),
                            'pressure': random.uniform(98, 115),
                            'rpm': random.uniform(5600, 6200),
                            'current': random.uniform(350, 420),
                            'voltage': random.uniform(380, 410)
                        }
                    else:
                        current_data = sensor.read_sensors()
                else:
                    current_data = sensor.read_sensors()

                # Display readings with metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if not st.session_state.machine_stopped:
                        temp_delta = current_data['temperature'] - 65
                    else:
                        temp_delta = 0
                    st.metric("🌡️ Temperature", f"{current_data['temperature']:.1f}°C",
                              f"{temp_delta:+.1f}°C" if not st.session_state.machine_stopped else "STOPPED",
                              delta_color="inverse")
                with col_b:
                    if not st.session_state.machine_stopped:
                        vib_delta = current_data['vibration'] - 4.5
                    else:
                        vib_delta = 0
                    st.metric("📳 Vibration", f"{current_data['vibration']:.2f} mm/s",
                              f"{vib_delta:+.2f}" if not st.session_state.machine_stopped else "STOPPED",
                              delta_color="inverse")
                with col_c:
                    if not st.session_state.machine_stopped:
                        press_delta = current_data['pressure'] - 70
                    else:
                        press_delta = 0
                    st.metric("💨 Pressure", f"{current_data['pressure']:.1f} PSI",
                              f"{press_delta:+.1f}" if not st.session_state.machine_stopped else "STOPPED",
                              delta_color="inverse")

                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    if not st.session_state.machine_stopped:
                        rpm_delta = current_data['rpm'] - 3500
                    else:
                        rpm_delta = 0
                    st.metric("⚙️ RPM", f"{current_data['rpm']:.0f}",
                              f"{rpm_delta:+.0f}" if not st.session_state.machine_stopped else "STOPPED",
                              delta_color="inverse")
                with col_e:
                    if not st.session_state.machine_stopped:
                        current_delta = current_data['current'] - 200
                    else:
                        current_delta = 0
                    st.metric("⚡ Current", f"{current_data['current']:.1f} A",
                              f"{current_delta:+.1f}" if not st.session_state.machine_stopped else "STOPPED",
                              delta_color="inverse")
                with col_f:
                    if not st.session_state.machine_stopped:
                        voltage_delta = current_data['voltage'] - 400
                    else:
                        voltage_delta = 0
                    st.metric("🔌 Voltage", f"{current_data['voltage']:.1f} V",
                              f"{voltage_delta:+.1f}" if not st.session_state.machine_stopped else "STOPPED",
                              delta_color="normal")

        with col2:
            # Health score
            health_score = sensor.calculate_health_score(current_data)
            current_data['health_score'] = health_score

            # Health gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={'text': "Health Score", 'font': {'color': 'white'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#4caf50" if health_score > 70 else "#ff9800" if health_score > 40 else "#f44336"},
                    'bgcolor': "rgba(0,0,0,0.5)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(244,67,54,0.3)"},
                        {'range': [40, 70], 'color': "rgba(255,152,0,0.3)"},
                        {'range': [70, 100], 'color': "rgba(76,175,80,0.3)"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                                    font={'color': 'white'})
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Health status
            if st.session_state.machine_stopped:
                st.markdown('<p class="status-critical">🔴 MACHINE STOPPED - Emergency Mode</p>',
                            unsafe_allow_html=True)
            elif health_score >= 70:
                st.markdown('<p class="status-healthy">✅ SYSTEM HEALTHY - Normal Operation</p>',
                            unsafe_allow_html=True)
            elif health_score >= 40:
                st.markdown('<p class="status-warning">⚠️ MODERATE WEAR - Monitor Closely</p>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-critical">🔴 CRITICAL CONDITION - Immediate Action Required</p>',
                            unsafe_allow_html=True)

        # Store data
        add_sensor_data_to_db(current_data)

        # ====================================================================
        # ALERTS AND REACTIONS
        # ====================================================================
        st.subheader("🚨 System Status & Alerts")

        # Calculate likelihood
        likelihood = thresholds.calculate_likelihood(current_data)

        # Train predictor and get predictions - Use ALL available data
        historical_df = load_and_prepare_data()
        if len(historical_df) > 10 and not st.session_state.machine_stopped:
            predictor.train_model(historical_df)
            predictions = predictor.predict_ttf(current_data)
        else:
            predictions = None

        # Generate alerts
        alerts = alert_gen.generate_alerts(current_data, likelihood, predictions, health_score)

        # Send email automatically for urgent/high priority or not normal risk
        if likelihood[
            'risk_level'] != 'NORMAL' and not st.session_state.email_sent_for_risk and not st.session_state.machine_stopped:
            # Get failure details for email
            failure_details = predictor.predict_failure_details(current_data, historical_df)
            email_sent = alert_gen.send_risk_email(likelihood['risk_level'], likelihood['likelihood'], current_data,
                                                   failure_details)
            if email_sent:
                st.session_state.email_sent_for_risk = True
                st.info("📧 Automatic email alert sent to maintenance team regarding system risk status.")

        # Display alerts and react
        if alerts:
            for alert in alerts:
                if alert['type'] == 'critical':
                    st.markdown(f"""
                    <div class="alert-critical">
                        <strong>🔴 {alert['severity']} ALERT</strong><br>
                        {alert['message']}<br>
                        <strong>Recommended Action:</strong> {alert['action']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Trigger visual and audio alert for critical
                    if st.session_state.last_alert_time is None or \
                            (datetime.now() - st.session_state.last_alert_time).seconds > 300:
                        st.session_state.last_alert_time = datetime.now()
                        st.toast("🔴 CRITICAL ALERT! System failure imminent!", icon="🚨")

                        # Send email for critical alerts
                        if st.button("📧 Send Email Alert to Maintenance"):
                            if alert_gen.send_email(alert, current_data):
                                st.success("✅ Email alert sent to maintenance team!")
                            else:
                                st.error("Failed to send email. Check configuration.")

                elif alert['type'] == 'warning':
                    st.markdown(f"""
                    <div class="alert-warning">
                        <strong>⚠️ {alert['severity']} ALERT</strong><br>
                        {alert['message']}<br>
                        <strong>Recommended Action:</strong> {alert['action']}
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning(alert['message'])

                elif alert['type'] == 'info':
                    st.info(alert['message'])

            # Critical visual indicators
            if any(a['type'] == 'critical' for a in alerts) and not st.session_state.machine_stopped:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                    <h2 style="color: white; margin: 0;">🚨 EMERGENCY MODE ACTIVATED 🚨</h2>
                    <p style="color: white;">Immediate maintenance intervention required!</p>
                </div>
                """, unsafe_allow_html=True)

                # Add siren effect in console
                print("\n" + "=" * 50)
                print("🔴 CRITICAL ALERT - SYSTEM FAILURE IMMINENT 🔴")
                print("=" * 50)

        else:
            st.success("✅ No active alerts. All systems operating within normal parameters.")
            if not st.session_state.machine_stopped:
                st.balloons()

        # ====================================================================
        # RECOMMENDATIONS
        # ====================================================================
        recommendations = recommender.generate_recommendations(alerts, current_data, predictions)

        if recommendations:
            st.subheader("💡 Prescriptive Recommendations")

            for rec in recommendations:
                priority_color = {"URGENT": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(rec['priority'], "🟢")
                with st.expander(f"{priority_color} {rec['priority']} PRIORITY: {rec['issue']}"):
                    st.markdown("**Immediate Actions Required:**")
                    for action in rec['actions']:
                        st.markdown(action)
                    st.info(f"⏱️ Estimated Downtime: {rec['downtime']}")

        # ====================================================================
        # VISUAL ANALYTICS
        # ====================================================================
        st.subheader("📊 Visual Analytics")

        # Get historical data
        historical_df = load_and_prepare_data()

        if len(historical_df) > 0:
            # Time series plot
            fig_trend = make_subplots(rows=2, cols=2,
                                      subplot_titles=('Temperature Trend', 'Vibration Trend',
                                                      'Pressure Trend', 'Health Score Trend'))

            # Temperature
            fig_trend.add_trace(go.Scatter(x=historical_df['timestamp'].values[::-1] if len(historical_df) > 0 else [],
                                           y=historical_df['temperature'].values[::-1] if len(
                                               historical_df) > 0 else [],
                                           mode='lines', name='Temperature',
                                           line=dict(color='red', width=2)), row=1, col=1)
            fig_trend.add_hline(y=85, line_dash="dash", line_color="red", row=1, col=1)
            fig_trend.add_hline(y=75, line_dash="dash", line_color="orange", row=1, col=1)

            # Vibration
            fig_trend.add_trace(go.Scatter(x=historical_df['timestamp'].values[::-1] if len(historical_df) > 0 else [],
                                           y=historical_df['vibration'].values[::-1] if len(historical_df) > 0 else [],
                                           mode='lines', name='Vibration',
                                           line=dict(color='orange', width=2)), row=1, col=2)
            fig_trend.add_hline(y=8.5, line_dash="dash", line_color="red", row=1, col=2)
            fig_trend.add_hline(y=6.5, line_dash="dash", line_color="orange", row=1, col=2)

            # Pressure
            fig_trend.add_trace(go.Scatter(x=historical_df['timestamp'].values[::-1] if len(historical_df) > 0 else [],
                                           y=historical_df['pressure'].values[::-1] if len(historical_df) > 0 else [],
                                           mode='lines', name='Pressure',
                                           line=dict(color='yellow', width=2)), row=2, col=1)
            fig_trend.add_hline(y=95, line_dash="dash", line_color="red", row=2, col=1)
            fig_trend.add_hline(y=85, line_dash="dash", line_color="orange", row=2, col=1)

            # Health Score
            if 'health_score' in historical_df.columns:
                fig_trend.add_trace(
                    go.Scatter(x=historical_df['timestamp'].values[::-1] if len(historical_df) > 0 else [],
                               y=historical_df['health_score'].values[::-1] if len(historical_df) > 0 else [],
                               mode='lines', name='Health Score',
                               line=dict(color='green', width=3)), row=2, col=2)

            fig_trend.update_layout(height=600, showlegend=False,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0.3)",
                                    font=dict(color='white'))
            fig_trend.update_xaxes(gridcolor='rgba(255,255,255,0.1)', color='white')
            fig_trend.update_yaxes(gridcolor='rgba(255,255,255,0.1)', color='white')

            st.plotly_chart(fig_trend, use_container_width=True)

    # ========================================================================
    # PATTERN ANALYSIS PAGE
    # ========================================================================
    elif page == "🔍 Pattern Analysis":
        st.header("🔍 Machine Data Pattern Analysis")

        historical_df = load_and_prepare_data()

        if len(historical_df) > 0:
            st.subheader("📈 Anomaly Detection")

            # Analyze current state
            current = historical_df.iloc[0].to_dict()
            anomalies = pattern_analyzer.detect_anomalies(current)

            if anomalies:
                for anomaly in anomalies:
                    if "CRITICAL" in anomaly:
                        st.error(anomaly)
                    elif "WARNING" in anomaly:
                        st.warning(anomaly)
                    else:
                        st.info(anomaly)
            else:
                st.success("✅ No anomalies detected in current readings")

            st.subheader("🔄 Degradation Pattern Analysis")
            pattern = pattern_analyzer.identify_degradation_pattern(historical_df)

            if "degradation" in pattern.lower():
                st.warning(f"⚠️ {pattern}")
            else:
                st.info(f"ℹ️ {pattern}")

            # Pattern visualization
            st.subheader("📊 Pattern Visualization")

            # Calculate rolling averages
            window_size = min(20, len(historical_df))
            historical_df['temp_ma'] = historical_df['temperature'].rolling(window=window_size).mean()
            historical_df['vib_ma'] = historical_df['vibration'].rolling(window=window_size).mean()

            fig_pattern = go.Figure()
            fig_pattern.add_trace(go.Scatter(x=historical_df['timestamp'].values[::-1],
                                             y=historical_df['temp_ma'].values[::-1],
                                             mode='lines', name='Temperature (20-pt MA)',
                                             line=dict(color='red', width=3)))
            fig_pattern.add_trace(go.Scatter(x=historical_df['timestamp'].values[::-1],
                                             y=historical_df['vib_ma'].values[::-1],
                                             mode='lines', name='Vibration (20-pt MA)',
                                             line=dict(color='orange', width=3)))

            fig_pattern.update_layout(title="Moving Average Trends - Pattern Detection",
                                      xaxis_title="Time",
                                      yaxis_title="Value",
                                      paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0.3)",
                                      font=dict(color='white'))
            st.plotly_chart(fig_pattern, use_container_width=True)
        else:
            st.warning(
                "No data available. Please upload a dataset or generate sample data using the Data Management panel in the sidebar.")

    # ========================================================================
    # FAILURE PREDICTION PAGE
    # ========================================================================
    elif page == "⚠️ Failure Prediction":
        st.header("⚠️ Failure Prediction & Time-to-Failure Analysis")

        # Load all available data
        historical_df = load_and_prepare_data()

        if len(historical_df) == 0:
            st.warning("📊 No data available for predictions!")
            st.info("Please use the 'Data Management' panel in the sidebar to:")
            st.markdown("""
            - Click **'Generate Sample Dataset'** to create test data
            - Or **upload your own CSV file** with sensor readings
            """)

            # Show expected format
            with st.expander("📁 Expected CSV Format"):
                st.markdown("""
                Your CSV should contain these columns:
                - **temperature** (float, 20-120°C)
                - **vibration** (float, 0-15 mm/s)
                - **pressure** (float, 40-120 PSI)
                - **rpm** (float/int, 2500-6000)
                - **current** (float, 150-400 A)
                - **voltage** (float, 350-480 V)

                Optional columns:
                - timestamp (datetime)
                - health_score (0-100)
                """)
            return

        if len(historical_df) > 0:
            # Get the most recent reading
            current = historical_df.iloc[0].to_dict()

            # Show data status
            st.info(f"📊 Using {len(historical_df)} records for prediction model")

            if st.session_state.custom_dataset_loaded:
                st.success("✅ Custom dataset loaded and active for predictions!")

            if st.session_state.machine_stopped:
                st.warning(
                    "🛑 Machine is currently stopped. Failure prediction is not applicable while the machine is not running.")

                # Show stop information
                st.subheader("📋 Machine Stop Information")
                st.info(f"""
                **Machine Status:** STOPPED
                **Stop Time:** {st.session_state.stop_machine_time.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.stop_machine_time else 'Unknown'}
                **Action Required:** Restart machine from sidebar to resume monitoring and predictions
                """)

                # Display current parameters anyway
                st.subheader("📊 Last Recorded Parameters (Before Stop)")
                param_col1, param_col2, param_col3 = st.columns(3)
                with param_col1:
                    st.metric("Temperature", f"{current['temperature']:.1f}°C")
                with param_col2:
                    st.metric("Vibration", f"{current['vibration']:.2f} mm/s")
                with param_col3:
                    st.metric("Pressure", f"{current['pressure']:.1f} PSI")

            elif len(historical_df) >= 10:
                # Train model with all available data
                predictor.train_model(historical_df)
                predictions = predictor.predict_ttf(current)

                # Get detailed failure prediction with date and time
                failure_details = predictor.predict_failure_details(current, historical_df)

                # Display detailed failure prediction
                st.subheader("🔮 Detailed Failure Prediction")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(30,30,46,0.9); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                        <h4>📋 Predicted Failure Type</h4>
                        <p style="font-size: 1.2rem; font-weight: bold; color: {'#f44336' if failure_details['severity'] == 'CRITICAL' else '#ff9800'}">
                            {failure_details['predicted_failure_type']}
                        </p>
                        <hr>
                        <h4>📊 Failure Probability</h4>
                        <div style="background: #333; border-radius: 10px; height: 20px; margin: 10px 0;">
                            <div style="background: linear-gradient(90deg, #f44336, #ff9800); width: {failure_details['probability'] * 100}%; height: 20px; border-radius: 10px;"></div>
                        </div>
                        <p><strong>{failure_details['probability']:.1%}</strong> chance of failure</p>
                        <hr>
                        <h4>⏰ Expected Timeframe</h4>
                        <p style="font-size: 1.1rem;"><strong>{failure_details['timeframe']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="background: rgba(30,30,46,0.9); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                        <h4>⚠️ Severity Level</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: {'#f44336' if failure_details['severity'] == 'CRITICAL' else '#ff9800' if failure_details['severity'] == 'HIGH' else '#4caf50'}">
                            {failure_details['severity']}
                        </p>
                        <hr>
                        <h4>🔧 Affected Components</h4>
                        <ul>
                            {''.join([f'<li>{comp}</li>' for comp in failure_details['affected_components']])}
                        </ul>
                        <hr>
                        <h4>💡 Recommended Action</h4>
                        <p style="background: #444; padding: 0.5rem; border-radius: 5px;">{failure_details['recommended_action']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Display Predicted Failure Date and Time
                st.subheader("📅 Predicted Failure Date & Time")

                predicted_dt = failure_details['predicted_failure_datetime']
                current_time = failure_details['prediction_timestamp']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Prediction Made At", current_time.strftime('%Y-%m-%d %H:%M:%S'))

                with col2:
                    if predicted_dt:
                        st.metric("Predicted Failure Date & Time", predicted_dt.strftime('%Y-%m-%d %H:%M:%S'))

                with col3:
                    hours_until = failure_details['predicted_ttf_hours']
                    st.metric("Hours Until Failure", f"{hours_until:.1f} hours")

                # Create countdown style visualization for failure date
                if predicted_dt:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                                padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                        <h3 style="color: white; margin: 0;">Predicted Failure Will Occur At:</h3>
                        <p style="font-size: 2rem; font-weight: bold; color: #ff6b6b; margin: 0.5rem 0;">
                            {predicted_dt.strftime('%A, %B %d, %Y at %H:%M:%S')}
                        </p>
                        <p style="color: white;">This is an estimate based on current sensor readings and historical failure patterns.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Time-to-Failure Analysis
                st.subheader("⏰ Time-to-Failure Analysis")

                if predictions:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        ttf_value = predictions['predicted_ttf']
                        st.metric("Predicted Time to Failure", f"{ttf_value:.1f} hours",
                                  delta=f"±{ttf_value * 0.3:.1f} hrs")

                        # TTF Gauge
                        fig_ttf = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=ttf_value,
                            title={'text': "Hours Until Failure", 'font': {'color': 'white'}},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 168], 'tickcolor': 'white'},
                                'bar': {
                                    'color': "#f44336" if ttf_value < 24 else "#ff9800" if ttf_value < 72 else "#4caf50"},
                                'steps': [
                                    {'range': [0, 24], 'color': "rgba(244,67,54,0.3)"},
                                    {'range': [24, 72], 'color': "rgba(255,152,0,0.3)"},
                                    {'range': [72, 168], 'color': "rgba(76,175,80,0.3)"}
                                ]
                            }
                        ))
                        fig_ttf.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                                              font={'color': 'white'})
                        st.plotly_chart(fig_ttf, use_container_width=True)

                    with col2:
                        st.metric("Prediction Confidence", f"{predictions['confidence']:.1%}")
                        st.metric("Severity Factor", f"{predictions['severity_factor']:.2f}")

                    with col3:
                        if predictions['predicted_ttf'] < 24:
                            st.error("🔴 IMMINENT FAILURE - Immediate action required!")
                            st.progress(0.95)
                        elif predictions['predicted_ttf'] < 72:
                            st.warning("⚠️ Failure expected within 3 days - Schedule maintenance")
                            st.progress(0.65)
                        else:
                            st.success("✅ Acceptable risk level - Continue monitoring")
                            st.progress(0.25)

                # Historical profiles
                st.subheader("📚 Historical Failure Profile Mapping")
                profiles = predictor.map_historical_profiles(current)

                if profiles:
                    for profile in profiles:
                        with st.expander(f"🔍 {profile['type']} - {profile['probability']:.0%} Match"):
                            st.markdown(f"**Pattern:** {profile['pattern']}")
                            st.markdown(f"**Typical Time to Failure:** {profile['typical_ttf']} hours")
                            st.progress(profile['probability'])
                else:
                    st.info("No matching historical failure profiles found for current conditions")

                # Add current parameter analysis
                st.subheader("📊 Current Parameter Analysis")

                param_col1, param_col2, param_col3 = st.columns(3)

                with param_col1:
                    temp_status = "🔴 CRITICAL" if current['temperature'] > 85 else "🟡 WARNING" if current[
                                                                                                      'temperature'] > 75 else "🟢 NORMAL"
                    st.metric("Temperature", f"{current['temperature']:.1f}°C",
                              delta=f"{current['temperature'] - 65:+.1f}°C")
                    st.markdown(f"**Status:** {temp_status}")

                with param_col2:
                    vib_status = "🔴 CRITICAL" if current['vibration'] > 8.5 else "🟡 WARNING" if current[
                                                                                                    'vibration'] > 6.5 else "🟢 NORMAL"
                    st.metric("Vibration", f"{current['vibration']:.2f} mm/s",
                              delta=f"{current['vibration'] - 4.5:+.2f}")
                    st.markdown(f"**Status:** {vib_status}")

                with param_col3:
                    press_status = "🔴 CRITICAL" if current['pressure'] > 95 else "🟡 WARNING" if current[
                                                                                                    'pressure'] > 85 else "🟢 NORMAL"
                    st.metric("Pressure", f"{current['pressure']:.1f} PSI", delta=f"{current['pressure'] - 70:+.1f}")
                    st.markdown(f"**Status:** {press_status}")

            else:
                st.warning(f"⚠️ Need more data for accurate predictions! Currently have {len(historical_df)} records.")
                st.info("Please upload more sensor data using the Data Management panel in the sidebar.")
                st.progress(min(len(historical_df) / 50, 1.0))
                st.caption(f"Progress: {len(historical_df)}/50 records needed for basic predictions")

    # ========================================================================
    # RISK ANALYSIS PAGE
    # ========================================================================
    elif page == "📈 Risk Analysis":
        st.header("📈 Statistical Risk Analysis")

        historical_df = load_and_prepare_data()

        if len(historical_df) > 0:
            current = historical_df.iloc[0].to_dict()
            likelihood = thresholds.calculate_likelihood(current)

            st.subheader("📊 Current Failure Likelihood")

            col1, col2 = st.columns(2)

            with col1:
                fig_likelihood = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=likelihood['score'],
                    title={'text': "Failure Likelihood Score", 'font': {'color': 'white'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bar': {'color': "#f44336" if likelihood['score'] > 70 else "#ff9800" if likelihood[
                                                                                                     'score'] > 40 else "#4caf50"},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(76,175,80,0.3)"},
                            {'range': [40, 70], 'color': "rgba(255,152,0,0.3)"},
                            {'range': [70, 100], 'color': "rgba(244,67,54,0.3)"}
                        ]
                    }
                ))
                fig_likelihood.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                                             font={'color': 'white'})
                st.plotly_chart(fig_likelihood, use_container_width=True)

            with col2:
                st.markdown("### Risk Assessment")
                if st.session_state.machine_stopped:
                    st.info("🛑 **MACHINE STOPPED**\n\nRisk assessment not applicable - Machine is not operational")
                    st.progress(0)
                elif likelihood['risk_level'] == 'CRITICAL':
                    st.error(
                        f"🔴 **CRITICAL RISK**\n\nFailure probability: {likelihood['likelihood']:.1%}\nImmediate action required!")
                    st.progress(0.95)
                elif likelihood['risk_level'] == 'WARNING':
                    st.warning(
                        f"⚠️ **ELEVATED RISK**\n\nFailure probability: {likelihood['likelihood']:.1%}\nSchedule inspection soon")
                    st.progress(0.65)
                else:
                    st.success(
                        f"✅ **NORMAL RISK**\n\nFailure probability: {likelihood['likelihood']:.1%}\nContinue normal monitoring")
                    st.progress(0.25)

            # Parameter risk matrix
            st.subheader("📊 Parameter Risk Matrix")

            risk_data = []
            for param, thresh in thresholds.thresholds.items():
                if param in current:
                    value = current[param]
                    if st.session_state.machine_stopped:
                        risk = "STOPPED"
                    elif value >= thresh['critical']:
                        risk = "CRITICAL"
                    elif value >= thresh['warning']:
                        risk = "WARNING"
                    else:
                        risk = "NORMAL"

                    risk_data.append({
                        'Parameter': param.upper(),
                        'Current': f"{value:.1f}",
                        'Warning': thresh['warning'],
                        'Critical': thresh['critical'],
                        'Risk Level': risk
                    })

            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True)

            # Distribution analysis
            if not st.session_state.machine_stopped and len(historical_df) > 10:
                st.subheader("📈 Statistical Distribution Analysis")

                fig_dist = make_subplots(rows=2, cols=2,
                                         subplot_titles=('Temperature Distribution', 'Vibration Distribution',
                                                         'Pressure Distribution', 'RPM Distribution'))

                fig_dist.add_trace(go.Histogram(x=historical_df['temperature'], nbinsx=30,
                                                marker_color='rgba(244,67,54,0.7)'), row=1, col=1)
                fig_dist.add_vline(x=current['temperature'], line_dash="dash",
                                   line_color="red", row=1, col=1)

                fig_dist.add_trace(go.Histogram(x=historical_df['vibration'], nbinsx=30,
                                                marker_color='rgba(255,152,0,0.7)'), row=1, col=2)
                fig_dist.add_vline(x=current['vibration'], line_dash="dash",
                                   line_color="red", row=1, col=2)

                fig_dist.add_trace(go.Histogram(x=historical_df['pressure'], nbinsx=30,
                                                marker_color='rgba(76,175,80,0.7)'), row=2, col=1)
                fig_dist.add_vline(x=current['pressure'], line_dash="dash",
                                   line_color="red", row=2, col=1)

                fig_dist.add_trace(go.Histogram(x=historical_df['rpm'], nbinsx=30,
                                                marker_color='rgba(33,150,243,0.7)'), row=2, col=2)
                fig_dist.add_vline(x=current['rpm'], line_dash="dash",
                                   line_color="red", row=2, col=2)

                fig_dist.update_layout(height=600, showlegend=False,
                                       paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0.3)",
                                       font=dict(color='white'))
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning(
                "No data available. Please upload a dataset or generate sample data using the Data Management panel in the sidebar.")

    # ========================================================================
    # ALERTS & ACTIONS PAGE
    # ========================================================================
    elif page == "🚨 Alerts & Actions":
        st.header("🚨 Active Alerts & Action Items")

        # Get recent alerts
        conn_alerts = sqlite3.connect('industrial_monitoring.db')
        try:
            alerts_df = pd.read_sql_query("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50", conn_alerts)
        except:
            alerts_df = pd.DataFrame()
        conn_alerts.close()

        if len(alerts_df) > 0:
            st.subheader("📋 Recent Alerts")
            st.dataframe(alerts_df[['timestamp', 'severity', 'message', 'acknowledged']],
                         use_container_width=True)
        else:
            st.info("No recent alerts. System operating normally.")

        # Current critical status
        historical_df = load_and_prepare_data()

        if len(historical_df) > 0:
            current = historical_df.iloc[0].to_dict()
            likelihood = thresholds.calculate_likelihood(current)

            if likelihood['risk_level'] == 'CRITICAL' and not st.session_state.machine_stopped:
                st.markdown("""
                <div class="alert-critical">
                    <h3>🚨 CRITICAL SITUATION DETECTED</h3>
                    <p>Immediate action required to prevent equipment failure!</p>
                </div>
                """, unsafe_allow_html=True)

                # Emergency checklist
                st.subheader("📋 Emergency Response Checklist")

                with st.form("emergency_checklist"):
                    st.write("Complete the following actions:")
                    action1 = st.checkbox("🛑 Stop machine operation immediately")
                    action2 = st.checkbox("📢 Alert maintenance team")
                    action3 = st.checkbox("🔧 Prepare diagnostic tools")
                    action4 = st.checkbox("📦 Locate replacement parts")
                    action5 = st.checkbox("📝 Document current readings")

                    if st.form_submit_button("Confirm Actions Completed"):
                        st.success("✅ Emergency response initiated. Maintenance team notified.")

                        # Send email notification
                        alert = {
                            'type': 'critical',
                            'severity': 'CRITICAL',
                            'message': 'Emergency response initiated for critical system failure',
                            'action': 'Emergency maintenance in progress'
                        }
                        alert_gen.send_email(alert, current)

            elif st.session_state.machine_stopped:
                st.info("🛑 Machine is currently stopped. No active alerts while machine is not running.")

        # Maintenance scheduler - WITH EMAIL ON SCHEDULE
        st.subheader("📅 Scheduled Maintenance Actions")

        with st.form("maintenance_scheduler"):
            st.write("Schedule maintenance tasks:")
            task = st.text_input("Maintenance Task", placeholder="e.g., Bearing replacement, Cooling system check")
            priority = st.selectbox("Priority", ["URGENT", "HIGH", "MEDIUM", "LOW"])
            scheduled_date = st.date_input("Scheduled Date")

            submitted = st.form_submit_button("Schedule Maintenance")

            if submitted and task:
                # Get current data for email
                current_data_for_email = sensor.read_sensors()

                # Save to database
                c = conn.cursor()
                c.execute("""INSERT INTO maintenance_schedule (timestamp, task, priority, scheduled_date, email_sent, completed)
                             VALUES (?, ?, ?, ?, ?, ?)""",
                          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), task, priority,
                           scheduled_date.strftime('%Y-%m-%d'), False, False))
                conn.commit()

                # Send email for scheduled maintenance
                email_sent = alert_gen.send_maintenance_email(task, priority, scheduled_date, current_data_for_email)

                if email_sent:
                    st.success(f"✅ Maintenance scheduled: {task} - {priority} priority for {scheduled_date}")
                    st.info(f"📧 Email notification sent to {st.session_state.email_config['recipient']}")
                else:
                    st.warning(f"✅ Maintenance scheduled: {task} - {priority} priority for {scheduled_date}")
                    st.warning("⚠️ Email notification failed. Check email configuration.")
            elif submitted and not task:
                st.error("Please enter a maintenance task")

        # Display scheduled maintenance
        try:
            scheduled_df = pd.read_sql_query("SELECT * FROM maintenance_schedule ORDER BY scheduled_date DESC LIMIT 10",
                                             conn)
            if len(scheduled_df) > 0:
                st.subheader("📋 Upcoming Maintenance Schedule")
                st.dataframe(scheduled_df[['task', 'priority', 'scheduled_date', 'email_sent']],
                             use_container_width=True)
        except:
            pass

    # ========================================================================
    # REPORTS PAGE
    # ========================================================================
    elif page == "📊 Reports":
        st.header("📊 System Reports & Analytics")

        # Get historical data
        historical_df = load_and_prepare_data()

        if len(historical_df) > 0:
            # Summary statistics
            st.subheader("📈 Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_health = historical_df['health_score'].mean() if 'health_score' in historical_df.columns else 50
                st.metric("Average Health Score", f"{avg_health:.1f}")

            with col2:
                max_temp = historical_df['temperature'].max()
                st.metric("Max Temperature", f"{max_temp:.1f}°C")

            with col3:
                max_vib = historical_df['vibration'].max()
                st.metric("Max Vibration", f"{max_vib:.2f} mm/s")

            with col4:
                total_readings = len(historical_df)
                st.metric("Total Data Points", f"{total_readings:,}")

            # Health score trend
            if 'health_score' in historical_df.columns:
                st.subheader("📊 Health Score Trend")
                fig_health = go.Figure()
                fig_health.add_trace(go.Scatter(
                    x=historical_df['timestamp'].values[::-1] if 'timestamp' in historical_df.columns else list(
                        range(len(historical_df))),
                    y=historical_df['health_score'].values[::-1],
                    mode='lines', name='Health Score',
                    fill='tozeroy',
                    line=dict(color='#4caf50', width=3)))
                fig_health.update_layout(height=400,
                                         paper_bgcolor="rgba(0,0,0,0)",
                                         plot_bgcolor="rgba(0,0,0,0.3)",
                                         font=dict(color='white'))
                st.plotly_chart(fig_health, use_container_width=True)

            # Machine stop events
            try:
                stop_log_df = pd.read_sql_query("SELECT * FROM machine_stop_log ORDER BY timestamp DESC LIMIT 10", conn)
                if len(stop_log_df) > 0:
                    st.subheader("🛑 Machine Stop History")
                    st.dataframe(stop_log_df[['timestamp', 'reason', 'restored_at']], use_container_width=True)
            except:
                pass

            # Export data
            st.subheader("📥 Export Data")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export All Sensor Data"):
                    csv = historical_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"sensor_data_export_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

            with col2:
                # Generate report
                if st.button("Generate PDF Report"):
                    st.info("PDF report generation would be implemented here")
                    st.warning("For demo purposes, CSV export is available above")
        else:
            st.warning(
                "No data available. Please upload a dataset or generate sample data using the Data Management panel in the sidebar.")

    conn.close()


if __name__ == "__main__":
    main()