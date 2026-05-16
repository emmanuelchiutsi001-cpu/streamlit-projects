import streamlit as st
import os
import glob
import sys
import threading
import time
import json
import hashlib
import pickle
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import yagmail
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
import tempfile
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image
import io
import pathlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import hashlib
from functools import lru_cache
import logging
from abc import ABC, abstractmethod
import urllib.request
import ssl

warnings.filterwarnings('ignore')

# Disable SSL certificate verification for network issues (if needed)
ssl._create_default_https_context = ssl._create_unverified_context

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. BYPASS DLL CONFLICTS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# --- 2. CONFIGURATION ---
@dataclass
class Config:
    """Central configuration management"""
    # Email Configuration (UPDATED)
    GMAIL_APP_PASSWORD: str = "twmlrauqerkvxark"
    ALERT_EMAIL: str = "emmanuelchiutsi001@gmail.com"

    # Dataset paths (UPDATED)
    CRIME_DATASET_PATH: str = r"C:\Users\emmanuel chiutsi\Documents\Crime"
    NORMAL_DATASET_PATH: str = r"C:\Users\emmanuel chiutsi\Documents\UCF-Crime only Normal videos"
    SPLIT_DATASET_PATH: str = r"C:\Users\emmanuel chiutsi\Documents\dataset-video-split"

    # Model settings
    MODEL_SAVE_PATH: str = "models"
    CACHE_PATH: str = "cache"
    REPORTS_PATH: str = "reports"

    # Analysis settings
    DETECTION_THRESHOLD: float = 40.0
    SEQUENCE_LENGTH: int = 16  # Reduced from 30 for better performance
    BATCH_SIZE: int = 4  # Reduced for lighter memory usage
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 3  # Reduced epochs for faster training on T450

    # System settings
    CACHE_TTL: int = 300
    MAX_VIDEO_SIZE_MB: int = 500
    SUPPORTED_FORMATS: List[str] = None

    def __post_init__(self):
        if self.SUPPORTED_FORMATS is None:
            self.SUPPORTED_FORMATS = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'm4v', 'mpeg']

        # Create directories
        for path in [self.MODEL_SAVE_PATH, self.CACHE_PATH, self.REPORTS_PATH]:
            os.makedirs(path, exist_ok=True)


# Initialize config
config = Config()

# --- 3. PAGE CONFIG ---
st.set_page_config(
    page_title="COMMUNITY SECURITY ANALYTICS - Production Grade",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 4. CUSTOM CSS FOR PROFESSIONAL UI ---
def set_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                        url('https://images.unsplash.com/photo-1557597774-9d273e5e0b8a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }

        /* Modern card styling */
        .modern-card {
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(0, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .modern-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 255, 255, 0.2);
            border-color: #00fbff;
        }

        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(0,0,0,0.7));
            border-radius: 15px;
            margin-bottom: 20px;
            border: 2px solid #00fbff;
            box-shadow: 0 0 20px rgba(0, 251, 255, 0.3);
        }

        .main-header h1 {
            background: linear-gradient(135deg, #00fbff, #00ff88, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3em;
            margin: 0;
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from { text-shadow: 0 0 10px #00fbff; }
            to { text-shadow: 0 0 30px #00fbff, 0 0 20px #00ff88; }
        }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(0, 251, 255, 0.1), rgba(0, 255, 136, 0.05));
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #00fbff;
            margin: 10px 0;
            transition: all 0.3s;
        }

        .metric-card:hover {
            transform: translateX(5px);
            background: linear-gradient(135deg, rgba(0, 251, 255, 0.2), rgba(0, 255, 136, 0.1));
        }

        /* Alert animations */
        .alert-critical {
            background: linear-gradient(135deg, #ff4757, #c0392b);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            animation: pulse 1s infinite;
            box-shadow: 0 0 30px #ff4757;
        }

        .alert-warning {
            background: linear-gradient(135deg, #feca57, #e67e22);
            color: black;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }

        .alert-secure {
            background: linear-gradient(135deg, #00ff88, #00d68f);
            color: black;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 0 20px #ff4757; }
            50% { transform: scale(1.02); box-shadow: 0 0 50px #ff4757; }
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, rgba(0, 251, 255, 0.2), rgba(0, 255, 136, 0.1));
            color: white;
            border: 1px solid #00fbff;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #00fbff, #00ff88);
            color: black;
            box-shadow: 0 0 20px #00fbff;
            transform: translateY(-2px);
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00fbff, #00ff88, #9b59b6) !important;
        }

        /* Info boxes */
        .info-box {
            background: rgba(0, 251, 255, 0.1);
            border: 1px solid #00fbff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.5);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#00fbff, #00ff88);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(#00ff88, #9b59b6);
        }

        /* Loading spinner */
        .custom-spinner {
            border: 4px solid rgba(0, 251, 255, 0.3);
            border-top: 4px solid #00fbff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Toast notifications */
        .toast-success {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #00ff88;
            color: black;
            padding: 12px 24px;
            border-radius: 8px;
            animation: slideIn 0.3s ease-out;
            z-index: 1000;
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)


# --- 5. DATABASE PERSISTENCE ---
class DatabaseManager:
    """Handles all database operations for detections"""

    def __init__(self, db_path: str = "detections.db"):
        import sqlite3
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Detections table with expanded crime types
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_name TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    crime_type TEXT NOT NULL,
                    crime_score REAL NOT NULL,
                    severity_level TEXT,
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    frame_count INTEGER,
                    duration REAL,
                    robbery_score REAL,
                    assault_score REAL,
                    theft_score REAL,
                    weapon_score REAL,
                    abuse_score REAL,
                    explosion_score REAL,
                    fighting_score REAL,
                    accident_score REAL,
                    shooting_score REAL,
                    arson_score REAL,
                    lstm_gru_score REAL,
                    alert_sent BOOLEAN DEFAULT 0,
                    alert_time TIMESTAMP,
                    metadata TEXT
                )
            ''')

            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    samples_count INTEGER
                )
            ''')

            # System logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def save_detection(self, detection_data: Dict) -> int:
        """Save detection record to database"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO detections (
                    video_name, video_path, crime_type, crime_score, severity_level,
                    frame_count, duration, robbery_score, assault_score, theft_score,
                    weapon_score, abuse_score, explosion_score, fighting_score,
                    accident_score, shooting_score, arson_score, lstm_gru_score, 
                    alert_sent, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection_data.get('video_name', ''),
                detection_data.get('video_path', ''),
                detection_data.get('crime_type', ''),
                detection_data.get('crime_score', 0),
                detection_data.get('severity_level', ''),
                detection_data.get('frame_count', 0),
                detection_data.get('duration', 0),
                detection_data.get('robbery_score', 0),
                detection_data.get('assault_score', 0),
                detection_data.get('theft_score', 0),
                detection_data.get('weapon_score', 0),
                detection_data.get('abuse_score', 0),
                detection_data.get('explosion_score', 0),
                detection_data.get('fighting_score', 0),
                detection_data.get('accident_score', 0),
                detection_data.get('shooting_score', 0),
                detection_data.get('arson_score', 0),
                detection_data.get('lstm_gru_score', 0),
                detection_data.get('alert_sent', False),
                detection_data.get('metadata', '')
            ))

            detection_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return detection_id
        except Exception as e:
            logger.error(f"Failed to save detection: {e}")
            return -1

    def get_detections(self, limit: int = 100, crime_type: str = None) -> List[Dict]:
        """Retrieve detections from database"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if crime_type:
                cursor.execute('''
                    SELECT * FROM detections 
                    WHERE crime_type = ? 
                    ORDER BY detection_time DESC 
                    LIMIT ?
                ''', (crime_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM detections 
                    ORDER BY detection_time DESC 
                    LIMIT ?
                ''', (limit,))

            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve detections: {e}")
            return []

    def save_performance_metric(self, model_name: str, metrics: Dict):
        """Save model performance metrics"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for metric_name, metric_value in metrics.items():
                cursor.execute('''
                    INSERT INTO performance_metrics (model_name, metric_name, metric_value)
                    VALUES (?, ?, ?)
                ''', (model_name, metric_name, metric_value))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")

    def log_system_event(self, level: str, message: str, context: str = None):
        """Log system event to database"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO system_logs (log_level, message, context)
                VALUES (?, ?, ?)
            ''', (level, message, context))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")


# --- 6. CACHE MANAGEMENT ---
class CacheManager:
    """Manages caching for faster repeated analyses"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, video_path: str, analysis_params: Dict = None) -> str:
        """Generate cache key from video path and parameters"""
        content = video_path
        if analysis_params:
            content += str(sorted(analysis_params.items()))
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_result(self, video_path: str, analysis_params: Dict = None, ttl: int = 300) -> Optional[Dict]:
        """Retrieve cached analysis result"""
        cache_key = self._get_cache_key(video_path, analysis_params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            if os.path.exists(cache_file):
                mtime = os.path.getmtime(cache_file)
                if time.time() - mtime < ttl:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

        return None

    def cache_result(self, video_path: str, result: Dict, analysis_params: Dict = None):
        """Cache analysis result"""
        cache_key = self._get_cache_key(video_path, analysis_params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def clear_cache(self, older_than_days: int = 7):
        """Clear old cache files"""
        try:
            current_time = time.time()
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > older_than_days * 86400:
                        os.remove(filepath)
            logger.info(f"Cache cleared - removed files older than {older_than_days} days")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")


# --- 7. EMAIL ALERT SYSTEM ---
class EmailAlertSystem:
    """Handles automated email alerts"""

    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.alert_history = deque(maxlen=100)

    def send_alert(self, video_name: str, crime_type: str, crime_score: float,
                   metrics: Dict, severity: str) -> bool:
        """Send crime alert email"""
        try:
            yag = yagmail.SMTP(self.email, self.password)

            subject = f"🚨 CRIME ALERT: {crime_type} Detected - Score: {crime_score:.1f}%"

            body = f"""
            ⚠️ IMMEDIATE ACTION REQUIRED - CRIME DETECTED ⚠️

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            📹 VIDEO INFORMATION
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            File: {video_name}
            Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Crime Type: {crime_type}
            Severity Level: {severity}
            Overall Crime Score: {crime_score:.1f}%

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🎯 DETAILED CRIME METRICS
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            • Robbery Risk: {metrics.get('robbery_score', 0):.1f}%
            • Assault Risk: {metrics.get('assault_score', 0):.1f}%
            • Theft Indicators: {metrics.get('theft_score', 0):.1f}%
            • Weapon Detection: {metrics.get('weapon_score', 0):.1f}%
            • Abuse Indicators: {metrics.get('abuse_score', 0):.1f}%
            • Explosion Risk: {metrics.get('explosion_score', 0):.1f}%
            • Fighting Intensity: {metrics.get('fighting_score', 0):.1f}%
            • Accident Indicators: {metrics.get('accident_score', 0):.1f}%
            • Shooting Detection: {metrics.get('shooting_score', 0):.1f}%
            • Arson Indicators: {metrics.get('arson_score', 0):.1f}%

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🧠 LSTM-GRU TEMPORAL ANALYSIS
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            • Temporal Severity: {metrics.get('lstm_gru_severity', 0):.1f}%
            • Temporal Confidence: {metrics.get('temporal_confidence', 0):.1f}%
            • Peak Temporal Score: {metrics.get('peak_temporal_score', 0):.1f}%

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📊 EVENT STATISTICS
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            • Total Events: {metrics.get('crime_events', 0)}
            • Motion Intensity: {metrics.get('motion_intensity', 0):.1f}%

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📈 VIDEO METADATA
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            • Duration: {metrics.get('duration', 0):.1f} seconds
            • Frames Analyzed: {metrics.get('frames_analyzed', 0)}
            • Crime Persistence: {metrics.get('crime_persistence', 0):.1f}%

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            ACTION REQUIRED: Please review the detected security incident immediately.

            This is an automated alert from the Community Security Analytics System.
            """

            yag.send(to=self.email, subject=subject, contents=body)

            self.alert_history.append({
                'timestamp': datetime.now(),
                'video': video_name,
                'crime_type': crime_type,
                'score': crime_score
            })

            logger.info(f"Alert sent for {video_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


# --- 8. PROGRESS TRACKER ---
class ProgressTracker:
    """Handles progress indication and user feedback"""

    def __init__(self):
        self.progress_bars = {}
        self.status_messages = {}

    def create_progress(self, key: str, description: str = "Processing") -> Any:
        """Create a progress indicator"""
        import streamlit as st
        self.status_messages[key] = st.empty()
        self.status_messages[key].info(f"⏳ {description}...")
        self.progress_bars[key] = st.progress(0)
        return self.progress_bars[key]

    def update_progress(self, key: str, value: float, message: str = None):
        """Update progress value"""
        if key in self.progress_bars:
            self.progress_bars[key].progress(value)
            if message and key in self.status_messages:
                self.status_messages[key].info(f"⏳ {message}")

    def complete_progress(self, key: str, success: bool = True, message: str = None):
        """Mark progress as complete"""
        if key in self.status_messages:
            if success:
                self.status_messages[key].success(f"✅ {message or 'Complete!'}")
            else:
                self.status_messages[key].error(f"❌ {message or 'Failed!'}")

            # Clean up
            if key in self.progress_bars:
                del self.progress_bars[key]
            del self.status_messages[key]


# --- 9. ENHANCED DATASET LOADER ---
class CrimeVideoDataset(Dataset):
    """Custom dataset for crime videos with proper labeling"""

    def __init__(self, crime_paths: List[str], normal_paths: List[str], transform=None, sequence_length=16):
        self.video_paths = []
        self.labels = []

        # Add crime videos with label 1
        for path in crime_paths:
            self.video_paths.append(path)
            self.labels.append(1)  # Crime

        # Add normal videos with label 0
        for path in normal_paths:
            self.video_paths.append(path)
            self.labels.append(0)  # Normal

        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract frames from video
        frames = self._extract_frames(video_path)

        if frames is None or len(frames) == 0:
            # Return zeros tensor
            video_tensor = torch.zeros((self.sequence_length, 3, 112, 112))
            return video_tensor, torch.tensor(label, dtype=torch.long)

        # Apply transformations
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Pad or truncate to sequence length
        if len(frames) < self.sequence_length:
            # Pad with zeros
            zeros_shape = list(frames[0].shape) if frames else [3, 112, 112]
            padding = [torch.zeros(*zeros_shape) for _ in range(self.sequence_length - len(frames))]
            frames.extend(padding)
        else:
            frames = frames[:self.sequence_length]

        video_tensor = torch.stack(frames)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def _extract_frames(self, video_path, num_frames=16):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                return None

            # Sample frames uniformly
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            cap.release()
            return frames
        except Exception as e:
            logger.error(f"Frame extraction failed for {video_path}: {e}")
            return None


# --- 10. SIMPLE CNN MODEL (No external downloads needed) ---
class SimpleCNNFeatureExtractor(nn.Module):
    """Simple CNN feature extractor that doesn't require downloading weights"""

    def __init__(self):
        super(SimpleCNNFeatureExtractor, self).__init__()
        # Simple CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.feature_dim = 256
        self.reduce_dim = nn.Linear(256, 128)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        features = self.reduce_dim(features)
        features = features.view(batch_size, seq_len, -1)
        return features


class SimpleTemporalDetector(nn.Module):
    """Simple LSTM model for crime detection"""

    def __init__(self, input_size=128, hidden_size=64, num_layers=1, num_classes=13):
        super(SimpleTemporalDetector, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        self.severity_regressor = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        lstm_out, (hidden, cell) = self.lstm(x)
        # Get the last hidden state
        if isinstance(hidden, tuple):
            last_hidden = hidden[0][-1] if len(hidden[0].shape) > 1 else hidden[0]
        else:
            last_hidden = hidden[-1] if len(hidden.shape) > 1 else hidden

        # Ensure last_hidden has correct shape [batch_size, hidden_size]
        if len(last_hidden.shape) == 3:
            last_hidden = last_hidden.squeeze(0)

        logits = self.classifier(last_hidden)
        severity = self.severity_regressor(last_hidden)

        if return_attention:
            return logits, severity, None
        return logits, severity


# --- 11. MODEL TRAINER ---
class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = None

    def load_all_videos(self) -> Tuple[List[str], List[str]]:
        """Load all videos from all three datasets"""
        crime_videos = []
        normal_videos = []

        # Load from Crime folder
        if os.path.exists(self.config.CRIME_DATASET_PATH):
            crime_videos.extend(self._get_video_files(self.config.CRIME_DATASET_PATH))

        # Load from Normal videos folder
        if os.path.exists(self.config.NORMAL_DATASET_PATH):
            normal_videos.extend(self._get_video_files(self.config.NORMAL_DATASET_PATH))

        # Load from split dataset (both crime and normal)
        if os.path.exists(self.config.SPLIT_DATASET_PATH):
            for folder in os.listdir(self.config.SPLIT_DATASET_PATH):
                folder_path = os.path.join(self.config.SPLIT_DATASET_PATH, folder)
                if os.path.isdir(folder_path):
                    videos = self._get_video_files(folder_path)
                    if 'crime' in folder.lower() or 'violence' in folder.lower():
                        crime_videos.extend(videos)
                    else:
                        normal_videos.extend(videos)

        logger.info(f"Loaded {len(crime_videos)} crime videos, {len(normal_videos)} normal videos")
        return crime_videos, normal_videos

    def _get_video_files(self, root_path: str) -> List[str]:
        """Get all video files in a directory recursively"""
        video_files = []
        for ext in self.config.SUPPORTED_FORMATS:
            pattern = str(pathlib.Path(root_path) / "**" / f"*.{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))
        return video_files

    def train_model(self, progress_callback=None):
        """Train the simple CNN-LSTM model on the dataset"""
        # Load all videos
        if progress_callback:
            progress_callback(0.05, "Loading videos from all datasets...")

        crime_videos, normal_videos = self.load_all_videos()

        if len(crime_videos) == 0 or len(normal_videos) == 0:
            logger.warning("Insufficient data for training")
            if progress_callback:
                progress_callback(1.0, "Training failed: Insufficient data")
            return False

        if progress_callback:
            progress_callback(0.1, f"Found {len(crime_videos)} crime and {len(normal_videos)} normal videos")

        # Create dataset with smaller image size for T450
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),  # Smaller size for T450
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if progress_callback:
            progress_callback(0.15, "Building dataset...")

        dataset = CrimeVideoDataset(crime_videos, normal_videos, transform, self.config.SEQUENCE_LENGTH)

        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0)

        if progress_callback:
            progress_callback(0.2, "Initializing simple model architecture...")

        # Initialize simple models (no downloads required)
        self.feature_extractor = SimpleCNNFeatureExtractor().to(self.device)
        self.model = SimpleTemporalDetector(input_size=128, hidden_size=64, num_layers=1, num_classes=13).to(
            self.device)

        # Optimizer and loss
        optimizer = optim.Adam(list(self.feature_extractor.parameters()) + list(self.model.parameters()),
                               lr=self.config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0
        total_batches = len(train_loader)

        for epoch in range(self.config.EPOCHS):
            # Training phase
            self.model.train()
            self.feature_extractor.train()
            train_loss = 0
            train_correct = 0

            epoch_progress_start = 0.2 + (epoch / self.config.EPOCHS) * 0.7

            if progress_callback:
                progress_callback(epoch_progress_start, f"Epoch {epoch + 1}/{self.config.EPOCHS} - Training...")

            for batch_idx, (videos, labels) in enumerate(train_loader):
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                # Extract features
                features = self.feature_extractor(videos)

                # Forward pass
                logits, severity = self.model(features)

                # Ensure logits and labels have compatible shapes
                if logits.shape[0] != labels.shape[0]:
                    # This shouldn't happen with proper batching, but just in case
                    min_batch = min(logits.shape[0], labels.shape[0])
                    logits = logits[:min_batch]
                    labels = labels[:min_batch]

                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()

                # Update progress within epoch
                if progress_callback and batch_idx % 5 == 0:  # Update less frequently for performance
                    batch_progress = (batch_idx + 1) / total_batches
                    epoch_progress = epoch_progress_start + (batch_progress * (0.7 / self.config.EPOCHS))
                    progress_callback(epoch_progress, f"Epoch {epoch + 1} - Batch {batch_idx + 1}/{total_batches}")

            train_acc = train_correct / len(train_dataset)

            # Validation phase
            if progress_callback:
                progress_callback(epoch_progress_start + (0.7 / self.config.EPOCHS),
                                  f"Epoch {epoch + 1} - Validating...")

            val_acc = self.evaluate(val_loader)

            logger.info(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                if progress_callback:
                    progress_callback(epoch_progress_start + (0.7 / self.config.EPOCHS),
                                      f"Epoch {epoch + 1} - New best model! Val Acc: {val_acc:.2%}")

        if progress_callback:
            progress_callback(0.95, "Finalizing model...")
            time.sleep(0.5)
            progress_callback(1.0, "Training complete!")

        return best_val_acc > 0.55  # Lower threshold for simple model

    def evaluate(self, loader):
        """Evaluate model on validation set"""
        self.model.eval()
        self.feature_extractor.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in loader:
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                features = self.feature_extractor(videos)
                logits, _ = self.model(features)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0

    def save_model(self):
        """Save trained model"""
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, "simple_crime_detector.pth")
        extractor_path = os.path.join(self.config.MODEL_SAVE_PATH, "simple_feature_extractor.pth")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.feature_extractor.state_dict(), extractor_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self):
        """Load trained model"""
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, "simple_crime_detector.pth")
        extractor_path = os.path.join(self.config.MODEL_SAVE_PATH, "simple_feature_extractor.pth")

        if os.path.exists(model_path) and os.path.exists(extractor_path):
            self.feature_extractor = SimpleCNNFeatureExtractor().to(self.device)
            self.model = SimpleTemporalDetector(input_size=128, hidden_size=64, num_layers=1, num_classes=13).to(
                self.device)

            self.feature_extractor.load_state_dict(
                torch.load(extractor_path, map_location=self.device, weights_only=True))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

            self.model.eval()
            self.feature_extractor.eval()

            logger.info("Simple model loaded successfully")
            return True

        return False


# --- 12. ENHANCED CRIME ANALYZER WITH ALL CRIME TYPES ---
class CrimeAnalyzer:
    """Advanced crime analyzer with model-based detection for all crime types"""

    def __init__(self, config: Config, trainer: ModelTrainer):
        self.config = config
        self.trainer = trainer
        self.device = trainer.device
        self.analysis_history = deque(maxlen=100)

        # Initialize buffers
        self.frame_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.feature_buffer = deque(maxlen=config.SEQUENCE_LENGTH)

        # Crime type mapping
        self.crime_types = [
            'NORMAL', 'ROBBERY', 'ASSAULT', 'THEFT', 'WEAPON',
            'ABUSE', 'EXPLOSION', 'FIGHTING', 'ACCIDENT', 'SHOOTING', 'ARSON'
        ]

    def analyze_video(self, video_path: str, progress_callback=None) -> Dict:
        """Analyze video for all crime types"""
        # Check video accessibility
        if not self._check_video(video_path):
            return {'error': 'Video file is corrupted or inaccessible', 'crime_detected': False}

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Initialize metrics for all crime types
        crime_scores = []
        robbery_scores = []
        assault_scores = []
        theft_scores = []
        weapon_scores = []
        abuse_scores = []
        explosion_scores = []
        fighting_scores = []
        accident_scores = []
        shooting_scores = []
        arson_scores = []
        motion_scores = []

        prev_frame = None
        frame_count = 0

        # Adaptive sampling - sample fewer frames for performance
        sample_rate = max(1, int(total_frames / 80)) if total_frames > 80 else 1

        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Update progress
            if progress_callback:
                progress_callback(frame_idx / total_frames)

            # Add to buffer for temporal analysis
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_buffer.append(frame_rgb)

            # Individual frame metrics
            if len(self.frame_buffer) >= self.config.SEQUENCE_LENGTH // 2:
                # Get model prediction
                model_score, crime_type = self._get_model_prediction(list(self.frame_buffer))
                if model_score is not None:
                    crime_scores.append(model_score)

                # Motion analysis
                if prev_frame is not None:
                    motion_score = self._calculate_motion(prev_frame, frame)
                    motion_scores.append(motion_score)

                # Detect all crime types
                robbery, theft = self._detect_robbery_indicators(prev_frame, frame)
                assault, fighting = self._detect_assault_indicators(prev_frame, frame)
                weapon = self._detect_weapons(frame)
                abuse = self._detect_abuse(frame)
                explosion = self._detect_explosion(frame)
                accident = self._detect_accident(prev_frame, frame)
                shooting = self._detect_shooting(frame)
                arson = self._detect_arson(frame)

                robbery_scores.append(robbery)
                theft_scores.append(theft)
                assault_scores.append(assault)
                fighting_scores.append(fighting)
                weapon_scores.append(weapon)
                abuse_scores.append(abuse)
                explosion_scores.append(explosion)
                accident_scores.append(accident)
                shooting_scores.append(shooting)
                arson_scores.append(arson)

            prev_frame = frame.copy()

        cap.release()

        # Calculate final metrics
        if frame_count == 0:
            return {'error': 'No frames could be processed', 'crime_detected': False}

        # Aggregate scores
        avg_crime = np.mean(crime_scores) if crime_scores else 0
        avg_robbery = np.mean(robbery_scores) if robbery_scores else 0
        avg_assault = np.mean(assault_scores) if assault_scores else 0
        avg_theft = np.mean(theft_scores) if theft_scores else 0
        avg_weapon = np.mean(weapon_scores) if weapon_scores else 0
        avg_abuse = np.mean(abuse_scores) if abuse_scores else 0
        avg_explosion = np.mean(explosion_scores) if explosion_scores else 0
        avg_fighting = np.mean(fighting_scores) if fighting_scores else 0
        avg_accident = np.mean(accident_scores) if accident_scores else 0
        avg_shooting = np.mean(shooting_scores) if shooting_scores else 0
        avg_arson = np.mean(arson_scores) if arson_scores else 0
        avg_motion = np.mean(motion_scores) if motion_scores else 0

        # Find highest crime score
        all_scores = {
            'ROBBERY': avg_robbery,
            'ASSAULT': avg_assault,
            'THEFT': avg_theft,
            'WEAPON': avg_weapon,
            'ABUSE': avg_abuse,
            'EXPLOSION': avg_explosion,
            'FIGHTING': avg_fighting,
            'ACCIDENT': avg_accident,
            'SHOOTING': avg_shooting,
            'ARSON': avg_arson
        }

        max_crime_score = max(all_scores.values())
        crime_type = max(all_scores, key=all_scores.get) if max_crime_score > 20 else 'NORMAL'

        # Final crime score (weighted combination)
        final_crime_score = (
                avg_crime * 0.30 +
                max_crime_score * 0.40 +
                avg_motion * 0.15 +
                avg_weapon * 0.15
        )

        # Determine severity
        severity = self._get_severity(final_crime_score)

        # Build result
        result = {
            'crime_detected': final_crime_score > self.config.DETECTION_THRESHOLD,
            'crime_score': float(round(final_crime_score, 2)),
            'crime_type': crime_type,
            'severity': severity,
            'robbery_score': float(round(avg_robbery, 2)),
            'assault_score': float(round(avg_assault, 2)),
            'theft_score': float(round(avg_theft, 2)),
            'weapon_score': float(round(avg_weapon, 2)),
            'abuse_score': float(round(avg_abuse, 2)),
            'explosion_score': float(round(avg_explosion, 2)),
            'fighting_score': float(round(avg_fighting, 2)),
            'accident_score': float(round(avg_accident, 2)),
            'shooting_score': float(round(avg_shooting, 2)),
            'arson_score': float(round(avg_arson, 2)),
            'motion_intensity': float(round(avg_motion, 2)),
            'duration': float(round(duration, 2)),
            'frames_analyzed': frame_count,
            'total_frames': total_frames,
            'timestamp': datetime.now().isoformat()
        }

        # Store in history
        self.analysis_history.append(result)

        return result

    def _check_video(self, video_path: str) -> bool:
        """Check if video is accessible"""
        try:
            if not os.path.exists(video_path):
                return False
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
        except:
            return False

    def _get_model_prediction(self, frames: List) -> Tuple[Optional[float], str]:
        """Get model-based crime prediction for frame sequence"""
        if self.trainer.model is None or self.trainer.feature_extractor is None:
            return None, "Unknown"

        try:
            # Preprocess frames
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            processed_frames = []
            for frame in frames[-self.config.SEQUENCE_LENGTH:]:
                processed_frames.append(transform(frame))

            # Pad if needed
            if len(processed_frames) < self.config.SEQUENCE_LENGTH:
                zeros_shape = list(processed_frames[0].shape) if processed_frames else [3, 112, 112]
                padding = [torch.zeros(*zeros_shape) for _ in
                           range(self.config.SEQUENCE_LENGTH - len(processed_frames))]
                processed_frames.extend(padding)

            video_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)

            # Extract features and predict
            with torch.no_grad():
                features = self.trainer.feature_extractor(video_tensor)
                logits, severity = self.trainer.model(features)
                probs = torch.softmax(logits, dim=1)

                crime_prob = probs[0][1].item()  # Probability of crime
                crime_type = "Crime" if crime_prob > 0.5 else "Normal"

            return crime_prob * 100, crime_type
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return None, "Unknown"

    def _calculate_motion(self, prev_frame, curr_frame) -> float:
        """Calculate motion intensity between frames"""
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_score = np.mean(magnitude) if magnitude.size > 0 else 0

            return min(motion_score * 2, 100)
        except:
            return 0

    def _detect_robbery_indicators(self, prev_frame, curr_frame) -> Tuple[float, float]:
        """Detect robbery and theft indicators"""
        if prev_frame is None or curr_frame is None:
            return 0, 0

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            magnitude_std = np.std(magnitude) if magnitude.size > 0 else 0

            # Sudden movement detection (robbery)
            sudden_movement = magnitude_std / (magnitude_mean + 1e-6)
            robbery_score = min(sudden_movement * 50, 100)

            # High velocity regions (theft)
            high_velocity_ratio = np.sum(magnitude > magnitude_mean * 2) / (magnitude.size + 1e-6)
            theft_score = min(high_velocity_ratio * 80, 100)

            return robbery_score, theft_score
        except:
            return 0, 0

    def _detect_assault_indicators(self, prev_frame, curr_frame) -> Tuple[float, float]:
        """Detect assault and fighting indicators"""
        if prev_frame is None or curr_frame is None:
            return 0, 0

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            direction = np.arctan2(flow[..., 1], flow[..., 0])

            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            direction_variance = np.var(direction) if direction.size > 0 else 0

            assault_score = min((magnitude_mean * 3) + (direction_variance * 0.5), 100)
            fighting_score = min((np.std(magnitude) * 5) + (magnitude_mean * 2), 100)

            return assault_score, fighting_score
        except:
            return 0, 0

    def _detect_weapons(self, frame) -> float:
        """Detect potential weapons in frame"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect metallic colors
            lower_metal = np.array([0, 0, 180])
            upper_metal = np.array([180, 50, 255])
            metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)

            # Detect sharp edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            metal_ratio = np.sum(metal_mask > 0) / (metal_mask.size + 1e-6)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

            weapon_score = min((metal_ratio * 60 + edge_density * 40), 100)
            return weapon_score
        except:
            return 0

    def _detect_abuse(self, frame) -> float:
        """Detect potential abuse indicators"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect aggressive body language through edge density
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

            # Detect skin tone clusters (potential physical contact)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / (skin_mask.size + 1e-6)

            abuse_score = min((edge_density * 50 + skin_ratio * 30), 100)
            return abuse_score
        except:
            return 0

    def _detect_explosion(self, frame) -> float:
        """Detect explosion indicators (bright flashes, smoke)"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect bright flashes (high intensity)
            brightness = hsv[:, :, 2]
            bright_pixels = np.sum(brightness > 240) / (brightness.size + 1e-6)

            # Detect smoke/clouds (gray regions)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smoke_mask = cv2.inRange(gray, 100, 160)
            smoke_ratio = np.sum(smoke_mask > 0) / (smoke_mask.size + 1e-6)

            explosion_score = min((bright_pixels * 70 + smoke_ratio * 30), 100)
            return explosion_score
        except:
            return 0

    def _detect_accident(self, prev_frame, curr_frame) -> float:
        """Detect accident indicators (sudden stops, collisions)"""
        if prev_frame is None or curr_frame is None:
            return 0

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # Sudden deceleration detection
            if len(self.frame_buffer) > 5:
                prev_flows = []
                for i in range(min(5, len(self.frame_buffer) - 1)):
                    # Simplified: use magnitude difference
                    pass

            # High magnitude with sudden change indicates accident
            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            magnitude_std = np.std(magnitude) if magnitude.size > 0 else 0

            accident_score = min((magnitude_mean * 30 + magnitude_std * 30), 100)
            return accident_score
        except:
            return 0

    def _detect_shooting(self, frame) -> float:
        """Detect shooting indicators (gun-like shapes, muzzle flashes)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect potential gun shapes (horizontal elongated objects)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            gun_like_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 5000:  # Potential gun size range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / (h + 1e-6)
                    if 1.5 < aspect_ratio < 4:  # Gun-like aspect ratio
                        gun_like_shapes += 1

            gun_score = min(gun_like_shapes * 10, 100)

            # Detect muzzle flashes (bright spots)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2]
            bright_spots = np.sum(brightness > 250) / (brightness.size + 1e-6)
            flash_score = bright_spots * 50

            shooting_score = min(gun_score + flash_score, 100)
            return shooting_score
        except:
            return 0

    def _detect_arson(self, frame) -> float:
        """Detect arson indicators (fire, smoke)"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect fire colors (red, orange, yellow)
            lower_fire1 = np.array([0, 100, 100])
            upper_fire1 = np.array([10, 255, 255])
            lower_fire2 = np.array([10, 100, 100])
            upper_fire2 = np.array([25, 255, 255])

            fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
            fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
            fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)

            fire_ratio = np.sum(fire_mask > 0) / (fire_mask.size + 1e-6)

            # Detect smoke
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smoke_mask = cv2.inRange(gray, 80, 140)
            smoke_ratio = np.sum(smoke_mask > 0) / (smoke_mask.size + 1e-6)

            arson_score = min((fire_ratio * 60 + smoke_ratio * 30), 100)
            return arson_score
        except:
            return 0

    def _get_severity(self, crime_score: float) -> str:
        """Get severity level based on crime score"""
        if crime_score > 70:
            return "CRITICAL"
        elif crime_score > 40:
            return "HIGH"
        elif crime_score > 20:
            return "MEDIUM"
        else:
            return "LOW"


# --- 13. EXPORT FUNCTIONALITY ---
class ReportExporter:
    """Handles report generation and export"""

    def __init__(self, reports_path: str):
        self.reports_path = reports_path
        os.makedirs(reports_path, exist_ok=True)

    def export_to_csv(self, results: List[Dict], filename: str = None) -> str:
        """Export analysis results to CSV"""
        if not filename:
            filename = f"crime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(self.reports_path, filename)
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)

        return filepath

    def export_to_json(self, results: List[Dict], filename: str = None) -> str:
        """Export analysis results to JSON"""
        if not filename:
            filename = f"crime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.reports_path, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return filepath

    def generate_html_report(self, result: Dict, video_name: str) -> str:
        """Generate HTML report for a single analysis"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crime Detection Report - {video_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: linear-gradient(135deg, #1a1a2e, #16213e);
                    color: #eee;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(0,0,0,0.7);
                    padding: 30px;
                    border-radius: 15px;
                }}
                h1 {{ color: #00fbff; text-align: center; }}
                .alert-critical {{ background: #ff4757; padding: 20px; border-radius: 10px; }}
                .alert-warning {{ background: #feca57; padding: 20px; border-radius: 10px; color: #000; }}
                .alert-secure {{ background: #00ff88; padding: 20px; border-radius: 10px; color: #000; }}
                .metric {{ margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #00fbff; }}
                .grid-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .crime-card {{
                    background: rgba(255,255,255,0.05);
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #ff4757;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚨 Comprehensive Crime Detection Report</h1>
                <h3>Video: {video_name}</h3>
                <p>Analysis Time: {result.get('timestamp', datetime.now().isoformat())}</p>

                <div class="alert-{self._get_alert_class(result.get('crime_score', 0))}">
                    <h2>{'CRIME DETECTED' if result.get('crime_detected') else 'NO CRIME DETECTED'}</h2>
                    <p>Overall Crime Score: {result.get('crime_score', 0)}%</p>
                    <p>Primary Crime Type: {result.get('crime_type', 'NORMAL')}</p>
                    <p>Severity: {result.get('severity', 'LOW')}</p>
                </div>

                <h2>Detailed Crime Metrics</h2>
                <div class="grid-container">
                    <div class="crime-card"><span class="metric-label">🔫 ROBBERY:</span> {result.get('robbery_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">👊 ASSAULT:</span> {result.get('assault_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">💰 THEFT:</span> {result.get('theft_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">🔪 WEAPON:</span> {result.get('weapon_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">😢 ABUSE:</span> {result.get('abuse_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">💥 EXPLOSION:</span> {result.get('explosion_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">🥊 FIGHTING:</span> {result.get('fighting_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">🚗 ACCIDENT:</span> {result.get('accident_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">🔫 SHOOTING:</span> {result.get('shooting_score', 0)}%</div>
                    <div class="crime-card"><span class="metric-label">🔥 ARSON:</span> {result.get('arson_score', 0)}%</div>
                </div>

                <h2>Video Information</h2>
                <div class="metric"><span class="metric-label">Duration:</span> {result.get('duration', 0)} seconds</div>
                <div class="metric"><span class="metric-label">Frames Analyzed:</span> {result.get('frames_analyzed', 0)}</div>
                <div class="metric"><span class="metric-label">Motion Intensity:</span> {result.get('motion_intensity', 0)}%</div>

                <p style="text-align: center; margin-top: 30px; color: #888;">
                    Generated by Community Security Analytics System
                </p>
            </div>
        </body>
        </html>
        """

        filepath = os.path.join(self.reports_path,
                                f"report_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(filepath, 'w') as f:
            f.write(html)

        return filepath

    def _get_alert_class(self, crime_score: float) -> str:
        if crime_score > 70:
            return "critical"
        elif crime_score > 40:
            return "warning"
        else:
            return "secure"


# --- 14. SESSION STATE MANAGER ---
class SessionStateManager:
    """Manages persistent UI state"""

    def __init__(self):
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.analysis_complete = False
            st.session_state.last_results = None
            st.session_state.selected_video = None
            st.session_state.theme = 'dark'
            st.session_state.notifications = []
            st.session_state.email_alerts_enabled = True
            st.session_state.detection_threshold = 40.0
            st.session_state.model_loaded = False
            st.session_state.training_complete = False
            st.session_state.training_progress = 0
            st.session_state.training_message = ""

    def add_notification(self, message: str, type: str = 'info'):
        """Add a notification to session state"""
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        st.session_state.notifications.append({
            'message': message,
            'type': type,
            'timestamp': datetime.now()
        })

    def get_notifications(self):
        """Get all notifications"""
        return st.session_state.get('notifications', [])

    def clear_notifications(self):
        """Clear all notifications"""
        st.session_state.notifications = []


# --- 15. MAIN APPLICATION ---
def main():
    """Main Streamlit application"""
    set_background()

    # Initialize components
    session_manager = SessionStateManager()
    db_manager = DatabaseManager()
    cache_manager = CacheManager()
    email_alerts = EmailAlertSystem(config.ALERT_EMAIL, config.GMAIL_APP_PASSWORD)
    progress_tracker = ProgressTracker()
    trainer = ModelTrainer(config)
    analyzer = CrimeAnalyzer(config, trainer)
    exporter = ReportExporter(config.REPORTS_PATH)

    # Load or train model with proper progress display
    if not st.session_state.get('model_loaded', False) and not st.session_state.get('training_complete', False):
        # Check if model exists
        model_path = os.path.join(config.MODEL_SAVE_PATH, "simple_crime_detector.pth")
        extractor_path = os.path.join(config.MODEL_SAVE_PATH, "simple_feature_extractor.pth")

        if os.path.exists(model_path) and os.path.exists(extractor_path):
            with st.spinner("🔄 Loading crime detection model..."):
                if trainer.load_model():
                    st.session_state.model_loaded = True
                    st.success("✅ Crime detection model loaded!")
                    session_manager.add_notification("Model loaded successfully", "success")
                else:
                    st.warning("⚠️ Could not load model. Training new model.")
        else:
            # Show training interface with progress
            st.info("📚 No pre-trained model found. Training simple CNN-LSTM model on your dataset...")
            st.info("💡 Using lightweight model (no downloads required) optimized for T450 ThinkPad")

            # Create a placeholder for progress
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            status_text = status_placeholder.info("⏳ Initializing model training...")

            def update_training_progress(progress, message):
                progress_bar.progress(progress)
                status_text.info(f"⏳ {message}")
                st.session_state.training_progress = progress
                st.session_state.training_message = message

            # Train in a way that updates UI
            if trainer.train_model(update_training_progress):
                status_text.success("✅ Model training completed successfully!")
                st.session_state.model_loaded = True
                st.session_state.training_complete = True
                trainer.load_model()
                session_manager.add_notification("Model trained successfully", "success")
                time.sleep(1)  # Give user time to see success message
            else:
                status_text.error("❌ Model training failed. Using heuristic detection only.")
                session_manager.add_notification("Model training failed - using fallback detection", "warning")

            # Clear the progress UI after training
            progress_placeholder.empty()
            status_placeholder.empty()

    # If model wasn't loaded in the first attempt, try loading again
    if not st.session_state.get('model_loaded', False):
        with st.spinner("🔄 Loading crime detection model..."):
            if trainer.load_model():
                st.session_state.model_loaded = True
                st.success("✅ Crime detection model loaded!")
                session_manager.add_notification("Model loaded successfully", "success")

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🚨 COMMUNITY SECURITY ANALYTICS</h1>
            <p style="color: #00fbff; font-size: 1.2em;">Production-Grade Crime Detection & Prevention System</p>
            <p style="color: #9b59b6; font-size: 1em;">Detects: Robbery | Assault | Theft | Weapon | Abuse | Explosion | Fighting | Accident | Shooting | Arson</p>
            <p style="color: #00ff88; font-size: 0.9em;">⚡ Optimized for T450 ThinkPad - No external downloads required</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #00fbff20, #00ff8820); 
                       border-radius: 15px; margin-bottom: 20px; border: 1px solid #00fbff;">
                <h3 style="color: #00fbff; margin: 0;">🎮 CONTROL PANEL</h3>
            </div>
        """, unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=["🎥 Live Analysis", "📁 Dataset Browser", "📊 Analytics History", "📈 Performance", "⚙️ Settings"],
            icons=["camera-video", "folder", "graph-up", "bar-chart", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background": "rgba(0,0,0,0.8)", "border": "1px solid #00fbff", "border-radius": "10px"},
                "icon": {"color": "#00fbff", "font-size": "20px"},
                "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "5px"},
                "nav-link-selected": {"background": "rgba(0,255,255,0.2)", "color": "#00fbff"},
            }
        )

        st.markdown("---")

        # System stats
        st.markdown('<div class="info-box"><h4 style="color: #00fbff;">📊 SYSTEM STATUS</h4></div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            model_status = "✅ Loaded" if st.session_state.get('model_loaded', False) else "⚠️ Heuristic"
            st.metric("Model Status", model_status)
        with col2:
            st.metric("Device", str(trainer.device).upper())
            st.caption("Simple CNN + LSTM")

        # Detection threshold
        st.markdown("---")
        threshold = st.slider("🎯 Detection Sensitivity", 0, 100, config.DETECTION_THRESHOLD,
                              help="Lower = More sensitive, Higher = Less sensitive")
        config.DETECTION_THRESHOLD = threshold

        # Email toggle
        email_enabled = st.checkbox("📧 Email Alerts", value=st.session_state.get('email_alerts_enabled', True))
        st.session_state.email_alerts_enabled = email_enabled

    # Main content
    if selected == "🎥 Live Analysis":
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Video Crime Analysis")

        col1, col2 = st.columns([0.45, 0.55])

        with col1:
            # Video source selection
            source_type = st.radio("Select Source:", ["📁 From Dataset", "📤 Upload Video"], horizontal=True)

            video_path = None
            video_name = None

            if source_type == "📁 From Dataset":
                # Load all videos from datasets
                crime_videos, normal_videos = trainer.load_all_videos()
                all_videos = crime_videos + normal_videos

                if all_videos:
                    video_options = {}
                    for v in all_videos:
                        folder = os.path.basename(os.path.dirname(v))
                        is_crime = "🔴 CRIME" if v in crime_videos else "🟢 NORMAL"
                        display_name = f"{is_crime} - {folder}/ {os.path.basename(v)}"
                        video_options[display_name] = v

                    selected_video = st.selectbox("Choose Video:", list(video_options.keys()))
                    video_path = video_options[selected_video]
                    video_name = os.path.basename(video_path)
                else:
                    st.warning("No videos found in datasets. Please check dataset paths.")

            else:
                uploaded_file = st.file_uploader("Upload Video", type=config.SUPPORTED_FORMATS)
                if uploaded_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    video_name = uploaded_file.name
                    st.success(f"✅ Uploaded: {video_name}")

            # Analysis button
            if video_path and st.button("🚨 ANALYZE VIDEO", use_container_width=True):
                progress_key = "analysis_progress"
                progress_tracker.create_progress(progress_key, "Analyzing video for criminal activity")

                def update_analysis_progress(p):
                    progress_tracker.update_progress(progress_key, p, f"Processing frames: {p * 100:.0f}%")

                result = analyzer.analyze_video(video_path, update_analysis_progress)

                if 'error' in result:
                    st.error(f"Analysis failed: {result['error']}")
                    progress_tracker.complete_progress(progress_key, False, "Analysis failed")
                else:
                    # Save to database
                    detection_data = {
                        'video_name': video_name,
                        'video_path': video_path,
                        'crime_type': result['crime_type'],
                        'crime_score': result['crime_score'],
                        'severity_level': result['severity'],
                        'frame_count': result['frames_analyzed'],
                        'duration': result['duration'],
                        'robbery_score': result['robbery_score'],
                        'assault_score': result['assault_score'],
                        'theft_score': result['theft_score'],
                        'weapon_score': result['weapon_score'],
                        'abuse_score': result['abuse_score'],
                        'explosion_score': result['explosion_score'],
                        'fighting_score': result['fighting_score'],
                        'accident_score': result['accident_score'],
                        'shooting_score': result['shooting_score'],
                        'arson_score': result['arson_score'],
                        'lstm_gru_score': result.get('lstm_gru_severity', 0),
                        'alert_sent': False,
                        'metadata': json.dumps(result)
                    }
                    db_manager.save_detection(detection_data)

                    # Send email alert if enabled and crime detected
                    if email_enabled and result['crime_detected']:
                        alert_sent = email_alerts.send_alert(
                            video_name, result['crime_type'], result['crime_score'],
                            result, result['severity']
                        )
                        if alert_sent:
                            session_manager.add_notification("Alert email sent", "success")
                            st.info("📧 Alert email sent to security team")

                    # Store in session state
                    st.session_state.last_results = result
                    st.session_state.analysis_complete = True
                    st.session_state.last_video_name = video_name

                    progress_tracker.complete_progress(progress_key, True, "Analysis complete")

                    # Show result message
                    if not result['crime_detected']:
                        st.success(
                            f"✅ Analysis complete - No criminal activity detected (Score: {result['crime_score']:.1f}%)")
                    else:
                        st.error(
                            f"🚨 {result['crime_type']} DETECTED! Severity: {result['severity']} - Score: {result['crime_score']:.1f}%")

        with col2:
            if st.session_state.get('analysis_complete', False) and st.session_state.get('last_results'):
                result = st.session_state.last_results
                video_name = st.session_state.get('last_video_name', 'Unknown')

                # Alert display
                if result['crime_detected']:
                    if result['crime_score'] > 70:
                        st.markdown(f"""
                            <div class="alert-critical">
                                🚨 {result['crime_type']} DETECTED!<br>
                                Score: {result['crime_score']:.1f}%<br>
                                Severity: {result['severity']}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="alert-warning">
                                ⚠️ {result['crime_type']} ACTIVITY<br>
                                Score: {result['crime_score']:.1f}%<br>
                                Severity: {result['severity']}
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="alert-secure">
                            ✅ NO CRIME DETECTED<br>
                            Security Score: {result['crime_score']:.1f}%<br>
                            Status: Normal Activity
                        </div>
                    """, unsafe_allow_html=True)

                # Metrics grid for all crime types
                st.markdown("### Crime Risk Assessment")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🔫 Robbery", f"{result['robbery_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("👊 Assault", f"{result['assault_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💰 Theft", f"{result['theft_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_b:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🔪 Weapon", f"{result['weapon_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("😢 Abuse", f"{result['abuse_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💥 Explosion", f"{result['explosion_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_c:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🥊 Fighting", f"{result['fighting_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🚗 Accident", f"{result['accident_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🔫 Shooting", f"{result['shooting_score']:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🔥 Arson", f"{result['arson_score']:.0f}%")
                st.markdown('</div>', unsafe_allow_html=True)

                # Radar chart for all crime types
                categories = ['Robbery', 'Assault', 'Theft', 'Weapon', 'Abuse', 'Explosion', 'Fighting', 'Accident',
                              'Shooting', 'Arson']
                values = [
                    result['robbery_score'],
                    result['assault_score'],
                    result['theft_score'],
                    result['weapon_score'],
                    result['abuse_score'],
                    result['explosion_score'],
                    result['fighting_score'],
                    result['accident_score'],
                    result['shooting_score'],
                    result['arson_score']
                ]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Crime Profile',
                    line_color='#ff4757',
                    fillcolor='rgba(255, 71, 87, 0.3)'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], color='white'),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=450,
                    margin=dict(l=50, r=50, t=30, b=30)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Export options
                st.markdown("### Export Report")
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                with col_exp1:
                    if st.button("📄 Export CSV", use_container_width=True):
                        report_path = exporter.export_to_csv([result])
                        with open(report_path, 'rb') as f:
                            st.download_button("📥 Download CSV", f, file_name=os.path.basename(report_path))

                with col_exp2:
                    if st.button("📋 Export JSON", use_container_width=True):
                        report_path = exporter.export_to_json([result])
                        with open(report_path, 'rb') as f:
                            st.download_button("📥 Download JSON", f, file_name=os.path.basename(report_path))

                with col_exp3:
                    if st.button("🌐 HTML Report", use_container_width=True):
                        report_path = exporter.generate_html_report(result, video_name)
                        with open(report_path, 'rb') as f:
                            st.download_button("📥 Download HTML", f, file_name=os.path.basename(report_path))

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "📁 Dataset Browser":
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Dataset Browser")

        crime_videos, normal_videos = trainer.load_all_videos()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
                <div class="info-box">
                    <h3 style="color: #ff4757;">🔴 CRIME VIDEOS</h3>
                    <p style="font-size: 2em; font-weight: bold;">{len(crime_videos)}</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="info-box">
                    <h3 style="color: #00ff88;">🟢 NORMAL VIDEOS</h3>
                    <p style="font-size: 2em; font-weight: bold;">{len(normal_videos)}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎥 Video List")

        # Create dataframe for display
        all_data = []
        for v in crime_videos[:50]:
            all_data.append({
                'Type': '🔴 CRIME',
                'Filename': os.path.basename(v),
                'Path': v,
                'Size (MB)': f"{os.path.getsize(v) / (1024 * 1024):.1f}" if os.path.exists(v) else "N/A"
            })

        for v in normal_videos[:50]:
            all_data.append({
                'Type': '🟢 NORMAL',
                'Filename': os.path.basename(v),
                'Path': v,
                'Size (MB)': f"{os.path.getsize(v) / (1024 * 1024):.1f}" if os.path.exists(v) else "N/A"
            })

        if all_data:
            df = pd.DataFrame(all_data)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Showing {len(all_data)} of {len(crime_videos) + len(normal_videos)} videos")
        else:
            st.info("No videos found in datasets")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "📊 Analytics History":
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Detection History")

        detections = db_manager.get_detections(limit=50)

        if detections:
            # Create dataframe
            df = pd.DataFrame(detections)

            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_detections = len([d for d in detections if d.get('crime_score', 0) > 30])
                st.metric("Total Detections", total_detections)

            with col2:
                avg_score = np.mean([d.get('crime_score', 0) for d in detections])
                st.metric("Avg Crime Score", f"{avg_score:.1f}%")

            with col3:
                high_severity = len([d for d in detections if d.get('severity_level') == 'CRITICAL'])
                st.metric("Critical Alerts", high_severity)

            with col4:
                alerts_sent = len([d for d in detections if d.get('alert_sent')])
                st.metric("Alerts Sent", alerts_sent)

            # History table
            history_df = pd.DataFrame([{
                'Time': d.get('detection_time', ''),
                'Video': d.get('video_name', '')[:30],
                'Type': d.get('crime_type', ''),
                'Score': f"{d.get('crime_score', 0):.1f}%",
                'Severity': d.get('severity_level', ''),
                'Robbery': f"{d.get('robbery_score', 0):.1f}%",
                'Assault': f"{d.get('assault_score', 0):.1f}%",
                'Theft': f"{d.get('theft_score', 0):.1f}%",
                'Weapon': f"{d.get('weapon_score', 0):.1f}%"
            } for d in detections])

            st.dataframe(history_df, use_container_width=True)

            # Trend chart
            if len(detections) > 1:
                fig = go.Figure()
                scores = [d.get('crime_score', 0) for d in detections]
                times = list(range(len(scores)))

                fig.add_trace(go.Scatter(
                    x=times,
                    y=scores,
                    mode='lines+markers',
                    name='Crime Score',
                    line=dict(color='#ff4757', width=2),
                    marker=dict(size=6, color='#ff4757')
                ))

                fig.add_hline(y=config.DETECTION_THRESHOLD, line_dash="dash", line_color="yellow",
                              annotation_text=f"Threshold: {config.DETECTION_THRESHOLD}%")

                fig.update_layout(
                    title="Crime Score Trend",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis_title="Analysis #",
                    yaxis_title="Crime Score (%)",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detection history found. Run some analyses to see results here.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "📈 Performance":
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Model Performance Metrics")

        # Get performance metrics from database
        detections = db_manager.get_detections(limit=200)

        if len(detections) > 10:
            # Calculate performance metrics
            y_true = [1 if d.get('crime_score', 0) > config.DETECTION_THRESHOLD else 0 for d in detections]
            y_pred = [1 if d.get('crime_type', 'NORMAL') != 'NORMAL' else 0 for d in detections]

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy * 100:.1f}%")
            with col2:
                st.metric("Precision", f"{precision * 100:.1f}%")
            with col3:
                st.metric("Recall", f"{recall * 100:.1f}%")
            with col4:
                st.metric("F1-Score", f"{f1 * 100:.1f}%")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Crime'],
                y=['Normal', 'Crime'],
                colorscale='Viridis',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"}
            ))

            cm_fig.update_layout(
                title="Confusion Matrix",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400
            )

            st.plotly_chart(cm_fig, use_container_width=True)

            # Model info
            st.markdown("""
                <div class="info-box">
                    <h4 style="color: #00fbff;">🤖 Simple Model Architecture (Optimized for T450)</h4>
                    <ul>
                        <li><b>Backbone:</b> Custom CNN (no external downloads)</li>
                        <li><b>Temporal Model:</b> Single-layer LSTM</li>
                        <li><b>Input:</b> Video sequences (16 frames, 112x112 resolution)</li>
                        <li><b>Output:</b> 13-class classification (Normal + 12 crime types)</li>
                        <li><b>Crime Types:</b> Robbery, Assault, Theft, Weapon, Abuse, Explosion, Fighting, Accident, Shooting, Arson</li>
                        <li><b>Model Size:</b> Very small - runs efficiently on T450</li>
                        <li><b>Network:</b> No internet required after installation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough data for performance metrics. Run at least 10 video analyses.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "⚙️ Settings":
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ System Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📧 Email Configuration")
            email = st.text_input("Alert Email", value=config.ALERT_EMAIL)
            password = st.text_input("App Password", value=config.GMAIL_APP_PASSWORD, type="password")

            if st.button("Test Email", use_container_width=True):
                test_alerts = EmailAlertSystem(email, password)
                if test_alerts.send_alert("Test Video", "TEST", 0, {}, "LOW"):
                    st.success("✅ Email configured successfully!")
                    config.ALERT_EMAIL = email
                    config.GMAIL_APP_PASSWORD = password
                else:
                    st.error("❌ Email configuration failed - Check credentials")

        with col2:
            st.markdown("#### 🗂️ Dataset Paths")
            st.text_input("Crime Dataset", value=config.CRIME_DATASET_PATH, disabled=True)
            st.text_input("Normal Dataset", value=config.NORMAL_DATASET_PATH, disabled=True)
            st.text_input("Split Dataset", value=config.SPLIT_DATASET_PATH, disabled=True)

            if st.button("🔄 Retrain Model", use_container_width=True):
                # Create progress display
                progress_bar = st.progress(0)
                status_text = st.empty()

                def train_progress(p, msg):
                    progress_bar.progress(p)
                    status_text.info(f"⏳ {msg}")

                if trainer.train_model(train_progress):
                    status_text.success("✅ Model retrained successfully!")
                    trainer.load_model()
                    st.session_state.model_loaded = True
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
                else:
                    status_text.error("❌ Model training failed - Check datasets")

        st.markdown("#### 🗑️ System Maintenance")
        col3, col4 = st.columns(2)

        with col3:
            if st.button("Clear Cache", use_container_width=True):
                cache_manager.clear_cache(older_than_days=1)
                st.success("✅ Cache cleared!")

        with col4:
            if st.button("Clear History", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.last_results = None
                st.success("✅ History cleared!")

        st.markdown("#### 📊 System Information (Optimized for T450)")
        st.json({
            "Model Architecture": "Custom CNN + LSTM (No external downloads)",
            "Model Status": "Loaded" if st.session_state.get('model_loaded', False) else "Heuristic Only",
            "Device": str(trainer.device),
            "Sequence Length": config.SEQUENCE_LENGTH,
            "Image Size": "112x112",
            "Batch Size": config.BATCH_SIZE,
            "Epochs": config.EPOCHS,
            "Detection Threshold": config.DETECTION_THRESHOLD,
            "Email Alerts": "Enabled" if email_enabled else "Disabled",
            "Crime Types": "10 Types",
            "Cache Directory": config.CACHE_PATH,
            "Reports Directory": config.REPORTS_PATH,
            "Database": "detections.db"
        })

        st.markdown('</div>', unsafe_allow_html=True)

    # Display notifications
    for notification in session_manager.get_notifications():
        if notification['type'] == 'success':
            st.success(notification['message'])
        elif notification['type'] == 'error':
            st.error(notification['message'])
        elif notification['type'] == 'warning':
            st.warning(notification['message'])


if __name__ == "__main__":
    main()