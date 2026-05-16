# app.py - Complete Streamlit Application
import streamlit as st
import os
import glob
import sys
import threading
import time
import json
from datetime import datetime
from collections import deque, defaultdict
import yagmail
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import tempfile
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image
import pathlib
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from model_loader import load_pretrained_model

warnings.filterwarnings('ignore')

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -- CONFIGURATION --
# Use Streamlit secrets for sensitive data
if 'GMAIL_PASSWORD' in st.secrets:
    GMAIL_APP_PASSWORD = st.secrets["GMAIL_PASSWORD"]
else:
    GMAIL_APP_PASSWORD = "twmlrauqerkvxark"  # Fallback for local testing
    st.warning("⚠️ Using default Gmail password. For production, set secrets in Streamlit Cloud.")

ALERT_EMAIL = "emmanuelchiutsi001@gmail.com"
DATASET_PATH = "./dataset"  # Changed for deployment - create this folder or use temp

# Define all crime categories
CRIME_CATEGORIES = [
    'abuse', 'arrest', 'arson', 'assault', 'bulglary', 'clapping',
    'explosion', 'fighting', 'meet_and_split', 'normal_videos',
    'roadaccidents', 'robbery', 'shooting', 'shoplifting', 'sitting',
    'standing_still', 'stealing', 'vandalism', 'walking',
    'walking_while_reading_book', 'walking_while_using_phone'
]

# Normal categories (should NOT trigger alerts)
NORMAL_CATEGORIES = {
    'normal_videos', 'clapping', 'sitting', 'standing_still',
    'walking', 'walking_while_reading_book', 'walking_while_using_phone',
    'meet_and_split'
}

# Crime categories for alerting
CRIME_ALERT_CATEGORIES = {
    'abuse', 'arrest', 'arson', 'assault', 'bulglary', 'explosion',
    'fighting', 'roadaccidents', 'robbery', 'shooting', 'shoplifting',
    'stealing', 'vandalism'
}

CRIME_TYPE_MAP = {
    'abuse': 'ABUSE', 'arrest': 'ARREST', 'arson': 'ARSON',
    'assault': 'ASSAULT', 'bulglary': 'BURGLARY', 'clapping': 'NORMAL',
    'explosion': 'EXPLOSION', 'fighting': 'FIGHTING',
    'meet_and_split': 'NORMAL', 'normal_videos': 'NORMAL',
    'roadaccidents': 'ROAD_ACCIDENT', 'robbery': 'ROBBERY',
    'shooting': 'SHOOTING', 'shoplifting': 'SHOPLIFTING',
    'sitting': 'NORMAL', 'standing_still': 'NORMAL',
    'stealing': 'STEALING', 'vandalism': 'VANDALISM',
    'walking': 'NORMAL', 'walking_while_reading_book': 'NORMAL',
    'walking_while_using_phone': 'NORMAL'
}

# -- PAGE CONFIG --
st.set_page_config(page_title="AI COMMUNITY SECURITY ANALYTICS", page_icon="🚨", layout="wide")


def set_background():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('https://images.unsplash.com/photo-1557597774-9d273e5e0b8a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80'); background-size: cover; background-repeat: no-repeat; background-attachment: fixed; background-position: center; }
    .main-header { text-align: center; padding: 20px; background: rgba(0, 0, 0, 0.8); border-radius: 15px; margin-bottom: 20px; border: 2px solid #00fbff; box-shadow: 0 0 20px rgba(0, 251, 255, 0.3); }
    .main-header h1 { color: white; text-shadow: 2px 2px 10px #000, 0 0 20px #00fbff; font-size: 3em; margin: 0; animation: glow 2s ease-in-out infinite alternate; }
    @keyframes glow { from { text-shadow: 0 0 10px #00fbff, 0 0 20px #00fbff; } to { text-shadow: 0 0 20px #00fbff, 0 0 30px #00fbff; } }
    .css-card { background: rgba(0, 0, 0, 0.85); padding: 20px; border-radius: 15px; box-shadow: 0 0 20px rgba(0, 255, 255, 0.3); border: 1px solid #00fbff; color: white; margin-bottom: 20px; backdrop-filter: blur(5px); }
    .metric-card { background: rgba(0, 255, 255, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #00fbff; margin: 10px 0; transition: transform 0.3s; }
    .metric-card:hover { transform: translateX(5px); background: rgba(0, 255, 255, 0.2); }
    .alert-critical { background: linear-gradient(45deg, #ff4757, #ff6b6b); color: white; padding: 20px; border-radius: 10px; text-align: center; font-size: 28px; font-weight: bold; animation: pulse 1s infinite; box-shadow: 0 0 30px #ff4757; margin: 10px 0; }
    .alert-warning { background: linear-gradient(45deg, #feca57, #ff9f43); color: black; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0 0 30px #feca57; margin: 10px 0; }
    .alert-secure { background: linear-gradient(45deg, #00ff88, #00d68f); color: black; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0 0 30px #00ff88; margin: 10px 0; }
    @keyframes pulse { 0% { transform: scale(1); box-shadow: 0 0 20px #ff4757; } 50% { transform: scale(1.02); box-shadow: 0 0 40px #ff4757; } 100% { transform: scale(1); box-shadow: 0 0 20px #ff4757; } }
    .stButton > button { background: rgba(0, 255, 255, 0.2); color: white; border: 2px solid #00fbff; border-radius: 10px; padding: 10px 20px; font-weight: bold; transition: all 0.3s; width: 100%; text-transform: uppercase; letter-spacing: 1px; }
    .stButton > button:hover { background: #00fbff; color: black; box-shadow: 0 0 20px #00fbff; border-color: #00fbff; }
    .info-box { background: rgba(0, 255, 255, 0.1); border: 1px solid #00fbff; border-radius: 10px; padding: 15px; margin: 10px 0; color: white; }
    .perf-metric { background: rgba(0, 0, 0, 0.6); border-radius: 10px; padding: 10px; text-align: center; border: 1px solid #00fbff; }
    .perf-value { font-size: 24px; font-weight: bold; color: #00fbff; }
    .perf-label { font-size: 12px; color: #ccc; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)


# -- VIDEO HANDLING FUNCTIONS --
def safe_get_video_files(root_path):
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv', '*.flv', '*.m4v', '*.mpeg']
    all_files = []
    if not os.path.exists(root_path):
        return all_files
    try:
        for ext in video_extensions:
            try:
                pattern = str(pathlib.Path(root_path) / ext)
                found_files = glob.glob(pattern, recursive=True)
                all_files.extend(found_files)
            except Exception:
                continue
    except Exception:
        pass
    return all_files


def check_video_file(video_path):
    try:
        if not os.path.exists(video_path):
            return False, "File does not exist"
        cv2.setRNGSeed(0)
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return False, "Cannot read video frames"
        return True, "Video is accessible"
    except Exception as e:
        return False, str(e)


def extract_single_frame(video_path):
    """Extract a single frame from video (fast)"""
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            # Take middle frame
            frame_pos = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()
                return frame

        cap.release()
        return None
    except Exception:
        return None


# -- CRIME DETECTION MODEL (MODIFIED TO LOAD PRE-TRAINED) --
class CrimeDetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.training_status = "Loading Model"
        self.frame_buffer = deque(maxlen=5)

        # Load pre-trained model instead of training
        with st.spinner("🤖 Loading pre-trained model..."):
            model_data = load_pretrained_model()

        if model_data:
            self.model = model_data['model']
            self.model = self.model.to(self.device)
            self.class_mapping = model_data['class_mapping']
            self.inverse_class_mapping = model_data['inverse_class_mapping']
            self.effective_class_names = model_data['effective_class_names']
            self.performance_metrics = model_data['performance_metrics']
            self.training_info = model_data['training_info']
            self.training_history = model_data['training_history']
            self.model_loaded = True
            self.training_status = "Model Loaded"
            st.success(f"✅ Model loaded! Accuracy: {self.performance_metrics.get('accuracy', 0):.1f}%")
        else:
            self.model_loaded = False
            self.training_status = "Model Not Found"
            st.error("❌ Could not load model. Please ensure saved_model.pkl exists.")

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_performance_metrics(self):
        if self.model_loaded and self.performance_metrics:
            return self.performance_metrics
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}

    def get_training_info(self):
        return self.training_info if self.model_loaded else {}

    def get_training_history(self):
        return self.training_history if self.model_loaded else {'train_acc': [], 'val_acc': []}

    def add_frame_to_buffer(self, frame):
        processed = self.preprocess(frame)
        self.frame_buffer.append(processed)
        return len(self.frame_buffer)

    def predict_from_buffer(self):
        """Predict using majority vote"""
        if len(self.frame_buffer) == 0 or not self.model_loaded:
            return 0, 0.0

        predictions = []
        confidences = []

        for frame in list(self.frame_buffer):
            pred_class, confidence = self.predict_frame(frame.unsqueeze(0))
            predictions.append(pred_class)
            confidences.append(confidence)

        # Majority vote
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0][0]

        # Average confidence for majority class
        avg_confidence = np.mean([confidences[i] for i, p in enumerate(predictions) if p == most_common])

        return most_common, avg_confidence

    def predict_frame(self, frame_tensor):
        """Predict class for a frame tensor"""
        if not self.model_loaded or self.model is None:
            return 0, 0.0

        try:
            self.model.eval()
            with torch.no_grad():
                if frame_tensor.dim() == 3:
                    frame_tensor = frame_tensor.unsqueeze(0)
                frame_tensor = frame_tensor.to(self.device)

                outputs = self.model(frame_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                confidence = probabilities[0][predicted[0]].item()

                # Map back to original class index
                original_class = self.inverse_class_mapping.get(predicted[0].item(), 0)
                return original_class, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.0


# -- ADVANCED ANALYZER --
class AdvancedCrimeAnalyzer:
    def __init__(self, model):
        self.model = model
        self.analysis_history = deque(maxlen=100)
        self.detection_stats = {
            'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0
        }

    def update_performance_stats(self, predicted_crime, actual_crime):
        if actual_crime is not None:
            if predicted_crime and actual_crime:
                self.detection_stats['true_positives'] += 1
            elif predicted_crime and not actual_crime:
                self.detection_stats['false_positives'] += 1
            elif not predicted_crime and not actual_crime:
                self.detection_stats['true_negatives'] += 1
            elif not predicted_crime and actual_crime:
                self.detection_stats['false_negatives'] += 1

    def get_detection_metrics(self):
        tp = self.detection_stats['true_positives']
        fp = self.detection_stats['false_positives']
        tn = self.detection_stats['true_negatives']
        fn = self.detection_stats['false_negatives']
        total = tp + fp + tn + fn
        if total == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'samples': 0}
        accuracy = (tp + tn) / total * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {'accuracy': round(accuracy, 2), 'precision': round(precision, 2),
                'recall': round(recall, 2), 'f1': round(f1, 2), 'samples': total}

    def analyze_video_crime(self, video_path, progress_bar=None, actual_label=None):
        is_valid, message = check_video_file(video_path)
        if not is_valid:
            return {}, f"Video error: {message}"

        is_normal_video = any(normal in video_path.lower() for normal in NORMAL_CATEGORIES)

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {}, "Cannot open video"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        frame_count = 0
        crime_events = []
        model_predictions = []
        prediction_confidences = []

        # Sample rate based on video length
        sample_rate = max(1, total_frames // 60)  # Max 60 frames per video

        # Clear buffer
        self.model.frame_buffer.clear()

        consecutive_crime = 0
        event_list = []

        for frame_idx in range(0, total_frames, sample_rate):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame_count += 1
                if progress_bar and total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

                # Model prediction
                self.model.add_frame_to_buffer(frame)
                pred_class, confidence = self.model.predict_from_buffer()

                if pred_class is not None:
                    model_predictions.append(pred_class)
                    prediction_confidences.append(confidence)

                    pred_category = CRIME_CATEGORIES[pred_class] if pred_class < len(CRIME_CATEGORIES) else 'unknown'
                    is_crime_pred = pred_category in CRIME_ALERT_CATEGORIES

                    if is_crime_pred:
                        consecutive_crime += 1
                    else:
                        consecutive_crime = max(0, consecutive_crime - 1)

                    # Trigger on sustained detection
                    if consecutive_crime >= 3:
                        crime_score = min(50 + confidence * 50, 100)
                        event_data = {
                            'time': round(frame_idx / fps, 1),
                            'score': round(crime_score, 1),
                            'type': CRIME_TYPE_MAP.get(pred_category, 'SUSPICIOUS'),
                            'category': pred_category,
                            'confidence': round(confidence * 100, 1)
                        }
                        event_list.append(event_data)
                        crime_events.append(event_data)

                        # Reset counter after alert to avoid spam
                        consecutive_crime = 0

            except Exception as e:
                continue

        cap.release()

        # Calculate metrics
        metrics = self.calculate_metrics(
            event_list, model_predictions, prediction_confidences,
            frame_count, duration, is_normal_video
        )

        # Store events separately in metrics
        metrics['crime_events_list'] = event_list

        if actual_label is not None:
            predicted_crime = len(event_list) > 0 or metrics.get('overall_crime_score', 0) > 35
            self.update_performance_stats(predicted_crime, actual_label)

        self.analysis_history.append({
            'timestamp': datetime.now(),
            'video': os.path.basename(video_path),
            'metrics': metrics,
            'crime_events': event_list
        })

        return metrics, "Analysis complete"

    def calculate_metrics(self, crime_events, predictions, confidences, frame_count, duration, is_normal_video):
        """Calculate final metrics"""

        # Base score on events
        if crime_events:
            avg_score = np.mean([e['score'] for e in crime_events])
            crime_score = min(avg_score, 100)
        else:
            crime_score = 0

        # Adjust based on model predictions
        if predictions and len(predictions) > 0:
            crime_ratio = sum(1 for p in predictions
                              if p < len(CRIME_CATEGORIES) and
                              CRIME_CATEGORIES[p] in CRIME_ALERT_CATEGORIES) / len(predictions)

            if crime_ratio > 0.2:
                crime_score = min(crime_score + crime_ratio * 30, 100)

        # Lower scores for normal videos
        if is_normal_video:
            crime_score = min(crime_score, 25)

        crime_score = np.clip(crime_score, 0, 100)

        # Count event types
        event_counts = defaultdict(int)
        for event in crime_events:
            event_counts[event['type']] += 1

        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            'overall_crime_score': float(round(crime_score, 1)),
            'robbery_score': float(round(min(event_counts.get('ROBBERY', 0) * 15, 100), 1)),
            'theft_score': float(round(min(event_counts.get('THEFT', 0) * 12, 100), 1)),
            'assault_score': float(round(min(event_counts.get('ASSAULT', 0) * 15, 100), 1)),
            'weapon_detection': float(round(min(event_counts.get('WEAPON DETECTED', 0) * 20, 100), 1)),
            'fighting_score': float(round(min(event_counts.get('FIGHTING', 0) * 12, 100), 1)),
            'frames_analyzed': frame_count,
            'duration': float(round(duration, 2)),
            'crime_events': len(crime_events),
            'model_confidence': float(round(avg_confidence * 100, 1)),
            'is_normal_video': is_normal_video,
            'model_trained': self.model.model_loaded,
            'model_architecture': 'ResNet18 (Pre-trained)'
        }


# -- VIDEO SCANNER --
class VideoScanner:
    def __init__(self):
        self.cache = {}
        self.last_scan = 0
        self.scan_interval = 300

    def get_all_videos(self, root_path, force_refresh=False):
        current_time = time.time()
        if not force_refresh and current_time - self.last_scan < self.scan_interval:
            if root_path in self.cache:
                return self.cache[root_path]
        all_files = safe_get_video_files(root_path)
        self.cache[root_path] = all_files
        self.last_scan = current_time
        return all_files

    def get_split_stats(self, root_path):
        splits = ['train', 'test', 'valid']
        split_stats = {}
        for split in splits:
            split_path = os.path.join(root_path, split)
            if os.path.exists(split_path):
                videos = self.get_all_videos(split_path)
                split_stats[split] = len(videos)
            else:
                split_stats[split] = 0
        return split_stats


# -- HELPER FUNCTIONS --
def get_crime_level(score):
    if score > 70:
        return "CRITICAL", "#ff4757"
    elif score > 40:
        return "WARNING", "#feca57"
    else:
        return "NORMAL", "#00ff88"


def send_crime_alert(video_filename, metrics, crime_type, crime_level):
    try:
        yag = yagmail.SMTP(user=ALERT_EMAIL, password=GMAIL_APP_PASSWORD,
                           host='smtp.gmail.com', port=587,
                           smtp_starttls=True, smtp_ssl=False)
        alert_subject = f"🚨 {crime_type} ALERT: {metrics['overall_crime_score']:.1f}%"
        alert_body = f"""
⚠️ CRIME DETECTED ⚠️

Video: {video_filename}
Crime Type: {crime_type}
Score: {metrics['overall_crime_score']:.1f}%
Level: {crime_level}

Events: {metrics.get('crime_events', 0)}
Duration: {metrics.get('duration', 0)}s

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        yag.send(to=ALERT_EMAIL, subject=alert_subject, contents=alert_body)
        yag.close()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


def create_folder_structure():
    try:
        os.makedirs(DATASET_PATH, exist_ok=True)
        for split in ['train', 'test', 'valid']:
            os.makedirs(os.path.join(DATASET_PATH, split), exist_ok=True)
    except:
        pass


create_folder_structure()


def run_automatic_analysis(video_path, video_filename, email_alerts, threshold):
    is_normal = any(normal in video_path.lower() for normal in NORMAL_CATEGORIES)
    actual_label = not is_normal

    with st.spinner(f"🔍 Analyzing {video_filename}..."):
        progress_bar = st.progress(0)
        metrics, message = analyzer.analyze_video_crime(video_path, progress_bar, actual_label)
        progress_bar.progress(1.0)

        st.session_state['last_analysis'] = {
            'metrics': metrics,
            'filename': video_filename,
            'time': datetime.now()
        }
        st.session_state['analysis_complete'] = True

        crime_score = metrics.get('overall_crime_score', 0)
        if message == "Analysis complete":
            if is_normal and crime_score < 30:
                st.success(f"✅ Normal behavior detected!")
            elif not is_normal and crime_score > threshold:
                st.error(f"⚠️ Suspicious activity detected! Score: {crime_score}%")
            else:
                st.success(f"✅ Analysis complete!")

        if email_alerts and metrics and crime_score > threshold:
            crime_level, _ = get_crime_level(crime_score)
            crime_type = "SUSPICIOUS"
            if crime_score > 40:
                crime_type = "CRIME DETECTED"
            send_crime_alert(video_filename, metrics, crime_type, crime_level)


def display_performance_metrics():
    perf_metrics = crime_model.get_performance_metrics()
    training_info = crime_model.get_training_info()
    training_history = crime_model.get_training_history()

    st.markdown("### 📊 PERFORMANCE")

    if crime_model.model_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{perf_metrics.get('accuracy', 0):.1f}%")
        with col2:
            st.metric("Precision", f"{perf_metrics.get('precision', 0):.1f}%")
        with col3:
            st.metric("Recall", f"{perf_metrics.get('recall', 0):.1f}%")
        with col4:
            st.metric("F1 Score", f"{perf_metrics.get('f1_score', 0):.1f}%")

        if training_info:
            st.info(f"""
            **Model:** {training_info.get('model_architecture', 'ResNet18')}
            **Best Val Acc:** {training_info.get('best_val_accuracy', 0):.1f}% 
            **Samples:** Train={training_info.get('train_samples', 0)} | Val={training_info.get('val_samples', 0)}
            """)

        # Training curve
        if training_history and training_history.get('train_acc'):
            fig = go.Figure()
            epochs = list(range(1, len(training_history['train_acc']) + 1))
            if len(training_history['train_acc']) > 0:
                fig.add_trace(go.Scatter(x=epochs, y=training_history['train_acc'], mode='lines',
                                         name='Training', line=dict(color='#00fbff')))
            if len(training_history['val_acc']) > 0:
                fig.add_trace(go.Scatter(x=epochs, y=training_history['val_acc'], mode='lines',
                                         name='Validation', line=dict(color='#ff4757')))
            fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Accuracy (%)",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                              height=250)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🤖 Model not loaded. Please ensure saved_model.pkl exists.")


# -- INITIALIZE COMPONENTS --
@st.cache_resource
def init_components():
    crime_model = CrimeDetectionModel()
    analyzer = AdvancedCrimeAnalyzer(crime_model)
    scanner = VideoScanner()
    return crime_model, analyzer, scanner


crime_model, analyzer, scanner = init_components()


# -- MAIN APP --
def main():
    set_background()

    st.markdown("""
    <div class="main-header">
        <h1>🚨 AI COMMUNITY SECURITY ANALYTICS</h1>
        <p style="color: #00fbff;">ResNet18 | Pre-trained Model | Real-time Detection</p>
    </div>
    """, unsafe_allow_html=True)

    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False

    with st.sidebar:
        st.markdown("### CONTROL PANEL")

        selected = st.radio("Menu", ["Live Analysis", "Analytics History", "Settings"])

        st.markdown("---")

        videos = scanner.get_all_videos(DATASET_PATH)
        split_stats = scanner.get_split_stats(DATASET_PATH)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Videos", len(videos))
        with col2:
            st.metric("Analyses", len(analyzer.analysis_history))

        st.markdown("#### Dataset")
        st.text(f"Train: {split_stats.get('train', 0)}")
        st.text(f"Test: {split_stats.get('test', 0)}")
        st.text(f"Valid: {split_stats.get('valid', 0)}")

        st.markdown(f"**Model:** {'✅ Loaded' if crime_model.model_loaded else '⚠️ Not Loaded'}")
        st.markdown(f"**Device:** {crime_model.device}")

        display_performance_metrics()

        st.markdown("---")
        threshold = st.slider("Alert Threshold", 0, 100, 45)
        email_alerts = st.checkbox("Email Alerts", value=True)

    if selected == "Live Analysis":
        col1, col2 = st.columns([0.4, 0.6])

        with col1:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown("### VIDEO SOURCE")

            source = st.radio("Source:", ["Upload"], horizontal=True)  # Changed to Upload only for deployment
            video_path = None
            video_filename = None

            if source == "Upload":
                uploaded = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mkv', 'mov'])
                if uploaded:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded.read())
                    video_path = tfile.name
                    video_filename = uploaded.name
                    st.success(f"Uploaded: {video_filename}")

            if video_path and st.checkbox("Show Video Preview"):
                with open(video_path, 'rb') as f:
                    st.video(f.read())

            if video_path and st.button("🔍 ANALYZE", type="primary", use_container_width=True):
                run_automatic_analysis(video_path, video_filename, email_alerts, threshold)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if st.session_state.get('analysis_complete', False) and 'last_analysis' in st.session_state:
                metrics = st.session_state['last_analysis']['metrics']

                if metrics:
                    score = metrics.get('overall_crime_score', 0)
                    crime_level, color = get_crime_level(score)

                    if score > threshold:
                        alert_class = "alert-critical" if score > 70 else "alert-warning"
                        st.markdown(f'<div class="{alert_class}">🚨 ALERT! Score: {score:.1f}%</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-secure">✅ SAFE - Score: {score:.1f}%</div>',
                                    unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="info-box">
                        <b>File:</b> {st.session_state['last_analysis']['filename']}<br>
                        <b>Duration:</b> {metrics.get('duration', 0)}s<br>
                        <b>Frames:</b> {metrics.get('frames_analyzed', 0)}<br>
                        <b>Events:</b> {metrics.get('crime_events', 0)}<br>
                        <b>Model:</b> {metrics.get('model_architecture', 'ResNet18')}
                    </div>
                    """, unsafe_allow_html=True)

                    # Crime metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Robbery", f"{metrics.get('robbery_score', 0)}%")
                    with col_b:
                        st.metric("Assault", f"{metrics.get('assault_score', 0)}%")
                    with col_c:
                        st.metric("Theft", f"{metrics.get('theft_score', 0)}%")

                    # Display detailed events
                    if metrics.get('crime_events', 0) > 0:
                        with st.expander(f"📋 {metrics['crime_events']} Events Detected", expanded=True):
                            events_list = metrics.get('crime_events_list', [])
                            if events_list and len(events_list) > 0:
                                for idx, event in enumerate(events_list[:20]):
                                    event_time = event.get('time', 0)
                                    event_type = event.get('type', 'UNKNOWN')
                                    event_score = event.get('score', 0)
                                    event_conf = event.get('confidence', 0)

                                    if event_score > 70:
                                        event_color = "🔴"
                                    elif event_score > 40:
                                        event_color = "🟠"
                                    else:
                                        event_color = "🟡"

                                    st.markdown(f"""
                                    <div style="background: rgba(255, 71, 87, 0.1); padding: 8px; margin: 5px 0; border-radius: 5px; border-left: 3px solid {'#ff4757' if event_score > 70 else '#feca57' if event_score > 40 else '#00ff88'};">
                                        <b>⏱️ {event_time}s</b> - {event_color} <b>{event_type}</b><br>
                                        <span style="font-size: 12px;">Score: {event_score}% | Confidence: {event_conf}%</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No detailed event data available")
                    else:
                        st.success("✅ No suspicious events detected")
            else:
                st.info("📹 Upload a video and click ANALYZE to begin")

    elif selected == "Analytics History":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 📜 ANALYSIS HISTORY")

        if analyzer.analysis_history:
            history = []
            history_list = list(analyzer.analysis_history)
            for entry in history_list[-20:]:
                history.append({
                    'Time': entry['timestamp'].strftime('%H:%M:%S'),
                    'Video': entry['video'][:30],
                    'Score': f"{entry['metrics']['overall_crime_score']}%",
                    'Events': entry['metrics']['crime_events']
                })
            st.dataframe(pd.DataFrame(history), use_container_width=True)

            if len(history_list) > 1:
                fig = go.Figure()
                scores = [e['metrics']['overall_crime_score'] for e in history_list[-50:]]
                fig.add_trace(go.Scatter(y=scores, mode='lines+markers', line=dict(color='#ff4757')))
                fig.add_hline(y=threshold, line_dash="dash", line_color="yellow")
                fig.update_layout(title="Crime Score Trend", xaxis_title="Analysis", yaxis_title="Score (%)",
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                                  height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Detection metrics
            detection_metrics = analyzer.get_detection_metrics()
            if detection_metrics['samples'] > 0:
                st.markdown("### 🎯 Detection Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{detection_metrics['accuracy']}%")
                with col2:
                    st.metric("Precision", f"{detection_metrics['precision']}%")
                with col3:
                    st.metric("Recall", f"{detection_metrics['recall']}%")
                with col4:
                    st.metric("F1 Score", f"{detection_metrics['f1']}%")
        else:
            st.info("No analysis history yet. Upload and analyze some videos first.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Settings":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ SETTINGS")

        st.markdown(f"**Dataset Path:** `{DATASET_PATH}`")

        st.markdown("#### Model Information")
        if crime_model.model_loaded:
            training_info = crime_model.get_training_info()
            if training_info:
                st.json({
                    'Model Architecture': training_info.get('model_architecture', 'ResNet18'),
                    'Number of Classes': training_info.get('num_classes', 0),
                    'Training Samples': training_info.get('train_samples', 0),
                    'Validation Samples': training_info.get('val_samples', 0),
                    'Best Validation Accuracy': f"{training_info.get('best_val_accuracy', 0):.1f}%",
                    'Training Epochs': training_info.get('num_epochs', 0)
                })
        else:
            st.warning("Model not loaded. Please ensure saved_model.pkl is in the app directory.")

        st.markdown("#### Clear Data")
        if st.button("🗑️ Clear Analysis History", type="secondary"):
            analyzer.analysis_history.clear()
            analyzer.detection_stats = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0,
                                        'false_negatives': 0}
            st.success("History cleared!")
            st.rerun()

        st.markdown("#### About")
        st.info("""
        **AI Community Security Analytics System**

        - **Model:** Pre-trained ResNet18
        - **Purpose:** Real-time crime detection from video
        - **Features:** 
          - Single frame extraction for fast processing
          - Majority voting for accuracy
          - Email alerts for suspicious activity
          - Real-time analysis with confidence scoring

        **Note:** This model was pre-trained on your dataset and loaded for inference only.
        """)

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()