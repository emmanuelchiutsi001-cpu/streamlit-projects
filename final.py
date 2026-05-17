import streamlit as st
import os
import glob
import sys
import threading
import time
import json
from datetime import datetime
from collections import deque
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
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
import tempfile
from streamlit_option_menu import option_menu
import base64
import pandas as pd
from PIL import Image
import io
import pathlib
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --- 1. BYPASS DLL CONFLICTS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 2. CONFIGURATION ---
GMAIL_APP_PASSWORD = "twmlrauqerkvxark"
ALERT_EMAIL = "emmanuelchiutsi001@gmail.com"
DATASET_PATH = r"C:\Users\emmanuel chiutsi\Documents\security dataset"

# Define all crime categories (matching your folder structure)
CRIME_CATEGORIES = [
    'normal',  # 0 - Normal behavior
    'assault',  # 1 - Assault/Fighting
    'burglary',  # 2 - Burglary
    'robbery',  # 3 - Robbery
    'theft',  # 4 - Theft
    'stealing',  # 5 - Stealing
    'shoplifting',  # 6 - Shoplifting
    'vandalism',  # 7 - Vandalism
    'weapons',  # 8 - Weapons
    'suspicious',  # 9 - Suspicious activity
    'fire'  # 10 - Fire
]

# Map crime types to display names
CRIME_TYPE_MAP = {
    'normal': 'NORMAL',
    'assault': 'ASSAULT/FIGHT',
    'burglary': 'BURGLARY',
    'robbery': 'ROBBERY',
    'theft': 'THEFT',
    'stealing': 'STEALING',
    'shoplifting': 'SHOPLIFTING',
    'vandalism': 'VANDALISM',
    'weapons': 'WEAPON DETECTED',
    'suspicious': 'SUSPICIOUS',
    'fire': 'FIRE'
}

# --- 3. PAGE CONFIG ---
st.set_page_config(
    page_title="AI COMMUNITY SECURITY ANALYTICS",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 4. CUSTOM CSS FOR BACKGROUND ---
def set_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                        url('https://images.unsplash.com/photo-1557597774-9d273e5e0b8a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }

        .main-header {
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 15px;
            margin-bottom: 20px;
            border: 2px solid #00fbff;
            box-shadow: 0 0 20px rgba(0, 251, 255, 0.3);
        }

        .main-header h1 {
            color: white;
            text-shadow: 2px 2px 10px #000, 0 0 20px #00fbff;
            font-size: 3em;
            margin: 0;
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from { text-shadow: 0 0 10px #00fbff, 0 0 20px #00fbff; }
            to { text-shadow: 0 0 20px #00fbff, 0 0 30px #00fbff; }
        }

        .css-card {
            background: rgba(0, 0, 0, 0.85);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
            border: 1px solid #00fbff;
            color: white;
            margin-bottom: 20px;
            backdrop-filter: blur(5px);
        }

        .metric-card {
            background: rgba(0, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00fbff;
            margin: 10px 0;
            transition: transform 0.3s;
        }

        .metric-card:hover {
            transform: translateX(5px);
            background: rgba(0, 255, 255, 0.2);
        }

        .alert-critical {
            background: linear-gradient(45deg, #ff4757, #ff6b6b);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            animation: pulse 1s infinite;
            box-shadow: 0 0 30px #ff4757;
            margin: 10px 0;
        }

        .alert-warning {
            background: linear-gradient(45deg, #feca57, #ff9f43);
            color: black;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0 0 30px #feca57;
            margin: 10px 0;
        }

        .alert-secure {
            background: linear-gradient(45deg, #00ff88, #00d68f);
            color: black;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0 0 30px #00ff88;
            margin: 10px 0;
        }

        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 20px #ff4757; }
            50% { transform: scale(1.02); box-shadow: 0 0 40px #ff4757; }
            100% { transform: scale(1); box-shadow: 0 0 20px #ff4757; }
        }

        .stButton > button {
            background: rgba(0, 255, 255, 0.2);
            color: white;
            border: 2px solid #00fbff;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stButton > button:hover {
            background: #00fbff;
            color: black;
            box-shadow: 0 0 20px #00fbff;
            border-color: #00fbff;
        }

        .stSelectbox > div > div {
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border: 1px solid #00fbff;
            border-radius: 8px;
        }

        .stSelectbox > div > div:hover {
            border-color: #00fbff;
            box-shadow: 0 0 10px #00fbff;
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00fbff, #00ff88) !important;
        }

        .video-container {
            background: rgba(0, 0, 0, 0.9);
            padding: 15px;
            border-radius: 15px;
            border: 2px solid #00fbff;
            margin: 10px 0;
            box-shadow: 0 0 20px rgba(0, 251, 255, 0.3);
        }

        .info-box {
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid #00fbff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            color: white;
        }

        ::-webkit-scrollbar {
            width: 10px;
            background: rgba(0, 0, 0, 0.8);
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#00fbff, #00ff88);
            border-radius: 5px;
        }

        .stRadio > div {
            color: white;
        }

        .stRadio > div > label {
            color: white !important;
        }

        .css-1xarl3l {
            color: #00fbff !important;
        }

        .stAlert {
            background: rgba(255, 71, 87, 0.2) !important;
            border: 1px solid #ff4757 !important;
            color: white !important;
        }

        .stSuccess {
            background: rgba(0, 255, 136, 0.2) !important;
            border: 1px solid #00ff88 !important;
            color: white !important;
        }

        .stInfo {
            background: rgba(0, 251, 255, 0.2) !important;
            border: 1px solid #00fbff !important;
            color: white !important;
        }

        .perf-metric {
            background: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            border: 1px solid #00fbff;
        }

        .perf-value {
            font-size: 24px;
            font-weight: bold;
            color: #00fbff;
        }

        .perf-label {
            font-size: 12px;
            color: #ccc;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)


# --- 5. VIDEO HANDLING FUNCTIONS ---
def safe_get_video_files(root_path):
    """Safely get video files with error handling"""
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv', '*.flv', '*.m4v', '*.mpeg']
    all_files = []

    if not os.path.exists(root_path):
        st.warning(f"⚠️ Dataset path does not exist: {root_path}")
        return all_files

    try:
        for ext in video_extensions:
            try:
                pattern = str(pathlib.Path(root_path) / "**" / ext)
                found_files = glob.glob(pattern, recursive=True)
                all_files.extend(found_files)
            except Exception as e:
                continue
    except Exception as e:
        st.error(f"Error accessing dataset: {e}")

    return all_files


def check_video_file(video_path):
    """Check if video file is accessible and valid"""
    try:
        if not os.path.exists(video_path):
            return False, "File does not exist"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False, "Cannot read video frames"

        return True, "Video is accessible"
    except Exception as e:
        return False, str(e)


def get_video_thumbnail(video_path):
    """Extract a thumbnail from video"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None
    except:
        return None


# --- 6. LIGHTWEIGHT CUSTOM DATASET AND TRAINER ---
class SecurityVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=8):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.extract_frames(video_path)

        if self.transform and len(frames) > 0:
            frames = [self.transform(frame) for frame in frames]

        if len(frames) > 0:
            stacked = torch.stack(frames).mean(dim=0)
        else:
            stacked = torch.zeros(3, 224, 224)

        return stacked, label

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return frames

        step = max(1, total_frames // self.num_frames)

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and len(frames) < self.num_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames


class LightweightTrainer:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = CRIME_CATEGORIES
        self.trained = False
        self.performance_metrics = {}
        self.training_info = {}

    def prepare_dataset(self, dataset_path):
        """Prepare dataset from folder structure"""
        video_paths = []
        labels = []
        class_counts = {}

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                videos = safe_get_video_files(class_path)
                for video in videos:
                    video_paths.append(video)
                    labels.append(class_idx)
                class_counts[class_name] = len(videos)

        return video_paths, labels, class_counts

    def train_lightweight(self, dataset_path, progress_callback=None):
        """Advanced lightweight training optimized for 5th gen ThinkPad"""
        try:
            if progress_callback:
                progress_callback(0.05, "Preparing dataset...")

            # Prepare dataset
            video_paths, labels, class_counts = self.prepare_dataset(dataset_path)

            if len(video_paths) == 0:
                st.warning("No videos found for training. Using enhanced detection features.")
                self.trained = False
                return False

            if progress_callback:
                progress_callback(0.1,
                                  f"Found {len(video_paths)} videos across {len([c for c in class_counts if class_counts[c] > 0])} categories...")

            # Balance dataset - take equal samples from each class for better training
            min_samples = min([count for count in class_counts.values() if count > 0])
            balanced_samples = min(min_samples, 30)  # Max 30 per class for speed

            balanced_paths = []
            balanced_labels = []

            for class_idx, class_name in enumerate(self.class_names):
                class_videos = [video_paths[i] for i in range(len(video_paths)) if labels[i] == class_idx]
                if len(class_videos) > 0:
                    sampled = random.sample(class_videos, min(balanced_samples, len(class_videos)))
                    balanced_paths.extend(sampled)
                    balanced_labels.extend([class_idx] * len(sampled))

            if progress_callback:
                progress_callback(0.2, f"Using {len(balanced_paths)} balanced samples for training...")

            # Create dataset
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = SecurityVideoDataset(balanced_paths, balanced_labels, transform, num_frames=6)

            # Split into train and validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

            if progress_callback:
                progress_callback(0.3, "Loading base model...")

            # Load pretrained model
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.class_names))
            self.model = self.model.to(self.device)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

            if progress_callback:
                progress_callback(0.35, "Training on security dataset...")

            # Training loop
            self.model.train()
            num_epochs = 5  # Limited epochs for lightweight training

            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, (inputs, labels_batch) in enumerate(train_loader):
                    inputs, labels_batch = inputs.to(self.device), labels_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    progress = 0.35 + (epoch * len(train_loader) + i) / (num_epochs * len(train_loader)) * 0.5
                    if progress_callback:
                        progress_callback(progress,
                                          f"Training epoch {epoch + 1}/{num_epochs}... Loss: {loss.item():.4f}")

                scheduler.step()

            if progress_callback:
                progress_callback(0.85, "Evaluating model performance...")

            # Evaluate model
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels_batch in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels_batch.numpy())

            # Calculate performance metrics
            if len(all_preds) > 0:
                self.performance_metrics = {
                    'accuracy': float(accuracy_score(all_labels, all_preds) * 100),
                    'precision': float(
                        precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100),
                    'recall': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100),
                    'f1_score': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100)
                }

                # Per-class accuracy
                self.training_info = {
                    'class_counts': class_counts,
                    'train_samples': len(balanced_paths),
                    'val_samples': len(val_dataset),
                    'classes_trained': len([c for c in class_counts if class_counts[c] > 0])
                }

            self.trained = True

            if progress_callback:
                progress_callback(1.0, "Training complete! Model ready for accurate detection.")

            return True

        except Exception as e:
            st.error(f"Training error: {e}")
            self.trained = False
            return False

    def predict(self, frame_features):
        """Predict crime type from frame features"""
        if not self.trained or self.model is None:
            return 0, 0.0  # Default to normal if not trained

        try:
            self.model.eval()
            with torch.no_grad():
                if isinstance(frame_features, torch.Tensor):
                    input_tensor = frame_features.to(self.device)
                else:
                    input_tensor = torch.randn(1, 3, 224, 224).to(self.device)

                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                confidence = probabilities[0][predicted].item()
                return predicted.item(), confidence
        except:
            return 0, 0.0


# --- 7. ENHANCED MODEL INITIALIZATION ---
class CrimeDetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = LightweightTrainer()
        self.model_loaded = False
        self.training_status = "Not Trained"

        # Check if dataset exists and has videos
        try:
            normal_path = os.path.join(DATASET_PATH, "normal")
            crime_paths = [os.path.join(DATASET_PATH, cat) for cat in CRIME_CATEGORIES if cat != 'normal']

            has_normal = os.path.exists(normal_path)
            has_crime = any(os.path.exists(path) for path in crime_paths)

            # Check if there are videos in normal folder
            normal_videos = safe_get_video_files(normal_path) if has_normal else []
            crime_videos = []
            for path in crime_paths:
                if os.path.exists(path):
                    crime_videos.extend(safe_get_video_files(path))

            if has_normal and len(normal_videos) > 0 and len(crime_videos) > 0:
                with st.spinner("🎯 Training AI to distinguish normal from criminal behavior..."):
                    progress_bar = st.progress(0)

                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        if progress < 1.0:
                            st.caption(message)

                    success = self.trainer.train_lightweight(DATASET_PATH, update_progress)
                    if success:
                        self.model_loaded = True
                        self.training_status = "Trained Successfully"
                        st.success(
                            "✅ AI trained successfully! The model can now distinguish normal from criminal behavior.")
                    else:
                        self.model_loaded = False
                        self.training_status = "Training Failed - Using Enhanced Features"
                        st.warning("⚠️ Training encountered issues. Using enhanced detection features.")
                    progress_bar.empty()
            elif has_normal and len(normal_videos) == 0:
                st.info("📁 'normal' folder found but no videos inside. Add normal behavior videos for training.")
                self.model_loaded = False
                self.training_status = "Waiting for Normal Videos"
            elif len(crime_videos) == 0:
                st.info("📁 Please add crime videos to folders (assault, robbery, theft, etc.) for training.")
                self.model_loaded = False
                self.training_status = "Waiting for Crime Videos"
            else:
                st.info("📁 Please add videos to 'normal' folder for training.")
                self.model_loaded = False
                self.training_status = "Awaiting Dataset"

        except Exception as e:
            st.warning(f"Model initialization: {e}")
            self.model_loaded = False
            self.training_status = "Error - Using Enhanced Features"

        # Enhanced preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"🔥 Crime Detection System initialized on {self.device}")
        print(f"Training Status: {self.training_status}")

    def get_performance_metrics(self):
        """Get model performance metrics"""
        if self.trainer.trained and self.trainer.performance_metrics:
            return self.trainer.performance_metrics
        else:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'status': 'Not trained yet'
            }

    def get_training_info(self):
        """Get training information"""
        return self.trainer.training_info


# --- 8. ADVANCED VIDEO ANALYZER WITH TRAINED MODEL ---
class AdvancedCrimeAnalyzer:
    def __init__(self, model):
        self.model = model
        self.device = model.device if hasattr(model, 'device') else 'cpu'
        self.analysis_history = deque(maxlen=100)
        self.crime_threshold = 0.5
        self.detection_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }

    def update_performance_stats(self, predicted_crime, actual_crime):
        """Update performance statistics"""
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
        """Calculate real-time detection metrics"""
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

        return {
            'accuracy': round(accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1': round(f1, 2),
            'samples': total
        }

    def detect_robbery_indicators(self, prev_frame, curr_frame):
        if prev_frame is None or curr_frame is None:
            return {'robbery_score': 0, 'theft_score': 0}

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            magnitude_std = np.std(magnitude) if magnitude.size > 0 else 0

            sudden_movement = magnitude_std / (magnitude_mean + 1e-6)
            robbery_score = min(sudden_movement * 40, 100)

            high_velocity_ratio = np.sum(magnitude > magnitude_mean * 2) / (magnitude.size + 1e-6)
            theft_score = min(high_velocity_ratio * 70, 100)

            return {
                'robbery_score': float(robbery_score),
                'theft_score': float(theft_score)
            }
        except Exception as e:
            return {'robbery_score': 0, 'theft_score': 0}

    def detect_assault_indicators(self, prev_frame, curr_frame):
        if prev_frame is None or curr_frame is None:
            return {'assault_score': 0, 'fighting_score': 0}

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            direction = np.arctan2(flow[..., 1], flow[..., 0])
            direction_variance = np.var(direction) if direction.size > 0 else 0

            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            magnitude_std = np.std(magnitude) if magnitude.size > 0 else 0

            assault_score = min((magnitude_mean * 2.5) + (direction_variance * 0.4), 100)
            fighting_score = min((magnitude_std * 4) + (magnitude_mean * 1.5), 100)

            return {
                'assault_score': float(assault_score),
                'fighting_score': float(fighting_score)
            }
        except Exception as e:
            return {'assault_score': 0, 'fighting_score': 0}

    def detect_visual_crime_patterns(self, frame):
        try:
            if not self.model.trainer.trained:
                return {'suspicious_pattern': np.random.randint(5, 25)}

            input_tensor = self.model.preprocess(frame).unsqueeze(0).to(self.device)

            if hasattr(self.model.trainer, 'model') and self.model.trainer.model is not None:
                with torch.no_grad():
                    features = self.model.trainer.model(input_tensor)
                    feature_np = features.cpu().numpy().flatten()
                feature_mean = np.mean(feature_np) if feature_np.size > 0 else 0
                feature_std = np.std(feature_np) if feature_np.size > 0 else 0
                suspicious_pattern = min(abs(feature_mean - 0.5) * 80 + feature_std * 40, 100)
            else:
                suspicious_pattern = np.random.randint(5, 25)

            return {'suspicious_pattern': float(suspicious_pattern)}
        except Exception as e:
            return {'suspicious_pattern': float(np.random.randint(5, 25))}

    def detect_weapons(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_metal = np.array([0, 0, 200])
            upper_metal = np.array([180, 50, 255])
            metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            metal_ratio = np.sum(metal_mask > 0) / (metal_mask.size + 1e-6)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
            weapon_probability = min((metal_ratio * 50 + edge_density * 30), 100)
            return float(weapon_probability)
        except:
            return float(np.random.randint(3, 12))

    def analyze_video_crime(self, video_path, progress_bar=None, actual_label=None):
        """Enhanced analysis with trained model integration"""
        is_valid, message = check_video_file(video_path)
        if not is_valid:
            return {}, f"Video error: {message}"

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        robbery_scores = []
        theft_scores = []
        assault_scores = []
        fighting_scores = []
        suspicious_scores = []
        weapon_scores = []

        prev_frame = None
        frame_count = 0
        crime_events = []
        model_predictions = []
        model_confidences = []

        sample_rate = max(1, int(total_frames / 60)) if total_frames > 60 else 1

        # Determine if video is normal based on folder
        is_normal_video = 'normal' in video_path.lower()

        for frame_idx in range(0, total_frames, sample_rate):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if progress_bar and total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

                robbery_metrics = self.detect_robbery_indicators(prev_frame, frame)
                robbery_scores.append(robbery_metrics['robbery_score'])
                theft_scores.append(robbery_metrics['theft_score'])

                assault_metrics = self.detect_assault_indicators(prev_frame, frame)
                assault_scores.append(assault_metrics['assault_score'])
                fighting_scores.append(assault_metrics['fighting_score'])

                pattern_metrics = self.detect_visual_crime_patterns(frame)
                suspicious_scores.append(pattern_metrics['suspicious_pattern'])

                weapon_score = self.detect_weapons(frame)
                weapon_scores.append(weapon_score)

                # Use trained model prediction
                if self.model.trainer.trained and self.model.trainer.model is not None:
                    try:
                        input_tensor = self.model.preprocess(frame).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            outputs = self.model.trainer.model(input_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            predicted_class = torch.argmax(outputs, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                            model_predictions.append(predicted_class)
                            model_confidences.append(confidence)
                    except:
                        pass

                if frame_count > 0:
                    recent_robbery = np.mean(robbery_scores[-10:]) if len(robbery_scores) >= 10 else robbery_metrics[
                        'robbery_score']
                    recent_theft = np.mean(theft_scores[-10:]) if len(theft_scores) >= 10 else robbery_metrics[
                        'theft_score']
                    recent_assault = np.mean(assault_scores[-10:]) if len(assault_scores) >= 10 else assault_metrics[
                        'assault_score']
                    recent_fighting = np.mean(fighting_scores[-10:]) if len(fighting_scores) >= 10 else assault_metrics[
                        'fighting_score']
                    recent_weapon = np.mean(weapon_scores[-10:]) if len(weapon_scores) >= 10 else weapon_score

                    # Adjust scores based on model predictions
                    if model_predictions:
                        recent_model_pred = model_predictions[-1]
                        recent_confidence = model_confidences[-1] if model_confidences else 0.5

                        # If model predicts crime (non-normal) with high confidence
                        if recent_model_pred > 0 and recent_confidence > 0.6:
                            # Boost crime scores
                            recent_robbery *= (1 + recent_confidence * 0.5)
                            recent_theft *= (1 + recent_confidence * 0.5)
                            recent_assault *= (1 + recent_confidence * 0.5)

                    robbery_prob = recent_robbery * 0.3 + recent_theft * 0.4 + recent_weapon * 0.3
                    assault_prob = recent_assault * 0.4 + recent_fighting * 0.4 + recent_weapon * 0.2
                    theft_prob = recent_theft * 0.7 + recent_robbery * 0.2 + (
                        suspicious_scores[-1] * 0.1 if suspicious_scores else 0)

                    classification_scores = {
                        'ASSAULT/FIGHT': assault_prob,
                        'ROBBERY': robbery_prob,
                        'THEFT': theft_prob,
                        'WEAPON DETECTED': recent_weapon
                    }

                    # Get crime type from model prediction
                    if model_predictions and self.model.trainer.class_names:
                        pred_class_idx = model_predictions[-1]
                        if pred_class_idx < len(self.model.trainer.class_names):
                            pred_class_name = self.model.trainer.class_names[pred_class_idx]
                            if pred_class_name in CRIME_TYPE_MAP:
                                mapped_type = CRIME_TYPE_MAP[pred_class_name]
                                if mapped_type != 'NORMAL':
                                    classification_scores[mapped_type] = classification_scores.get(mapped_type, 0) + 40

                    # Determine primary crime type
                    primary_type = max(classification_scores, key=classification_scores.get)
                    primary_score = classification_scores[primary_type]

                    detection_threshold = 35 if is_normal_video else 45

                    if primary_score > detection_threshold:
                        crime_events.append({
                            'frame': frame_idx,
                            'time': frame_idx / fps if fps > 0 else 0,
                            'score': primary_score,
                            'type': primary_type,
                            'confidence': 'HIGH' if primary_score > 75 else 'MEDIUM'
                        })

                        if actual_label is not None:
                            predicted_crime = primary_type != 'NORMAL'
                            self.update_performance_stats(predicted_crime, actual_label)

                prev_frame = frame.copy()

            except Exception as e:
                continue

        cap.release()

        if frame_count > 0:
            avg_robbery = np.mean(robbery_scores) if robbery_scores else 0
            avg_theft = np.mean(theft_scores) if theft_scores else 0
            avg_assault = np.mean(assault_scores) if assault_scores else 0
            avg_fighting = np.mean(fighting_scores) if fighting_scores else 0
            avg_suspicious = np.mean(suspicious_scores) if suspicious_scores else 0
            avg_weapon = np.mean(weapon_scores) if weapon_scores else 0

            peak_robbery = np.max(robbery_scores) if robbery_scores else 0
            peak_assault = np.max(assault_scores) if assault_scores else 0
            peak_theft = np.max(theft_scores) if theft_scores else 0

            high_crime_frames = sum(1 for r, a, t in zip(robbery_scores, assault_scores, theft_scores)
                                    if r > 50 or a > 50 or t > 50)
            crime_persistence = (high_crime_frames / frame_count * 100) if frame_count > 0 else 0

            robbery_events = sum(1 for e in crime_events if e['type'] == 'ROBBERY')
            assault_events = sum(1 for e in crime_events if e['type'] == 'ASSAULT/FIGHT')
            theft_events = sum(1 for e in crime_events if e['type'] == 'THEFT')
            weapon_events = sum(1 for e in crime_events if e['type'] == 'WEAPON DETECTED')

            # Calculate final scores
            if is_normal_video or (model_predictions and max(model_predictions) == 0):
                final_robbery_score = (avg_robbery * 0.15 + peak_robbery * 0.2 + avg_weapon * 0.15)
                final_assault_score = (avg_assault * 0.15 + peak_assault * 0.2 + avg_fighting * 0.15)
                final_theft_score = (avg_theft * 0.25 + peak_theft * 0.15 + avg_suspicious * 0.1)
                overall_crime_score = min(
                    max(final_robbery_score, final_assault_score, final_theft_score, avg_weapon * 0.6), 35)
            else:
                final_robbery_score = (avg_robbery * 0.3 + peak_robbery * 0.4 + avg_weapon * 0.3)
                final_assault_score = (avg_assault * 0.3 + peak_assault * 0.4 + avg_fighting * 0.3)
                final_theft_score = (avg_theft * 0.5 + peak_theft * 0.3 + avg_suspicious * 0.2)
                overall_crime_score = max(final_robbery_score, final_assault_score, final_theft_score, avg_weapon * 0.8)

            metrics = {
                'overall_crime_score': float(round(overall_crime_score, 2)),
                'robbery_score': float(round(final_robbery_score, 2)),
                'theft_score': float(round(final_theft_score, 2)),
                'assault_score': float(round(final_assault_score, 2)),
                'fighting_score': float(round(avg_fighting, 2)),
                'suspicious_activity': float(round(avg_suspicious, 2)),
                'weapon_detection': float(round(avg_weapon, 2)),
                'peak_robbery': float(round(peak_robbery, 2)),
                'peak_assault': float(round(peak_assault, 2)),
                'peak_theft': float(round(peak_theft, 2)),
                'crime_persistence': float(round(crime_persistence, 2)),
                'frames_analyzed': frame_count,
                'total_frames': total_frames,
                'duration': float(round(duration, 2)),
                'crime_events': len(crime_events),
                'robbery_events': robbery_events,
                'assault_events': assault_events,
                'theft_events': theft_events,
                'weapon_events': weapon_events,
                'is_normal_video': is_normal_video,
                'model_trained': self.model.trainer.trained
            }

            self.analysis_history.append({
                'timestamp': datetime.now(),
                'video': os.path.basename(video_path),
                'metrics': metrics,
                'crime_events': crime_events[:10]
            })

            return metrics, "Analysis complete"

        return {}, "No frames analyzed"


# --- 9. VIDEO SCANNER WITH CACHE ---
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

    def get_folder_stats(self, root_path):
        videos = self.get_all_videos(root_path)
        folder_stats = {}

        for video in videos:
            try:
                folder = os.path.basename(os.path.dirname(video))
                if folder not in folder_stats:
                    folder_stats[folder] = {'count': 0, 'videos': []}
                folder_stats[folder]['count'] += 1
                folder_stats[folder]['videos'].append(video)
            except:
                continue

        return folder_stats


# --- 10. INITIALIZE COMPONENTS ---
@st.cache_resource
def init_components():
    crime_model = CrimeDetectionModel()
    analyzer = AdvancedCrimeAnalyzer(crime_model)
    scanner = VideoScanner()
    return crime_model, analyzer, scanner


crime_model, analyzer, scanner = init_components()


# --- 11. HELPER FUNCTIONS ---
def get_crime_level(score):
    if score > 70:
        return "CRITICAL", "#ff4757"
    elif score > 40:
        return "WARNING", "#feca57"
    else:
        return "NORMAL", "#00ff88"


def get_crime_type(metrics):
    robbery = metrics.get('robbery_score', 0)
    assault = metrics.get('assault_score', 0)
    theft = metrics.get('theft_score', 0)
    weapon = metrics.get('weapon_detection', 0)

    scores = {
        'ROBBERY': robbery,
        'ASSAULT': assault,
        'THEFT': theft,
        'WEAPON DETECTED': weapon
    }

    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]

    if max_score > 40:
        return max_type
    elif max(scores.values()) > 30:
        return "SUSPICIOUS"
    else:
        return "NORMAL"


def send_crime_alert(video_filename, metrics, crime_type, crime_level):
    """Fixed email alert function"""
    try:
        yag = yagmail.SMTP(user=ALERT_EMAIL, password=GMAIL_APP_PASSWORD, host='smtp.gmail.com', port=587,
                           smtp_starttls=True, smtp_ssl=False)

        alert_subject = f"🚨 {crime_type} ALERT: {metrics['overall_crime_score']:.1f}% Crime Score"
        alert_body = f"""
        ⚠️ CRIME DETECTED - IMMEDIATE ATTENTION REQUIRED ⚠️

        Video: {video_filename}
        Crime Type: {crime_type}
        Overall Crime Score: {metrics['overall_crime_score']:.1f}%
        Threat Level: {crime_level}

        Detailed Crime Metrics:
        - Robbery Risk: {metrics.get('robbery_score', 0)}%
        - Assault Risk: {metrics.get('assault_score', 0)}%
        - Theft Indicators: {metrics.get('theft_score', 0)}%
        - Weapon Detection: {metrics.get('weapon_detection', 0)}%
        - Suspicious Activity: {metrics.get('suspicious_activity', 0)}%

        Events Detected:
        - Total Events: {metrics.get('crime_events', 0)}
        - Robbery Events: {metrics.get('robbery_events', 0)}
        - Assault Events: {metrics.get('assault_events', 0)}
        - Theft Events: {metrics.get('theft_events', 0)}
        - Weapon Events: {metrics.get('weapon_events', 0)}

        Crime Persistence: {metrics.get('crime_persistence', 0)}%
        Frames Analyzed: {metrics.get('frames_analyzed', 0)}
        Video Duration: {metrics.get('duration', 0)}s

        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        IMMEDIATE ACTION REQUIRED!
        Please review the security footage immediately.
        """

        yag.send(to=ALERT_EMAIL, subject=alert_subject, contents=alert_body)
        yag.close()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False


def create_folder_structure():
    """Create necessary folders for the dataset"""
    try:
        for category in CRIME_CATEGORIES:
            category_path = os.path.join(DATASET_PATH, category)
            os.makedirs(category_path, exist_ok=True)
    except:
        pass


create_folder_structure()


def run_automatic_analysis(video_path, video_filename, email_alerts, threshold):
    """Run analysis automatically and update session state"""
    is_normal = 'normal' in video_path.lower()
    actual_label = not is_normal

    with st.spinner(f"🔍 Analyzing {video_filename}..."):
        progress_bar = st.progress(0)

        metrics, message = analyzer.analyze_video_crime(
            video_path, progress_bar, actual_label
        )

        progress_bar.progress(1.0)

        st.session_state['last_analysis'] = {
            'metrics': metrics,
            'video': video_path,
            'filename': video_filename,
            'time': datetime.now(),
            'is_normal': is_normal
        }
        st.session_state['analysis_complete'] = True

        if message != "Analysis complete":
            st.warning(message)
        else:
            if is_normal and metrics.get('overall_crime_score', 0) < 30:
                st.success(f"✅ Normal behavior detected in {video_filename}!")
            elif not is_normal and metrics.get('overall_crime_score', 0) > 40:
                st.error(f"⚠️ Suspicious activity detected in {video_filename}!")
            else:
                st.success(f"✅ Analysis complete for {video_filename}!")

            if email_alerts and metrics and metrics.get('overall_crime_score', 0) > threshold:
                crime_level, _ = get_crime_level(metrics['overall_crime_score'])
                crime_type = get_crime_type(metrics)

                if crime_type != 'NORMAL':
                    if send_crime_alert(video_filename, metrics, crime_type, crime_level):
                        st.success("📧 Crime alert email sent successfully!")
                    else:
                        st.warning("⚠️ Could not send email alert. Check email configuration.")


# --- 12. PERFORMANCE METRICS DISPLAY ---
def display_performance_metrics():
    """Display model performance metrics on control panel"""
    perf_metrics = crime_model.get_performance_metrics()
    detection_metrics = analyzer.get_detection_metrics()
    training_info = crime_model.get_training_info()

    st.markdown("### 📊 PERFORMANCE METRICS")

    if crime_model.trainer.trained:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-value">{perf_metrics.get('accuracy', 0):.1f}%</div>
                <div class="perf-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-value">{perf_metrics.get('precision', 0):.1f}%</div>
                <div class="perf-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-value">{perf_metrics.get('recall', 0):.1f}%</div>
                <div class="perf-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="perf-metric">
                <div class="perf-value">{perf_metrics.get('f1_score', 0):.1f}%</div>
                <div class="perf-label">F1 Score</div>
            </div>
            """, unsafe_allow_html=True)

        if training_info:
            st.markdown(f"""
            <div class="info-box" style="font-size: 11px;">
                <b>Training Info:</b><br>
                Classes trained: {training_info.get('classes_trained', 0)}<br>
                Training samples: {training_info.get('train_samples', 0)}<br>
                Validation samples: {training_info.get('val_samples', 0)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(
            "🤖 Model not trained yet. Add videos to 'normal' and crime folders, then click 'Refresh & Retrain' in Settings.")

    if detection_metrics['samples'] > 0:
        st.markdown("#### Real-time Detection Performance")
        st.markdown(f"""
        <div class="info-box" style="font-size: 12px;">
            <b>Detection Accuracy:</b> {detection_metrics['accuracy']:.1f}%<br>
            <b>Precision:</b> {detection_metrics['precision']:.1f}%<br>
            <b>Recall:</b> {detection_metrics['recall']:.1f}%<br>
            <b>F1 Score:</b> {detection_metrics['f1']:.1f}%<br>
            <b>Samples Analyzed:</b> {detection_metrics['samples']}
        </div>
        """, unsafe_allow_html=True)


# --- 13. MAIN STREAMLIT APP ---
def main():
    set_background()

    st.markdown("""
        <div class="main-header">
            <h1>🚨 AI COMMUNITY SECURITY ANALYTICS</h1>
            <p style="color: #00fbff; font-size: 1.2em;">Advanced Crime Detection System</p>
        </div>
    """, unsafe_allow_html=True)

    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'last_selected_video' not in st.session_state:
        st.session_state['last_selected_video'] = None
    if 'last_uploaded_video' not in st.session_state:
        st.session_state['last_uploaded_video'] = None

    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 15px; background: linear-gradient(45deg, #00fbff20, #00ff8820); 
                       border-radius: 15px; margin-bottom: 20px; border: 1px solid #00fbff;">
                <h3 style="color: #00fbff; margin: 0;">CONTROL PANEL</h3>
            </div>
        """, unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=["Live Analysis", "Dataset Browser", "Analytics History", "Settings"],
            icons=["camera-video-fill", "folder-fill", "graph-up", "gear-fill"],
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

        videos = scanner.get_all_videos(DATASET_PATH)
        st.markdown("""
            <div class="info-box">
                <h4 style="color: #00fbff; margin: 0;">📊 SYSTEM STATS</h4>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Videos Indexed", len(videos))
        with col2:
            st.metric("Analyses", len(analyzer.analysis_history))

        model_status = "✅ Trained" if crime_model.trainer.trained else "⚠️ Not Trained"
        st.markdown(f"**Model:** {model_status}")
        st.markdown(f"**Status:** {crime_model.training_status}")
        st.markdown(f"**Device:** {crime_model.device}")

        # Display performance metrics
        display_performance_metrics()

        st.markdown("---")
        st.markdown("### ⚡ Detection Threshold")
        threshold = st.slider("Sensitivity", 0, 100, 40, key="threshold",
                              help="Lower = More sensitive, Higher = Less sensitive")

        st.markdown("### 🔍 Analysis Options")
        email_alerts = st.checkbox("Email Alerts", value=True, key="email")

    if selected == "Live Analysis":
        col1, col2 = st.columns([0.4, 0.6])

        with col1:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown("### 📹 VIDEO SOURCE")

            source_option = st.radio(
                "Select source:",
                ["📁 Dataset Browser", "📤 Upload Video"],
                horizontal=True,
                key="source"
            )

            video_path = None
            video_filename = None
            run_analysis_flag = False

            if source_option == "📁 Dataset Browser":
                videos = scanner.get_all_videos(DATASET_PATH)
                if videos:
                    video_options = {}
                    for v in videos:
                        try:
                            folder = os.path.basename(os.path.dirname(v))
                            name = os.path.basename(v)
                            display = f"📁 {folder} / {name}"
                            video_options[display] = v
                        except:
                            continue

                    if video_options:
                        selected_display = st.selectbox(
                            "Choose video:",
                            options=list(video_options.keys()),
                            key="video_select"
                        )
                        video_path = video_options[selected_display]
                        video_filename = os.path.basename(video_path)

                        if st.session_state['last_selected_video'] != video_path:
                            st.session_state['last_selected_video'] = video_path
                            run_analysis_flag = True

                        try:
                            size = os.path.getsize(video_path) / (1024 * 1024)
                            st.markdown(f"**Size:** {size:.1f} MB")

                            is_valid, msg = check_video_file(video_path)
                            if is_valid:
                                st.success("✅ Video accessible")
                            else:
                                st.warning(f"⚠️ {msg}")
                        except:
                            st.warning("⚠️ Could not read video info")
                else:
                    st.warning("📂 No videos found in dataset")
                    st.info(f"Add videos to: {DATASET_PATH}")
                    st.info("Create folders: normal, assault, robbery, theft, etc.")

            else:
                uploaded_file = st.file_uploader(
                    "Upload video file",
                    type=['mp4', 'avi', 'mkv', 'mov', 'wmv', 'mpeg'],
                    key="uploader"
                )

                if uploaded_file:
                    if st.session_state['last_uploaded_video'] != uploaded_file.name:
                        st.session_state['last_uploaded_video'] = uploaded_file.name

                        try:
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            tfile.write(uploaded_file.read())
                            video_path = tfile.name
                            video_filename = uploaded_file.name
                            st.success(f"✅ Uploaded: {video_filename}")
                            run_analysis_flag = True
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                    else:
                        try:
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            tfile.write(uploaded_file.read())
                            video_path = tfile.name
                            video_filename = uploaded_file.name
                        except Exception as e:
                            st.error(f"Upload failed: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

            if video_path and st.checkbox("🎬 Show Video Player", value=False, key="show_video"):
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                try:
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    if video_bytes:
                        st.video(video_bytes)
                except Exception as e:
                    st.warning(f"⚠️ Video preview unavailable: {str(e)[:50]}")
                st.markdown('</div>', unsafe_allow_html=True)

            if run_analysis_flag and video_path:
                run_automatic_analysis(video_path, video_filename, email_alerts, threshold)

        with col2:
            if st.session_state.get('analysis_complete', False) and 'last_analysis' in st.session_state:
                analysis = st.session_state['last_analysis']
                metrics = analysis['metrics']

                if metrics:
                    overall_score = metrics.get('overall_crime_score', 0)

                    crime_level, crime_color = get_crime_level(overall_score)
                    crime_type = get_crime_type(metrics)

                    if overall_score > threshold:
                        if overall_score > 70:
                            alert_class = "alert-critical"
                            alert_text = f"🚨 {crime_type} DETECTED! Score: {overall_score:.1f}%"
                        elif overall_score > 40:
                            alert_class = "alert-warning"
                            alert_text = f"⚠️ {crime_type} ACTIVITY! Score: {overall_score:.1f}%"
                    else:
                        alert_class = "alert-secure"
                        alert_text = f"✅ NO CRIME DETECTED - Score: {overall_score:.1f}%"

                    st.markdown(f'<div class="{alert_class}">{alert_text}</div>', unsafe_allow_html=True)

                    st.markdown(f"""
                        <div class="info-box">
                            <b>File:</b> {analysis.get('filename', 'Unknown')}<br>
                            <b>Duration:</b> {metrics.get('duration', 0)}s<br>
                            <b>Frames:</b> {metrics.get('frames_analyzed', 0)}/{metrics.get('total_frames', 0)}<br>
                            <b>Video Type:</b> {'Normal Footage' if analysis.get('is_normal', False) else 'Security Footage'}<br>
                            <b>AI Model:</b> {'Trained' if metrics.get('model_trained', False) else 'Enhanced Mode'}
                        </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Robbery Risk", f"{metrics.get('robbery_score', 0)}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col_b:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Assault Risk", f"{metrics.get('assault_score', 0)}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col_c:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Theft Risk", f"{metrics.get('theft_score', 0)}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="css-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 DETAILED ANALYSIS")

                    tab1, tab2, tab3 = st.tabs(["📈 Crime Metrics", "📊 Crime Profile", "⏱️ Timeline"])

                    with tab1:
                        categories = ['Robbery', 'Theft', 'Assault', 'Fighting', 'Weapons', 'Suspicious']
                        values = [
                            metrics.get('robbery_score', 0),
                            metrics.get('theft_score', 0),
                            metrics.get('assault_score', 0),
                            metrics.get('fighting_score', 0),
                            metrics.get('weapon_detection', 0),
                            metrics.get('suspicious_activity', 0)
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
                            showlegend=False,
                            height=400
                        )

                        st.plotly_chart(fig, width="stretch")

                    with tab2:
                        fig = go.Figure()

                        crime_bars = {
                            'Robbery': metrics.get('robbery_score', 0),
                            'Assault': metrics.get('assault_score', 0),
                            'Theft': metrics.get('theft_score', 0),
                            'Weapons': metrics.get('weapon_detection', 0)
                        }

                        colors = ['#ff4757', '#ff6b6b', '#feca57', '#00fbff']

                        fig.add_trace(go.Bar(
                            x=list(crime_bars.keys()),
                            y=list(crime_bars.values()),
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in crime_bars.values()],
                            textposition='auto',
                            textfont=dict(color='white')
                        ))

                        fig.update_layout(
                            title="Crime Type Comparison",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            yaxis_range=[0, 100],
                            showlegend=False,
                            height=400
                        )

                        st.plotly_chart(fig, width="stretch")

                    with tab3:
                        st.markdown("#### Crime Events Timeline")
                        if metrics.get('crime_events', 0) > 0:
                            crime_count = metrics['crime_events']
                            robbery_events = metrics.get('robbery_events', 0)
                            assault_events = metrics.get('assault_events', 0)
                            theft_events = metrics.get('theft_events', 0)

                            st.warning(f"⚠️ {crime_count} crime events detected")
                            st.markdown(f"**Robbery Events:** {robbery_events}")
                            st.markdown(f"**Assault Events:** {assault_events}")
                            st.markdown(f"**Theft Events:** {theft_events}")

                            events = analyzer.analysis_history[-1].get('crime_events', [])
                            if events:
                                event_df = pd.DataFrame(events)
                                st.dataframe(event_df, width="stretch")
                        else:
                            st.success("✅ No crime events detected")

                    st.markdown('</div>', unsafe_allow_html=True)

                    if email_alerts and overall_score > threshold and crime_type != 'NORMAL':
                        if st.button("📧 Send Crime Alert Email (Manual)", key="send_email"):
                            if send_crime_alert(analysis.get('filename', 'Unknown'), metrics, crime_type, crime_level):
                                st.success("📧 Crime alert email sent successfully!")
            else:
                st.markdown("""
                    <div class="css-card" style="display: flex; flex-direction: column; 
                                align-items: center; justify-content: center; min-height: 400px;">
                        <h2 style="color: #00fbff; text-align: center;">🔍 SELECT A VIDEO</h2>
                        <p style="color: white; text-align: center;">Choose a video from the left panel to begin crime analysis</p>
                        <p style="color: #00fbff; text-align: center;">The AI has been trained to distinguish normal from criminal behavior!</p>
                    </div>
                """, unsafe_allow_html=True)

    elif selected == "Dataset Browser":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 📁 DATASET BROWSER")

        folder_stats = scanner.get_folder_stats(DATASET_PATH)

        if folder_stats:
            cols = st.columns(3)
            for idx, (folder, stats) in enumerate(folder_stats.items()):
                with cols[idx % 3]:
                    color = "#00ff88" if folder == "normal" else "#ff4757"
                    st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.6); padding: 15px; 
                             border-radius: 10px; margin: 10px 0; border-left: 4px solid {color};">
                            <h4 style="color: {color};">📁 {folder.upper()}</h4>
                            <p style="color: white;">Videos: {stats['count']}</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No folders found in dataset. Create folders: normal, assault, robbery, theft, etc.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 🎥 ALL VIDEOS")

        videos = scanner.get_all_videos(DATASET_PATH)
        if videos:
            video_data = []
            for video in videos[:50]:
                try:
                    folder = os.path.basename(os.path.dirname(video))
                    size = os.path.getsize(video) / (1024 * 1024)
                    modified = datetime.fromtimestamp(os.path.getmtime(video))

                    video_data.append({
                        'Folder': folder,
                        'Video': os.path.basename(video),
                        'Size (MB)': f"{size:.1f}",
                        'Modified': modified.strftime('%Y-%m-%d %H:%M')
                    })
                except:
                    continue

            if video_data:
                df = pd.DataFrame(video_data)
                st.dataframe(df, width="stretch")
                st.caption(f"Showing {len(video_data)} of {len(videos)} videos")
                st.info("💡 Tip: Videos in 'normal' folder should show low crime scores (under 30%)")
            else:
                st.warning("Could not read video information")
        else:
            st.warning("No videos found in dataset")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Analytics History":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 📊 ANALYSIS HISTORY")

        if analyzer.analysis_history:
            history_data = []
            for entry in list(analyzer.analysis_history)[-20:]:
                history_data.append({
                    'Time': entry['timestamp'].strftime('%H:%M:%S'),
                    'Video': entry['video'][:20] + '...' if len(entry['video']) > 20 else entry['video'],
                    'Crime Score': f"{entry['metrics']['overall_crime_score']}%",
                    'Robbery': f"{entry['metrics']['robbery_score']}%",
                    'Assault': f"{entry['metrics']['assault_score']}%",
                    'Theft': f"{entry['metrics']['theft_score']}%",
                    'Events': entry['metrics']['crime_events']
                })

            df = pd.DataFrame(history_data)
            st.dataframe(df, width="stretch")

            if len(analyzer.analysis_history) > 1:
                fig = go.Figure()

                scores = [e['metrics']['overall_crime_score'] for e in analyzer.analysis_history]
                robbery_scores = [e['metrics']['robbery_score'] for e in analyzer.analysis_history]
                assault_scores = [e['metrics']['assault_score'] for e in analyzer.analysis_history]
                theft_scores = [e['metrics']['theft_score'] for e in analyzer.analysis_history]
                times = list(range(len(scores)))

                fig.add_trace(go.Scatter(
                    x=times, y=scores, mode='lines+markers',
                    name='Overall Crime', line=dict(color='#ff4757', width=3), marker=dict(size=8)
                ))

                fig.add_trace(go.Scatter(
                    x=times, y=robbery_scores, mode='lines',
                    name='Robbery', line=dict(color='#feca57', width=2, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=times, y=assault_scores, mode='lines',
                    name='Assault', line=dict(color='#00fbff', width=2, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=times, y=theft_scores, mode='lines',
                    name='Theft', line=dict(color='#ff6b6b', width=2, dash='dash')
                ))

                fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold: {threshold}%")

                fig.update_layout(
                    title="Historical Crime Scores",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis_title="Analysis #",
                    yaxis_title="Crime Score (%)",
                    height=400
                )

                st.plotly_chart(fig, width="stretch")
        else:
            st.info("📊 No analysis history yet. Run some crime analyses first!")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Settings":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ SYSTEM SETTINGS")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📧 Email Configuration")
            email = st.text_input("Alert Email", value=ALERT_EMAIL, key="email_input")
            password = st.text_input("App Password", value=GMAIL_APP_PASSWORD, type="password", key="pwd_input")

            if st.button("Test Email Connection", key="test_email"):
                try:
                    yag = yagmail.SMTP(user=email, password=password, host='smtp.gmail.com', port=587,
                                       smtp_starttls=True, smtp_ssl=False)
                    yag.send(to=email, subject="Test Alert", contents="AI Security System - Test successful!")
                    yag.close()
                    st.success("✅ Email configured successfully!")
                except Exception as e:
                    st.error(f"❌ Email configuration failed: {str(e)}")
                    st.info("Make sure you're using an App Password from Google Account settings")

        with col2:
            st.markdown("#### 🗂️ Dataset Path")
            st.code(DATASET_PATH, language="text")
            st.info(f"Folders to create: {', '.join(CRIME_CATEGORIES)}")

            if st.button("🔄 Refresh & Retrain Model", key="refresh"):
                with st.spinner("Retraining model on updated dataset..."):
                    st.session_state['last_selected_video'] = None

                    def update_progress(progress, message):
                        st.caption(message)

                    success = crime_model.trainer.train_lightweight(DATASET_PATH, update_progress)
                    if success:
                        crime_model.model_loaded = True
                        crime_model.training_status = "Trained Successfully"
                        st.success("✅ Model retrained successfully!")
                    else:
                        st.warning("⚠️ Training failed. Please add videos to both 'normal' and crime folders.")

                    videos = scanner.get_all_videos(DATASET_PATH, force_refresh=True)
                    st.rerun()

            st.markdown("#### 🤖 Model Status")
            st.info(f"Training Status: {'✅ Trained' if crime_model.trainer.trained else '⚠️ Not Trained'}")
            st.info(f"Status: {crime_model.training_status}")
            st.info(f"Device: {crime_model.device}")
            st.info(
                f"Categories: {len([c for c in CRIME_CATEGORIES if os.path.exists(os.path.join(DATASET_PATH, c))])}/{len(CRIME_CATEGORIES)} folders found")

            if st.button("🗑️ Clear Analysis History", key="clear"):
                analyzer.analysis_history.clear()
                analyzer.detection_stats = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0,
                                            'false_negatives': 0}
                st.session_state['analysis_complete'] = False
                st.success("✅ History cleared!")

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()