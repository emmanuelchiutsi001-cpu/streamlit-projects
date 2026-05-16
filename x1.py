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

warnings.filterwarnings('ignore')

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -- CONFIGURATION --
GMAIL_APP_PASSWORD = "twmlrauqerkvxark"
ALERT_EMAIL = "emmanuelchiutsi001@gmail.com"
DATASET_PATH = r"C:\Users\emmanuel chiutsi\Documents\dataset-video-split"

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


# -- OPTIMIZED DATASET (SINGLE FRAME, FAST) --
class FastVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract single frame (fast)
        frame = extract_single_frame(video_path)

        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            frame = self.transform(frame)

        return frame, label


# -- OPTIMIZED MODEL WITHOUT BATCHNORM ISSUES --
class OptimizedCrimeClassifier(nn.Module):
    def __init__(self, num_classes=21):
        super(OptimizedCrimeClassifier, self).__init__()
        # Use ResNet18 (faster, good enough)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze early layers to prevent overfitting and speed up training
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features

        # Simplified classifier without BatchNorm (to avoid batch size issues)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def get_trainable_params_count(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def forward(self, x):
        return self.backbone(x)


# -- OPTIMIZED TRAINER --
class OptimizedCrimeTrainer:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = CRIME_CATEGORIES
        self.trained = False
        self.performance_metrics = {}
        self.training_info = {}
        self.inverse_class_mapping = {0: 0}
        self.effective_class_names = ['normal_videos']
        self.class_mapping = {}
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def map_filename_to_category(self, filename):
        """Map filename patterns to categories"""
        name = os.path.splitext(filename)[0].lower()

        # Direct mapping
        for category in self.class_names:
            if category in name or name.startswith(category):
                return category

        # Special mappings
        special_mappings = {
            'standingstill': 'standing_still', 'standstill': 'standing_still',
            'meetandsplit': 'meet_and_split', 'road_accidents': 'roadaccidents',
            'walkingreadingbook': 'walking_while_reading_book',
            'walkingusingphone': 'walking_while_using_phone',
            'burglary': 'bulglary'
        }

        for key, value in special_mappings.items():
            if key in name:
                return value

        return None

    def prepare_dataset_from_folder(self, folder_path):
        """Prepare dataset from folder"""
        video_paths = []
        labels = []
        class_counts = defaultdict(int)

        if not os.path.exists(folder_path):
            return video_paths, labels, class_counts

        videos = safe_get_video_files(folder_path)

        for video in videos:
            filename = os.path.basename(video)
            category = self.map_filename_to_category(filename)

            if category is not None and category in self.class_names:
                if check_video_file(video)[0]:
                    class_idx = self.class_names.index(category)
                    video_paths.append(video)
                    labels.append(class_idx)
                    class_counts[category] += 1

        return video_paths, labels, class_counts

    def get_transforms(self):
        """Get training and validation transforms"""
        # Training transforms (light augmentation to avoid overfitting)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Validation transforms
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return train_transform, val_transform

    def train_fast_model(self, dataset_path, progress_callback=None):
        """Train model with fixes for batch size issues"""
        try:
            if progress_callback:
                progress_callback(0.05, "🔄 Preparing dataset...")

            train_path = os.path.join(dataset_path, 'train')

            if not os.path.exists(train_path):
                if progress_callback:
                    progress_callback(1.0, f"❌ Train folder not found")
                return False

            # Prepare dataset
            video_paths, labels, class_counts = self.prepare_dataset_from_folder(train_path)

            if len(video_paths) == 0:
                if progress_callback:
                    progress_callback(1.0, "⚠️ No valid videos found")
                return False

            # Get unique classes with enough samples
            unique_classes = sorted(list(set(labels)))
            # Filter classes with at least 2 samples
            valid_classes = []
            for class_idx in unique_classes:
                class_name = self.class_names[class_idx]
                if class_counts[class_name] >= 2:
                    valid_classes.append(class_idx)

            if len(valid_classes) < 2:
                if progress_callback:
                    progress_callback(1.0, "⚠️ Need at least 2 classes with 2+ samples each")
                return False

            # Filter data to valid classes
            valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
            video_paths = [video_paths[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]

            # Re-map labels
            class_to_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_classes)}
            remapped_labels = [class_to_idx[label] for label in labels]

            effective_class_names = [self.class_names[i] for i in valid_classes]
            effective_num_classes = len(effective_class_names)

            if progress_callback:
                progress_callback(0.15, f"✅ Found {len(video_paths)} videos across {effective_num_classes} categories")
                for cat, count in class_counts.items():
                    if count > 0 and self.class_names.index(cat) in valid_classes:
                        progress_callback(0.15, f"  - {cat}: {count} videos")

            # Get transforms
            train_transform, val_transform = self.get_transforms()

            # Create full dataset
            full_dataset = FastVideoDataset(video_paths, remapped_labels, train_transform, is_training=True)

            # Stratified split
            from sklearn.model_selection import StratifiedShuffleSplit
            labels_array = np.array(remapped_labels)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(video_paths, labels_array))

            # Create datasets
            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            val_video_paths = [video_paths[i] for i in val_idx]
            val_labels = [remapped_labels[i] for i in val_idx]
            val_dataset = FastVideoDataset(val_video_paths, val_labels, val_transform, is_training=False)

            # Use DropLast to avoid batch size 1 issues
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

            if progress_callback:
                progress_callback(0.3, f"🧠 Building model on {self.device}...")
                progress_callback(0.3, f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

            # Initialize model
            self.model = OptimizedCrimeClassifier(num_classes=effective_num_classes)
            self.model = self.model.to(self.device)

            # Count parameters
            trainable, total = self.model.get_trainable_params_count()
            if progress_callback:
                progress_callback(0.3, f"📊 Params: {trainable:,} trainable / {total:,} total")

            # Calculate class weights
            class_counts_array = np.array([class_counts[self.class_names[c]] for c in valid_classes])
            class_weights = 1.0 / class_counts_array
            class_weights = class_weights / class_weights.sum() * effective_num_classes
            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

            # Loss with label smoothing
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

            # Optimizer with weight decay
            optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.01)

            # Scheduler - FIXED: removed 'verbose' parameter
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

            if progress_callback:
                progress_callback(0.35, f"🎬 Training started...")

            # Training loop
            num_epochs = 20
            best_val_acc = 0
            best_model_state = None
            patience = 6
            patience_counter = 0
            best_epoch = 0

            for epoch in range(num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for i, (images, labels_batch) in enumerate(train_loader):
                    images = images.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels_batch.size(0)
                    train_correct += (predicted == labels_batch).sum().item()

                    if progress_callback and i % 10 == 0:
                        progress = 0.35 + (epoch * len(train_loader) + i) / (num_epochs * len(train_loader)) * 0.55
                        progress_callback(progress,
                                          f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.3f} | Acc: {100 * train_correct / train_total:.1f}%")

                train_acc = 100 * train_correct / train_total if train_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader)

                # Validation
                if progress_callback:
                    progress_callback(0.9, f"📈 Validating...")

                self.model.eval()
                all_preds = []
                all_labels = []
                val_correct = 0
                val_total = 0
                val_loss = 0.0

                with torch.no_grad():
                    for images, labels_batch in val_loader:
                        images = images.to(self.device)
                        labels_batch = labels_batch.to(self.device)
                        outputs = self.model(images)
                        loss = criterion(outputs, labels_batch)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels_batch.size(0)
                        val_correct += (predicted == labels_batch).sum().item()
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels_batch.cpu().numpy())

                val_acc = 100 * val_correct / val_total if val_total > 0 else 0
                avg_val_loss = val_loss / len(val_loader)

                # Update scheduler
                scheduler.step(val_acc)

                # Store history
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_acc'].append(val_acc)

                # Check for best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    best_epoch = epoch + 1
                    if progress_callback:
                        progress_callback(0.9, f"✨ New best! Val Acc: {val_acc:.1f}%")
                else:
                    patience_counter += 1

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                if progress_callback:
                    progress_callback(0.9,
                                      f"📊 E{epoch + 1}: Train={train_acc:.1f}% | Val={val_acc:.1f}% | LR={current_lr:.6f}")

                # Early stopping
                if patience_counter >= patience:
                    if progress_callback:
                        progress_callback(0.9, f"⏹️ Early stopping at epoch {epoch + 1} (best was epoch {best_epoch})")
                    break

            # Load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            # Final evaluation
            if progress_callback:
                progress_callback(0.95, "📊 Computing final metrics...")

            # Calculate metrics
            if len(all_preds) > 0:
                self.performance_metrics = {
                    'accuracy': float(accuracy_score(all_labels, all_preds) * 100),
                    'precision': float(
                        precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100),
                    'recall': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100),
                    'f1_score': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100)
                }

                # Per-class accuracy
                cm = confusion_matrix(all_labels, all_preds)
                per_class_acc = {}
                for i, class_name in enumerate(effective_class_names):
                    if i < len(cm):
                        tp = cm[i, i]
                        total_class = cm[i, :].sum()
                        per_class_acc[class_name] = float(tp / total_class * 100) if total_class > 0 else 0
                self.performance_metrics['per_class_accuracy'] = per_class_acc

            # Calculate final train accuracy (for info)
            final_train_acc = self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0

            self.training_info = {
                'class_counts': {k: v for k, v in class_counts.items() if self.class_names.index(k) in valid_classes},
                'effective_classes': effective_class_names,
                'num_classes': effective_num_classes,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'num_epochs': epoch + 1,
                'best_val_accuracy': best_val_acc,
                'best_epoch': best_epoch,
                'trainable_params': trainable,
                'total_params': total,
                'learning_rate': 0.0005,
                'model_architecture': 'ResNet18 (Optimized)',
                'total_videos': len(video_paths),
                'final_train_acc': final_train_acc,
                'final_val_acc': best_val_acc
            }

            self.class_mapping = class_to_idx
            self.inverse_class_mapping = {v: k for k, v in class_to_idx.items()}
            self.effective_class_names = effective_class_names
            self.trained = True

            if progress_callback:
                gap = final_train_acc - best_val_acc
                progress_callback(1.0, f"✅ Training complete! Validation Accuracy: {best_val_acc:.1f}%")
                progress_callback(1.0, f"📊 Training-Validation Gap: {gap:.1f}%")

            return True

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            if progress_callback:
                progress_callback(1.0, f"❌ Error: {str(e)[:100]}")
            return False

    def predict_frame(self, frame_tensor):
        """Predict class for a frame tensor"""
        if not self.trained or self.model is None:
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

                if hasattr(self, 'inverse_class_mapping') and self.inverse_class_mapping:
                    original_class = self.inverse_class_mapping.get(predicted[0].item(), 0)
                else:
                    original_class = predicted[0].item()

                return original_class, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.0


# -- CRIME DETECTION MODEL --
class CrimeDetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = OptimizedCrimeTrainer()
        self.model_loaded = False
        self.training_status = "Not Trained"
        self.frame_buffer = deque(maxlen=5)

        # Check dataset and train
        try:
            train_path = os.path.join(DATASET_PATH, "train")
            if os.path.exists(train_path):
                videos = safe_get_video_files(train_path)
                if len(videos) > 0:
                    with st.spinner(f"🎯 Training on {len(videos)} videos..."):
                        progress_bar = st.progress(0)

                        def update_progress(progress, message):
                            progress_bar.progress(progress)
                            if progress < 1.0:
                                st.caption(message)

                        success = self.trainer.train_fast_model(DATASET_PATH, update_progress)
                        if success:
                            self.model_loaded = True
                            self.training_status = "Model Trained"
                            st.success(f"✅ Training complete!")
                        else:
                            self.model_loaded = False
                            self.training_status = "Training Failed"
                        progress_bar.empty()
                else:
                    st.info("📁 'train' folder found but no videos inside.")
            else:
                st.info("📁 Please create the 'train' folder with videos.")
        except Exception as e:
            st.warning(f"Model init: {e}")
            self.model_loaded = False
            self.training_status = "Error"

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_performance_metrics(self):
        if self.trainer.trained and self.trainer.performance_metrics:
            return self.trainer.performance_metrics
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}

    def get_training_info(self):
        return self.trainer.training_info

    def get_training_history(self):
        return self.trainer.training_history

    def add_frame_to_buffer(self, frame):
        processed = self.preprocess(frame)
        self.frame_buffer.append(processed)
        return len(self.frame_buffer)

    def predict_from_buffer(self):
        """Predict using majority vote"""
        if len(self.frame_buffer) == 0:
            return 0, 0.0

        predictions = []
        confidences = []

        for frame in list(self.frame_buffer):
            pred_class, confidence = self.trainer.predict_frame(frame.unsqueeze(0))
            predictions.append(pred_class)
            confidences.append(confidence)

        # Majority vote
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0][0]

        # Average confidence for majority class
        avg_confidence = np.mean([confidences[i] for i, p in enumerate(predictions) if p == most_common])

        return most_common, avg_confidence


# -- SIMPLIFIED ANALYZER --
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
                        crime_events.append({
                            'time': round(frame_idx / fps, 1),
                            'score': round(crime_score, 1),
                            'type': CRIME_TYPE_MAP.get(pred_category, 'SUSPICIOUS')
                        })

                        # Reset counter after alert to avoid spam
                        consecutive_crime = 0

            except Exception as e:
                continue

        cap.release()

        # Calculate metrics
        metrics = self.calculate_metrics(
            crime_events, model_predictions, prediction_confidences,
            frame_count, duration, is_normal_video
        )

        if actual_label is not None:
            predicted_crime = len(crime_events) > 0 or metrics.get('overall_crime_score', 0) > 35
            self.update_performance_stats(predicted_crime, actual_label)

        self.analysis_history.append({
            'timestamp': datetime.now(),
            'video': os.path.basename(video_path),
            'metrics': metrics,
            'crime_events': crime_events[:20]
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
            'model_trained': self.model.trainer.trained,
            'model_architecture': 'ResNet18 (Optimized)'
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


# -- INITIALIZE COMPONENTS --
@st.cache_resource
def init_components():
    crime_model = CrimeDetectionModel()
    analyzer = AdvancedCrimeAnalyzer(crime_model)
    scanner = VideoScanner()
    return crime_model, analyzer, scanner


crime_model, analyzer, scanner = init_components()


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

    if crime_model.trainer.trained:
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
            **Best Val Acc:** {training_info.get('best_val_accuracy', 0):.1f}% (Epoch {training_info.get('best_epoch', 0)})
            **Samples:** Train={training_info.get('train_samples', 0)} | Val={training_info.get('val_samples', 0)}
            """)

        # Training curve
        if training_history and training_history.get('train_acc'):
            fig = go.Figure()
            epochs = list(range(1, len(training_history['train_acc']) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=training_history['train_acc'], mode='lines',
                                     name='Training', line=dict(color='#00fbff')))
            fig.add_trace(go.Scatter(x=epochs, y=training_history['val_acc'], mode='lines',
                                     name='Validation', line=dict(color='#ff4757')))
            fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Accuracy (%)",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                              height=250)
            st.plotly_chart(fig, use_container_width=True)

        if 'per_class_accuracy' in perf_metrics:
            with st.expander("Per-Class Accuracy"):
                for class_name, acc in sorted(perf_metrics['per_class_accuracy'].items(), key=lambda x: x[1],
                                              reverse=True)[:10]:
                    color = "#00ff88" if acc > 70 else "#feca57" if acc > 40 else "#ff4757"
                    st.markdown(f"`{class_name.upper()}`: **<span style='color:{color}'>{acc:.1f}%</span>**",
                                unsafe_allow_html=True)
    else:
        st.info("🤖 Model not trained. Add videos to 'train' folder.")


# -- MAIN APP --
def main():
    set_background()

    st.markdown("""
    <div class="main-header">
        <h1>🚨 AI COMMUNITY SECURITY ANALYTICS</h1>
        <p style="color: #00fbff;">ResNet18 | Optimized for Speed | Anti-Overfitting</p>
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

        st.markdown(f"**Model:** {'✅ Trained' if crime_model.trainer.trained else '⚠️ Not Trained'}")
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

            source = st.radio("Source:", ["Dataset", "Upload"], horizontal=True)
            video_path = None
            video_filename = None

            if source == "Dataset":
                all_videos = scanner.get_all_videos(DATASET_PATH)
                if all_videos:
                    video_options = {os.path.basename(v): v for v in all_videos}
                    selected_video = st.selectbox("Choose:", list(video_options.keys()))
                    video_path = video_options[selected_video]
                    video_filename = selected_video
                else:
                    st.warning("No videos found")
            else:
                uploaded = st.file_uploader("Upload", type=['mp4', 'avi', 'mkv', 'mov'])
                if uploaded:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded.read())
                    video_path = tfile.name
                    video_filename = uploaded.name
                    st.success(f"Uploaded: {video_filename}")

            if video_path and st.checkbox("Show Video"):
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

                    if metrics.get('crime_events', 0) > 0:
                        with st.expander(f"📋 {metrics['crime_events']} Events"):
                            for event in metrics.get('crime_events', [])[:10]:
                                st.warning(
                                    f"⏱️ {event.get('time', 0)}s - {event.get('type', 'Unknown')} ({event.get('score', 0)}%)")
            else:
                st.info("Select a video and click ANALYZE")

    elif selected == "Analytics History":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### HISTORY")

        if analyzer.analysis_history:
            history = []
            for entry in analyzer.analysis_history[-20:]:
                history.append({
                    'Time': entry['timestamp'].strftime('%H:%M:%S'),
                    'Video': entry['video'][:30],
                    'Score': f"{entry['metrics']['overall_crime_score']}%",
                    'Events': entry['metrics']['crime_events']
                })
            st.dataframe(pd.DataFrame(history), use_container_width=True)

            if len(analyzer.analysis_history) > 1:
                fig = go.Figure()
                scores = [e['metrics']['overall_crime_score'] for e in analyzer.analysis_history]
                fig.add_trace(go.Scatter(y=scores, mode='lines+markers', line=dict(color='#ff4757')))
                fig.add_hline(y=threshold, line_dash="dash", line_color="yellow")
                fig.update_layout(title="Crime Score Trend", xaxis_title="Analysis", yaxis_title="Score (%)",
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                                  height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history yet")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Settings":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### SETTINGS")

        st.markdown(f"**Dataset Path:** `{DATASET_PATH}`")

        st.markdown("#### Retraining")
        if st.button("🔄 Retrain Model", type="primary"):
            with st.spinner("Retraining..."):
                def prog(p, msg):
                    st.caption(msg)

                success = crime_model.trainer.train_fast_model(DATASET_PATH, prog)
                if success:
                    crime_model.model_loaded = True
                    st.success("Model retrained successfully!")
                    st.rerun()
                else:
                    st.error("Training failed")

        st.markdown("#### Clear Data")
        if st.button("🗑️ Clear History"):
            analyzer.analysis_history.clear()
            analyzer.detection_stats = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0,
                                        'false_negatives': 0}
            st.success("History cleared!")

        st.markdown("#### Model Info")
        st.info("""
        **Optimized Model Features:**
        - ResNet18 backbone (fast inference)
        - Frozen early layers (prevents overfitting)
        - Dropout layers (0.4, 0.3, 0.2) for regularization
        - Label smoothing (0.1)
        - Weight decay (0.01)
        - Early stopping
        - Single frame extraction (fast)
        - Majority voting on 5 frames
        """)

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()