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
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
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
import math

warnings.filterwarnings('ignore')

# -- 1. BYPASS DLL CONFLICTS --
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -- 2. CONFIGURATION --
GMAIL_APP_PASSWORD = "twmlrauqerkvxark"
ALERT_EMAIL = "emmanuelchiutsi001@gmail.com"
DATASET_PATH = r"C:\Users\emmanuel chiutsi\Documents\security dataset"

# Define all crime categories
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

# -- 3. PAGE CONFIG --
st.set_page_config(
    page_title="AI COMMUNITY SECURITY ANALYTICS",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -- 4. CUSTOM CSS FOR BACKGROUND --
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


# -- 5. VIDEO HANDLING FUNCTIONS --
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
            except Exception:
                continue
    except Exception as e:
        st.error(f"Error accessing dataset: {e}")

    return all_files


def check_video_file(video_path):
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


# -- 6. POSITIONAL ENCODING FOR TRANSFORMER --
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 3:
            if x.shape[0] == self.pe.shape[0]:
                x = x + self.pe[:x.shape[0]]
            else:
                x = x + self.pe[:x.shape[1]].transpose(0, 1)
        return self.dropout(x)


# -- 7. FROZEN BACKBONE + TRAINABLE TEMPORAL ATTENTION --
class MobileNetV3TemporalAttention(nn.Module):
    """
    Strategy: "Extract then Attend"
    - MobileNetV3 is FROZEN (used only as feature extractor)
    - Only the Temporal Attention layer is trained
    - This prevents overfitting on small datasets
    """

    def __init__(self, num_classes=11, d_model=512, nhead=8, num_encoder_layers=2,
                 dim_feedforward=1024, dropout=0.2, max_seq_len=16):
        super(MobileNetV3TemporalAttention, self).__init__()

        # Load pretrained MobileNetV3-Large
        self.mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # Get feature dimension
        self.feature_dim = self.mobilenet.classifier[0].in_features

        # Remove classification head - we only need features
        self.mobilenet.classifier = nn.Identity()

        # FREEZE the entire MobileNetV3 backbone
        # This implements the "Extract" strategy - backbone is frozen
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # Projection layer to map features to d_model (trainable)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding (frozen)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder for temporal attention (TRAINABLE)
        # This is the "Attend" part - only this gets trained
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Learnable CLS token (trainable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Attention pooling (trainable)
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        # Classification head (trainable)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, num_classes)
        )

        # Initialize trainable weights
        self._init_weights()

    def _init_weights(self):
        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.cls_token, std=0.02)

    def extract_features(self, frames_batch):
        """Extract features using FROZEN MobileNetV3"""
        batch_size, seq_len, C, H, W = frames_batch.shape
        frames_flat = frames_batch.view(batch_size * seq_len, C, H, W)

        # No gradients for backbone
        with torch.no_grad():
            features = self.mobilenet(frames_flat)

        features = features.view(batch_size, seq_len, self.feature_dim)
        features = self.feature_projection(features)  # This is trainable
        return features

    def forward(self, frames_batch):
        """
        Forward pass: Extract features (frozen) then Attend (trainable)
        frames_batch: (batch_size, sequence_length, 3, 224, 224)
        Returns: (output, attn_weights) - 2 values
        """
        # Step 1: EXTRACT - Frozen backbone feature extraction
        spatial_features = self.extract_features(frames_batch)

        # Step 2: Add positional encoding
        spatial_features = self.pos_encoder(spatial_features)

        # Step 3: Add CLS token
        batch_size = spatial_features.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence_with_cls = torch.cat([cls_tokens, spatial_features], dim=1)

        # Step 4: ATTEND - Transformer encoder (TRAINABLE)
        transformer_output = self.transformer_encoder(sequence_with_cls)

        # Step 5: Attention pooling
        cls_output = transformer_output[:, 0, :]
        kv = transformer_output[:, 1:, :]
        attn_output, attn_weights = self.attention_pool(
            query=cls_output.unsqueeze(1),
            key=kv,
            value=kv
        )

        # Step 6: Combine and classify
        combined = cls_output + attn_output.squeeze(1)
        combined = combined / 2
        output = self.classifier(combined)

        return output, attn_weights

    def get_trainable_params_count(self):
        """Count trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# -- 8. VIDEO DATASET WITH 16-FRAME SAMPLING --
class SecurityVideoDataset(Dataset):
    """Dataset that extracts exactly 16 frames from each video"""

    def __init__(self, video_paths, labels, transform=None, sequence_length=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.extract_frame_sequence(video_path)

        if self.transform and len(frames) > 0:
            frames = [self.transform(frame) for frame in frames]

        # Pad if not enough frames
        if len(frames) < self.sequence_length:
            pad_count = self.sequence_length - len(frames)
            if len(frames) > 0:
                frames = frames + [frames[-1].clone()] * pad_count
            else:
                frames = [torch.zeros(3, 224, 224)] * self.sequence_length

        # Stack into tensor
        sequence = torch.stack(frames[:self.sequence_length])
        return sequence, label

    def extract_frame_sequence(self, video_path):
        """Extract exactly 16 evenly spaced frames from the video"""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return frames

        # Calculate optimal step to get exactly sequence_length frames
        if total_frames <= self.sequence_length:
            # If video is short, take all frames and repeat if needed
            step = 1
            num_frames_to_take = total_frames
        else:
            step = total_frames // self.sequence_length
            num_frames_to_take = self.sequence_length

        frame_indices = []
        for i in range(num_frames_to_take):
            idx = min(i * step, total_frames - 1)
            frame_indices.append(idx)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()

        # If we need more frames (video shorter than sequence length), repeat last frame
        while len(frames) < self.sequence_length and frames:
            frames.append(frames[-1])

        return frames[:self.sequence_length]


# -- 9. ENHANCED TRAINER WITH FROZEN BACKBONE + LEARNABLE ATTENTION --
class CrimeTransformerTrainer:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = CRIME_CATEGORIES
        self.trained = False
        self.performance_metrics = {}
        self.training_info = {}
        self.inverse_class_mapping = {0: 0}  # Default fallback
        self.effective_class_names = ['normal']
        self.class_mapping = {}

    def prepare_dataset(self, dataset_path):
        """Prepare dataset from folder structure"""
        video_paths = []
        labels = []
        class_counts = {}

        # Get all classes that actually have videos
        existing_classes = []
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                videos = safe_get_video_files(class_path)
                if videos:
                    for video in videos:
                        video_paths.append(video)
                        labels.append(class_idx)
                    class_counts[class_name] = len(videos)
                    existing_classes.append(class_name)

        return video_paths, labels, class_counts, existing_classes

    def train_temporal_attention(self, dataset_path, progress_callback=None):
        """
        Train ONLY the Temporal Attention layer with very small learning rate.
        Backbone (MobileNetV3) is FROZEN.
        """
        try:
            if progress_callback:
                progress_callback(0.05, "🔄 Preparing dataset for Temporal Attention training...")

            video_paths, labels, class_counts, existing_classes = self.prepare_dataset(dataset_path)

            if len(video_paths) == 0:
                if progress_callback:
                    progress_callback(1.0, "⚠️ No videos found for training. Using enhanced detection features.")
                self.trained = False
                return False

            num_classes_with_data = len(existing_classes)

            if progress_callback:
                progress_callback(0.1,
                                  f"📊 Found {len(video_paths)} videos across {num_classes_with_data} categories...")

            # Balance dataset - take up to 15 samples per class for efficient training
            samples_per_class = min(15, min([count for count in class_counts.values() if count > 0]))

            balanced_paths = []
            balanced_labels = []
            used_classes = []

            for class_idx, class_name in enumerate(self.class_names):
                class_videos = [video_paths[i] for i in range(len(video_paths)) if labels[i] == class_idx]
                if len(class_videos) > 0:
                    sampled = random.sample(class_videos, min(samples_per_class, len(class_videos)))
                    balanced_paths.extend(sampled)
                    balanced_labels.extend([class_idx] * len(sampled))
                    if len(sampled) > 0:
                        used_classes.append(class_idx)

            # Filter class_names to only those with data
            effective_class_names = [self.class_names[i] for i in used_classes]
            effective_num_classes = len(effective_class_names)

            # Remap labels to consecutive indices
            class_to_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(used_classes)}
            balanced_labels_remapped = [class_to_idx[label] for label in balanced_labels]

            if progress_callback:
                progress_callback(0.2,
                                  f"🎯 Using {len(balanced_paths)} balanced samples for training (classes: {len(effective_class_names)})...")

            # Simple transform (no heavy augmentation to avoid overfitting on small dataset)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Create dataset with 16-frame sequences
            dataset = SecurityVideoDataset(
                balanced_paths,
                balanced_labels_remapped,
                transform,
                sequence_length=16
            )

            # Split into train and validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

            if progress_callback:
                progress_callback(0.3, "🧠 Building MobileNetV3 + Temporal Attention model...")
                progress_callback(0.3, "🔒 Backbone (MobileNetV3) is FROZEN - only temporal attention layer trains")

            # Initialize model
            self.model = MobileNetV3TemporalAttention(
                num_classes=effective_num_classes,  # Use only classes that have data
                d_model=512,
                nhead=8,
                num_encoder_layers=2,  # Reduced for small dataset
                dim_feedforward=1024,
                dropout=0.2,
                max_seq_len=17
            )
            self.model = self.model.to(self.device)

            # Count trainable parameters
            trainable, total = self.model.get_trainable_params_count()
            if progress_callback:
                progress_callback(0.3,
                                  f"📊 Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.1f}% trainable)")

            # Use standard CrossEntropyLoss (no class weights for now)
            criterion = nn.CrossEntropyLoss()

            # VERY SMALL LEARNING RATE for transformers on small datasets (0.00005)
            # Only optimize parameters that require gradients (temporal attention layers)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=0.00005,  # Very small learning rate
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )

            # Simple StepLR scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

            if progress_callback:
                progress_callback(0.35, "🎬 Training Temporal Attention on 16-frame sequences...")
                progress_callback(0.35, f"💡 Learning Rate: 0.00005 (very small for transformer)")

            # Training loop - fewer epochs for small dataset
            self.model.train()
            num_epochs = 15  # Moderate epochs

            best_val_acc = 0
            patience_counter = 0
            best_model_state = None

            for epoch in range(num_epochs):
                running_loss = 0.0
                self.model.train()

                for i, (sequences, labels_batch) in enumerate(train_loader):
                    sequences = sequences.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs, attn_weights = self.model(sequences)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()

                    running_loss += loss.item()

                    progress = 0.35 + (epoch * len(train_loader) + i) / (num_epochs * len(train_loader)) * 0.5
                    if progress_callback and i % 5 == 0:
                        progress_callback(progress, f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}")

                scheduler.step()

                # Validation
                if progress_callback:
                    progress_callback(0.85, f"📈 Validating (Epoch {epoch + 1}/{num_epochs})...")

                self.model.eval()
                all_preds = []
                all_labels = []
                val_loss = 0.0

                with torch.no_grad():
                    for sequences, labels_batch in val_loader:
                        sequences = sequences.to(self.device)
                        labels_batch = labels_batch.to(self.device)
                        outputs, _ = self.model(sequences)
                        loss = criterion(outputs, labels_batch)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels_batch.cpu().numpy())

                val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model state
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if progress_callback:
                    progress_callback(0.85, f"📊 Epoch {epoch + 1}: Val Acc = {val_acc:.2%} | Best = {best_val_acc:.2%}")

                if patience_counter >= 5:
                    if progress_callback:
                        progress_callback(0.85, "⏹️ Early stopping triggered")
                    break

            # Load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            # Evaluate on validation set
            if progress_callback:
                progress_callback(0.92, "📊 Evaluating model performance...")

            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for sequences, labels_batch in val_loader:
                    sequences = sequences.to(self.device)
                    outputs, _ = self.model(sequences)
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
                per_class_acc = {}
                cm = confusion_matrix(all_labels, all_preds)
                for i, class_name in enumerate(effective_class_names):
                    if i < len(cm):
                        tp = cm[i, i]
                        total_class = cm[i, :].sum()
                        per_class_acc[class_name] = float(tp / total_class * 100) if total_class > 0 else 0

                self.performance_metrics['per_class_accuracy'] = per_class_acc

            self.training_info = {
                'class_counts': class_counts,
                'effective_classes': effective_class_names,
                'num_classes': effective_num_classes,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'sequence_length': 16,
                'num_epochs': num_epochs,
                'best_val_accuracy': best_val_acc * 100,
                'trainable_params': trainable,
                'total_params': total,
                'learning_rate': 0.00005,
                'model_architecture': 'MobileNetV3 (Frozen) + Temporal Attention (Trainable)'
            }

            # Store class mapping for inference
            self.class_mapping = class_to_idx
            self.inverse_class_mapping = {v: k for k, v in class_to_idx.items()}
            self.effective_class_names = effective_class_names

            self.trained = True

            if progress_callback:
                progress_callback(1.0, "✅ Training complete! Temporal Attention layer trained successfully.")
                progress_callback(1.0, f"🎯 Validation Accuracy: {self.performance_metrics.get('accuracy', 0):.1f}%")

            return True

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.trained = False
            return False

    def predict_sequence(self, frame_sequence):
        """Predict crime type from a sequence of frames - FIXED VERSION"""
        if not self.trained or self.model is None:
            return 0, 0.0

        try:
            self.model.eval()
            with torch.no_grad():
                if isinstance(frame_sequence, torch.Tensor):
                    if frame_sequence.dim() == 3:
                        frame_sequence = frame_sequence.unsqueeze(0)
                    input_tensor = frame_sequence.to(self.device)
                else:
                    return 0, 0.5

                # Model returns (output, attn_weights) - 2 values, NOT 5
                outputs, _ = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                confidence = probabilities[0][predicted].item()

                # Map back to original class index
                if hasattr(self, 'inverse_class_mapping') and self.inverse_class_mapping:
                    original_class = self.inverse_class_mapping.get(predicted.item(), 0)
                else:
                    original_class = predicted.item()

                return original_class, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.0


# -- 10. CRIME DETECTION MODEL --
class CrimeDetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = CrimeTransformerTrainer()
        self.model_loaded = False
        self.training_status = "Not Trained"
        self.frame_buffer = deque(maxlen=16)

        # Check if dataset exists and has videos
        try:
            normal_path = os.path.join(DATASET_PATH, "normal")
            crime_paths = [os.path.join(DATASET_PATH, cat) for cat in CRIME_CATEGORIES if cat != 'normal']

            has_normal = os.path.exists(normal_path)
            has_crime = any(os.path.exists(path) for path in crime_paths)

            normal_videos = safe_get_video_files(normal_path) if has_normal else []
            crime_videos = []
            for path in crime_paths:
                if os.path.exists(path):
                    crime_videos.extend(safe_get_video_files(path))

            if has_normal and len(normal_videos) > 0 and len(crime_videos) > 0:
                with st.spinner("🎯 Training Temporal Attention Layer (Backbone Frozen)..."):
                    progress_bar = st.progress(0)

                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        if progress < 1.0:
                            st.caption(message)

                    success = self.trainer.train_temporal_attention(DATASET_PATH, update_progress)
                    if success:
                        self.model_loaded = True
                        self.training_status = "Temporal Attention Trained"
                        st.success(
                            "✅ Temporal Attention training complete! Backbone is frozen, only attention layer learns temporal patterns.")
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

        # Preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"🔥 Crime Detection System (Frozen Backbone + Temporal Attention) initialized on {self.device}")
        print(f"Training Status: {self.training_status}")

    def get_performance_metrics(self):
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
        return self.trainer.training_info

    def add_frame_to_buffer(self, frame):
        processed = self.preprocess(frame)
        self.frame_buffer.append(processed)
        return len(self.frame_buffer) == 16

    def predict_from_buffer(self):
        if len(self.frame_buffer) < 12:
            return 0, 0.0
        sequence = torch.stack(list(self.frame_buffer))
        return self.trainer.predict_sequence(sequence)


# -- 11. ADVANCED VIDEO ANALYZER --
class AdvancedCrimeAnalyzer:
    def __init__(self, model):
        self.model = model
        self.device = model.device if hasattr(model, 'device') else 'cpu'
        self.analysis_history = deque(maxlen=100)
        self.detection_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        self.frame_sequence_buffer = []

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

        return {
            'accuracy': round(accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1': round(f1, 2),
            'samples': total
        }

    def compute_optical_flow_features(self, prev_frame, curr_frame):
        if prev_frame is None or curr_frame is None:
            return {'motion_magnitude': 0, 'motion_variance': 0, 'sudden_change': 0}

        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            magnitude_std = np.std(magnitude) if magnitude.size > 0 else 0

            sudden_change = magnitude_std / (magnitude_mean + 1e-6)

            return {
                'motion_magnitude': float(magnitude_mean),
                'motion_variance': float(magnitude_std),
                'sudden_change': float(sudden_change)
            }
        except Exception:
            return {'motion_magnitude': 0, 'motion_variance': 0, 'sudden_change': 0}

    def detect_crime_indicators(self, prev_frame, curr_frame):
        """Returns dictionary with 5 crime indicators"""
        indicators = {}
        motion = self.compute_optical_flow_features(prev_frame, curr_frame)

        indicators['robbery'] = min(motion['sudden_change'] * 45 + motion['motion_magnitude'] * 2, 100)
        indicators['theft'] = min((motion['motion_variance'] * 35) / (motion['motion_magnitude'] + 1e-6), 100)
        indicators['assault'] = min((motion['motion_magnitude'] * 2.5) + (motion['motion_variance'] * 3.5), 100)
        indicators['fighting'] = min((motion['motion_variance'] * 6) / (motion['motion_magnitude'] + 1e-6) * 60, 100)
        indicators['weapon'] = 0  # Added to ensure 5 keys

        return indicators

    def detect_weapons(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_metal = np.array([0, 0, 200])
            upper_metal = np.array([180, 50, 255])
            metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            elongated_objects = 0
            for contour in contours:
                if len(contour) > 20:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                    if aspect_ratio > 3:
                        elongated_objects += 1

            metal_ratio = np.sum(metal_mask > 0) / (metal_mask.size + 1e-6)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
            weapon_shape_score = min(elongated_objects * 5, 30)

            weapon_score = min((metal_ratio * 40 + edge_density * 25 + weapon_shape_score), 100)
            return float(weapon_score)
        except:
            return float(np.random.randint(3, 12))

    def analyze_video_crime(self, video_path, progress_bar=None, actual_label=None):
        is_valid, message = check_video_file(video_path)
        if not is_valid:
            return {}, f"Video error: {message}"

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        crime_scores = defaultdict(list)
        prev_frame = None
        frame_count = 0
        crime_events = []
        self.frame_sequence_buffer = []
        model_predictions = []
        prediction_confidences = []

        if total_frames < 100:
            sample_rate = 1
        elif total_frames < 300:
            sample_rate = 2
        else:
            sample_rate = max(1, int(total_frames / 90))

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

                if prev_frame is not None:
                    indicators = self.detect_crime_indicators(prev_frame, frame)
                    for key, value in indicators.items():
                        crime_scores[key].append(value)

                weapon_score = self.detect_weapons(frame)
                crime_scores['weapon'].append(weapon_score)

                processed_frame = self.model.preprocess(frame)
                self.frame_sequence_buffer.append(processed_frame)

                if len(self.frame_sequence_buffer) >= 16:
                    sequence = torch.stack(self.frame_sequence_buffer[-16:])
                    pred_class, confidence = self.model.trainer.predict_sequence(sequence)
                    model_predictions.append(pred_class)
                    prediction_confidences.append(confidence)
                    self.frame_sequence_buffer = self.frame_sequence_buffer[-8:]

                if len(crime_scores.get('assault', [])) > 3:
                    recent_assault = np.mean(crime_scores['assault'][-5:]) if len(crime_scores['assault']) >= 5 else \
                    crime_scores['assault'][-1] if crime_scores['assault'] else 0
                    recent_robbery = np.mean(crime_scores.get('robbery', [0])[-5:]) if crime_scores.get('robbery') and len(crime_scores['robbery']) >= 5 else crime_scores.get('robbery', [0])[
                        -1] if crime_scores.get('robbery') else 0
                    recent_theft = np.mean(crime_scores.get('theft', [0])[-5:]) if crime_scores.get('theft') and len(
                        crime_scores['theft']) >= 5 else crime_scores.get('theft', [0])[-1] if crime_scores.get(
                        'theft') else 0
                    recent_weapon = np.mean(crime_scores.get('weapon', [0])[-5:]) if crime_scores.get('weapon') and len(
                        crime_scores['weapon']) >= 5 else crime_scores.get('weapon', [0])[-1] if crime_scores.get(
                        'weapon') else 0

                    combined_score = (
                                recent_assault * 0.35 + recent_robbery * 0.25 + recent_theft * 0.2 + recent_weapon * 0.2)

                    if model_predictions and prediction_confidences:
                        recent_pred = model_predictions[-1]
                        recent_conf = prediction_confidences[-1]
                        if recent_pred > 0 and recent_conf > 0.6:
                            combined_score *= (1 + recent_conf * 0.6)

                    crime_type = self.determine_crime_type(crime_scores,
                                                           model_predictions[-1] if model_predictions else 0)
                    event_threshold = 30 if is_normal_video else 35

                    if combined_score > event_threshold:
                        crime_events.append({
                            'frame': frame_idx,
                            'time': frame_idx / fps if fps > 0 else 0,
                            'score': round(combined_score, 2),
                            'type': crime_type,
                            'confidence': 'HIGH' if combined_score > 75 else 'MEDIUM'
                        })

                prev_frame = frame.copy()

            except Exception as e:
                continue

        cap.release()

        metrics = self.calculate_aggregate_metrics(
            crime_scores, frame_count, total_frames, duration,
            crime_events, is_normal_video, model_predictions,
            prediction_confidences
        )

        if actual_label is not None:
            predicted_crime = metrics.get('overall_crime_score', 0) > 40
            self.update_performance_stats(predicted_crime, actual_label)

        self.analysis_history.append({
            'timestamp': datetime.now(),
            'video': os.path.basename(video_path),
            'metrics': metrics,
            'crime_events': crime_events[:20]
        })

        return metrics, "Analysis complete"

    def determine_crime_type(self, crime_scores, model_pred_class):
        if model_pred_class > 0 and model_pred_class < len(CRIME_CATEGORIES):
            model_type = CRIME_TYPE_MAP.get(CRIME_CATEGORIES[model_pred_class], 'SUSPICIOUS')
            if model_type != 'NORMAL':
                return model_type

        avg_assault = np.mean(crime_scores.get('assault', [0])) if crime_scores.get('assault') else 0
        avg_robbery = np.mean(crime_scores.get('robbery', [0])) if crime_scores.get('robbery') else 0
        avg_theft = np.mean(crime_scores.get('theft', [0])) if crime_scores.get('theft') else 0
        avg_fighting = np.mean(crime_scores.get('fighting', [0])) if crime_scores.get('fighting') else 0
        avg_weapon = np.mean(crime_scores.get('weapon', [0])) if crime_scores.get('weapon') else 0

        scores = {
            'ASSAULT/FIGHT': avg_assault * 0.6 + avg_fighting * 0.4,
            'ROBBERY': avg_robbery,
            'THEFT': avg_theft,
            'WEAPON DETECTED': avg_weapon * 1.2
        }

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        if max_score > 35:
            return max_type
        elif max(scores.values()) > 25:
            return "SUSPICIOUS"
        else:
            return "NORMAL"

    def calculate_aggregate_metrics(self, crime_scores, frame_count, total_frames,
                                    duration, crime_events, is_normal_video,
                                    model_predictions, prediction_confidences):

        avg_assault = np.mean(crime_scores.get('assault', [0])) if crime_scores.get('assault') else 0
        avg_robbery = np.mean(crime_scores.get('robbery', [0])) if crime_scores.get('robbery') else 0
        avg_theft = np.mean(crime_scores.get('theft', [0])) if crime_scores.get('theft') else 0
        avg_fighting = np.mean(crime_scores.get('fighting', [0])) if crime_scores.get('fighting') else 0
        avg_weapon = np.mean(crime_scores.get('weapon', [0])) if crime_scores.get('weapon') else 0

        peak_assault = max(crime_scores.get('assault', [0])) if crime_scores.get('assault') else 0
        peak_robbery = max(crime_scores.get('robbery', [0])) if crime_scores.get('robbery') else 0
        peak_theft = max(crime_scores.get('theft', [0])) if crime_scores.get('theft') else 0

        avg_model_conf = np.mean(prediction_confidences) if prediction_confidences else 0
        crime_prediction_rate = sum(1 for p in model_predictions if p > 0) / len(
            model_predictions) if model_predictions else 0

        if is_normal_video or crime_prediction_rate < 0.2:
            final_assault = avg_assault * 0.2 + peak_assault * 0.1
            final_robbery = avg_robbery * 0.15 + peak_robbery * 0.1
            final_theft = avg_theft * 0.2 + peak_theft * 0.1
            overall_score = min(max(final_assault, final_robbery, final_theft, avg_weapon * 0.3), 35)
        else:
            transformer_boost = 1 + (crime_prediction_rate * 0.7) + (avg_model_conf * 0.3)
            final_assault = (avg_assault * 0.3 + peak_assault * 0.4) * transformer_boost
            final_robbery = (avg_robbery * 0.3 + peak_robbery * 0.4) * transformer_boost
            final_theft = (avg_theft * 0.4 + peak_theft * 0.3) * transformer_boost
            overall_score = min(max(final_assault, final_robbery, final_theft, avg_weapon * 0.8), 100)

        event_counts = {'ROBBERY': 0, 'ASSAULT/FIGHT': 0, 'THEFT': 0, 'WEAPON DETECTED': 0, 'SUSPICIOUS': 0}
        for event in crime_events:
            event_type = event['type']
            if event_type in event_counts:
                event_counts[event_type] += 1
            else:
                event_counts['SUSPICIOUS'] += 1

        return {
            'overall_crime_score': float(round(overall_score, 2)),
            'robbery_score': float(round(final_robbery, 2)),
            'theft_score': float(round(final_theft, 2)),
            'assault_score': float(round(final_assault, 2)),
            'fighting_score': float(round(avg_fighting, 2)),
            'weapon_detection': float(round(avg_weapon, 2)),
            'peak_robbery': float(round(peak_robbery, 2)),
            'peak_assault': float(round(peak_assault, 2)),
            'peak_theft': float(round(peak_theft, 2)),
            'frames_analyzed': frame_count,
            'total_frames': total_frames,
            'duration': float(round(duration, 2)),
            'crime_events': len(crime_events),
            'robbery_events': event_counts['ROBBERY'],
            'assault_events': event_counts['ASSAULT/FIGHT'],
            'theft_events': event_counts['THEFT'],
            'weapon_events': event_counts['WEAPON DETECTED'],
            'suspicious_events': event_counts['SUSPICIOUS'],
            'is_normal_video': is_normal_video,
            'model_trained': self.model.trainer.trained,
            'model_confidence': float(round(avg_model_conf * 100, 2)),
            'temporal_crime_rate': float(round(crime_prediction_rate * 100, 2)),
            'model_architecture': 'MobileNetV3 (Frozen) + Temporal Attention'
        }


# -- 12. VIDEO SCANNER --
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


# -- 13. INITIALIZE COMPONENTS --
@st.cache_resource
def init_components():
    crime_model = CrimeDetectionModel()
    analyzer = AdvancedCrimeAnalyzer(crime_model)
    scanner = VideoScanner()
    return crime_model, analyzer, scanner


crime_model, analyzer, scanner = init_components()


# -- 14. HELPER FUNCTIONS --
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

Model Analysis:
- Architecture: {metrics.get('model_architecture', 'MobileNetV3 + Temporal Attention')}
- Model Confidence: {metrics.get('model_confidence', 0)}%
- Temporal Crime Rate: {metrics.get('temporal_crime_rate', 0)}%

Events Detected:
- Total Events: {metrics.get('crime_events', 0)}
- Robbery Events: {metrics.get('robbery_events', 0)}
- Assault Events: {metrics.get('assault_events', 0)}
- Theft Events: {metrics.get('theft_events', 0)}
- Weapon Events: {metrics.get('weapon_events', 0)}

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
    try:
        for category in CRIME_CATEGORIES:
            category_path = os.path.join(DATASET_PATH, category)
            os.makedirs(category_path, exist_ok=True)
    except:
        pass


create_folder_structure()


def run_automatic_analysis(video_path, video_filename, email_alerts, threshold):
    is_normal = 'normal' in video_path.lower()
    actual_label = not is_normal

    with st.spinner(f"🔍 Analyzing {video_filename} with Temporal Attention model..."):
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


# -- 15. PERFORMANCE METRICS DISPLAY --
def display_performance_metrics():
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
                <b>📐 {training_info.get('model_architecture', 'MobileNetV3 + Temporal Attention')}</b><br>
                Classes trained: {training_info.get('num_classes', 0)}<br>
                Training samples: {training_info.get('train_samples', 0)}<br>
                Validation samples: {training_info.get('val_samples', 0)}<br>
                Sequence length: {training_info.get('sequence_length', 16)} frames<br>
                Validation Accuracy: {training_info.get('best_val_accuracy', 0):.1f}%<br>
                <b>Learning Rate:</b> {training_info.get('learning_rate', 0.00005)}<br>
                <b>Trainable Params:</b> {training_info.get('trainable_params', 0):,} / {training_info.get('total_params', 0):,}
            </div>
            """, unsafe_allow_html=True)

            if 'per_class_accuracy' in perf_metrics:
                with st.expander("📊 Per-Class Accuracy", expanded=False):
                    for class_name, acc in perf_metrics['per_class_accuracy'].items():
                        color = "#00ff88" if acc > 70 else "#feca57" if acc > 40 else "#ff4757"
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                            <span>{class_name.upper()}:</span>
                            <span style="color: {color}; font-weight: bold;">{acc:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info(
            "🤖 Temporal Attention model not trained yet. Add videos to 'normal' and crime folders, then click 'Refresh & Retrain' in Settings.")

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


# -- 16. MAIN STREAMLIT APP --
def main():
    set_background()

    st.markdown("""
    <div class="main-header">
        <h1>🚨 AI COMMUNITY SECURITY ANALYTICS</h1>
        <p style="color: #00fbff; font-size: 1.2em;">Frozen Backbone + Temporal Attention | LR=0.00005</p>
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

        model_status = "✅ Temporal Attention" if crime_model.trainer.trained else "⚠️ Not Trained"
        st.markdown(f"**Model:** {model_status}")
        st.markdown(f"**Status:** {crime_model.training_status}")
        st.markdown(f"**Device:** {crime_model.device}")

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
                    st.warning("📂 No videos found in dataset")

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
                        <b>AI Model:</b> {metrics.get('model_architecture', 'MobileNetV3 + Temporal Attention')}<br>
                        <b>Model Confidence:</b> {metrics.get('model_confidence', 0)}%<br>
                        <b>Temporal Crime Rate:</b> {metrics.get('temporal_crime_rate', 0)}%
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
                        categories = ['Robbery', 'Theft', 'Assault', 'Fighting', 'Weapons']
                        values = [
                            metrics.get('robbery_score', 0),
                            metrics.get('theft_score', 0),
                            metrics.get('assault_score', 0),
                            metrics.get('fighting_score', 0),
                            metrics.get('weapon_detection', 0)
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
                    <p style="color: #00fbff; text-align: center;">Backbone is FROZEN - Only Temporal Attention learns from your videos!</p>
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
                    'Events': entry['metrics']['crime_events'],
                    'Model Conf': f"{entry['metrics'].get('model_confidence', 0)}%"
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
                    title="Historical Crime Scores (Temporal Attention Analysis)",
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
                with st.spinner("Retraining Temporal Attention layer (Backbone Frozen)..."):
                    st.session_state['last_selected_video'] = None

                    def update_progress(progress, message):
                        st.caption(message)

                    success = crime_model.trainer.train_temporal_attention(DATASET_PATH, update_progress)
                    if success:
                        crime_model.model_loaded = True
                        crime_model.training_status = "Temporal Attention Trained"
                        st.success("✅ Temporal Attention layer retrained successfully!")
                    else:
                        st.warning("⚠️ Training failed. Please add videos to both 'normal' and crime folders.")

                    videos = scanner.get_all_videos(DATASET_PATH, force_refresh=True)
                    st.rerun()

        st.markdown("#### 🤖 Model Architecture: Frozen Backbone + Temporal Attention")
        st.info("""
        **Strategy: "Extract then Attend"**

        **Step 1 - EXTRACT (FROZEN):**
        - MobileNetV3-Large backbone is completely frozen
        - It acts only as a feature extractor
        - No training occurs on the backbone (prevents overfitting)

        **Step 2 - ATTEND (TRAINABLE):**
        - Only the Temporal Attention layer is trained
        - Transformer Encoder learns relationships between frames
        - Attention mechanism focuses on critical moments in the video

        **Key Benefits for Small Datasets:**
        - Very small learning rate: **0.00005**
        - Only ~2-5% of parameters are trainable
        - Prevents overfitting on limited data
        - Fast training (even on CPU)

        **How to Improve Accuracy:**
        1. Add 15-20 videos per crime category
        2. Ensure videos are 30-60 seconds long
        3. Include varied scenarios (different angles, lighting)
        4. Balance normal vs crime videos equally
        """)

        st.markdown("#### 📊 Model Status")
        st.info(
            f"Training Status: {'✅ Temporal Attention Trained' if crime_model.trainer.trained else '⚠️ Not Trained'}")
        st.info(f"Status: {crime_model.training_status}")
        st.info(f"Device: {crime_model.device}")

        if crime_model.trainer.trained:
            trainable, total = crime_model.trainer.model.get_trainable_params_count() if crime_model.trainer.model else (
            0, 0)
            st.success(f"🎯 Validation Accuracy: {crime_model.get_performance_metrics().get('accuracy', 0):.1f}%")
            st.info(f"📊 Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.1f}% trainable)")
            st.info(f"💡 Learning Rate: 0.00005")

        if st.button("🗑️ Clear Analysis History", key="clear"):
            analyzer.analysis_history.clear()
            analyzer.detection_stats = {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0,
                                        'false_negatives': 0}
            st.session_state['analysis_complete'] = False
            st.success("✅ History cleared!")

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()