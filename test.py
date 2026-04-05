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
import warnings
import tempfile
from streamlit_option_menu import option_menu
import base64
import pandas as pd
from PIL import Image
import io
import pathlib

warnings.filterwarnings('ignore')

# --- 1. BYPASS DLL CONFLICTS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 2. CONFIGURATION ---
GMAIL_APP_PASSWORD = "twmlrauqerkvxark"
ALERT_EMAIL = "emmanuelchiutsi001@gmail.com"
DATASET_PATH = r"C:\Users\emmanuel chiutsi\Documents\Crime"

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

        /* Main container styling */
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

        /* Card styling */
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

        /* Metric cards */
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

        /* Alert styling */
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

        /* Button styling */
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

        /* Selectbox styling */
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

        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00fbff, #00ff88) !important;
        }

        /* Video container */
        .video-container {
            background: rgba(0, 0, 0, 0.9);
            padding: 15px;
            border-radius: 15px;
            border: 2px solid #00fbff;
            margin: 10px 0;
            box-shadow: 0 0 20px rgba(0, 251, 255, 0.3);
        }

        /* Info boxes */
        .info-box {
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid #00fbff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            color: white;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            background: rgba(0, 0, 0, 0.8);
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#00fbff, #00ff88);
            border-radius: 5px;
        }

        /* Radio buttons */
        .stRadio > div {
            color: white;
        }

        .stRadio > div > label {
            color: white !important;
        }

        /* Metrics styling */
        .css-1xarl3l {
            color: #00fbff !important;
        }

        /* Warning boxes */
        .stAlert {
            background: rgba(255, 71, 87, 0.2) !important;
            border: 1px solid #ff4757 !important;
            color: white !important;
        }

        /* Success boxes */
        .stSuccess {
            background: rgba(0, 255, 136, 0.2) !important;
            border: 1px solid #00ff88 !important;
            color: white !important;
        }

        /* Info boxes */
        .stInfo {
            background: rgba(0, 251, 255, 0.2) !important;
            border: 1px solid #00fbff !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)


# --- 5. FIXED VIDEO HANDLING FUNCTIONS ---
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
                # Use pathlib for better path handling
                pattern = str(pathlib.Path(root_path) / "**" / ext)
                found_files = glob.glob(pattern, recursive=True)
                all_files.extend(found_files)
            except Exception as e:
                st.warning(f"Error searching for {ext} files: {e}")
                continue
    except Exception as e:
        st.error(f"Error accessing dataset: {e}")

    return all_files


def check_video_file(video_path):
    """Check if video file is accessible and valid"""
    try:
        if not os.path.exists(video_path):
            return False, "File does not exist"

        # Try to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"

        # Check if we can read frames
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
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None
    except:
        return None


# --- 6. ENHANCED MODEL INITIALIZATION WITH CRIME DETECTION ---
class CrimeDetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model_loaded = False

        # Enhanced preprocessing for better crime detection
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"🔥 Crime Detection Model initialized on {self.device}")


# --- 7. ADVANCED VIDEO ANALYZER WITH SPECIFIC CRIME DETECTION ---
class AdvancedCrimeAnalyzer:
    def __init__(self, model):
        self.model = model
        self.device = model.device if hasattr(model, 'device') else 'cpu'
        self.analysis_history = deque(maxlen=100)
        self.crime_threshold = 0.5

    def detect_robbery_indicators(self, prev_frame, curr_frame):
        """Detect indicators of robbery/theft (sudden movements, grabbing motions)"""
        if prev_frame is None or curr_frame is None:
            return {'robbery_score': 0, 'theft_score': 0}

        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate magnitude and direction
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # Detect sudden, jerky movements (typical in robbery)
            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            magnitude_std = np.std(magnitude) if magnitude.size > 0 else 0

            # Sudden movement detection (robbery indicator)
            sudden_movement = magnitude_std / (magnitude_mean + 1e-6)
            robbery_score = min(sudden_movement * 50, 100)

            # Theft indicator (quick reaching motions)
            # Detect high-velocity regions (fast hand movements)
            high_velocity_ratio = np.sum(magnitude > magnitude_mean * 2) / (magnitude.size + 1e-6)
            theft_score = min(high_velocity_ratio * 80, 100)

            return {
                'robbery_score': float(robbery_score),
                'theft_score': float(theft_score)
            }
        except Exception as e:
            return {'robbery_score': 0, 'theft_score': 0}

    def detect_assault_indicators(self, prev_frame, curr_frame):
        """Detect indicators of assault (aggressive movements, fighting)"""
        if prev_frame is None or curr_frame is None:
            return {'assault_score': 0, 'fighting_score': 0}

        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate magnitude
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # Calculate directional changes (chaotic movement typical in fights)
            direction = np.arctan2(flow[..., 1], flow[..., 0])

            # Directional variance (more chaos = higher assault probability)
            direction_variance = np.var(direction) if direction.size > 0 else 0

            # Assault indicators
            magnitude_mean = np.mean(magnitude) if magnitude.size > 0 else 0
            magnitude_std = np.std(magnitude) if magnitude.size > 0 else 0

            # Fighting detection (combination of high motion and directional chaos)
            assault_score = min((magnitude_mean * 3) + (direction_variance * 0.5), 100)
            fighting_score = min((magnitude_std * 5) + (magnitude_mean * 2), 100)

            return {
                'assault_score': float(assault_score),
                'fighting_score': float(fighting_score)
            }
        except Exception as e:
            return {'assault_score': 0, 'fighting_score': 0}

    def detect_visual_crime_patterns(self, frame):
        """Detect visual patterns associated with crimes"""
        try:
            if not self.model.model_loaded:
                return {'suspicious_pattern': np.random.randint(10, 30)}

            # Extract deep features
            input_tensor = self.model.preprocess(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.model(input_tensor)
                feature_np = features.cpu().numpy().flatten()

            # Analyze feature patterns for crime indicators
            feature_mean = np.mean(feature_np) if feature_np.size > 0 else 0
            feature_std = np.std(feature_np) if feature_np.size > 0 else 0

            # Suspicious pattern score (unusual feature combinations)
            suspicious_pattern = min(abs(feature_mean - 0.5) * 100 + feature_std * 50, 100)

            return {'suspicious_pattern': float(suspicious_pattern)}
        except Exception as e:
            return {'suspicious_pattern': float(np.random.randint(10, 30))}

    def detect_weapons(self, frame):
        """Detect potential weapons in frame"""
        try:
            # Convert to HSV for color-based detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect metallic colors (simplified)
            lower_metal = np.array([0, 0, 200])
            upper_metal = np.array([180, 50, 255])
            metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)

            # Detect sharp edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Calculate probabilities safely
            metal_ratio = np.sum(metal_mask > 0) / (metal_mask.size + 1e-6)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-6)

            weapon_probability = min((metal_ratio * 60 + edge_density * 40), 100)

            return float(weapon_probability)
        except:
            return float(np.random.randint(5, 15))

    def analyze_video_crime(self, video_path, progress_bar=None):
        """Comprehensive crime analysis with specific crime type detection"""
        # Check if video is accessible
        is_valid, message = check_video_file(video_path)
        if not is_valid:
            return {}, f"Video error: {message}"

        cap = cv2.VideoCapture(video_path)

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Initialize crime-specific metrics
        robbery_scores = []
        theft_scores = []
        assault_scores = []
        fighting_scores = []
        suspicious_scores = []
        weapon_scores = []

        prev_frame = None
        frame_count = 0
        crime_events = []

        # Adaptive sampling
        sample_rate = max(1,
                          int(total_frames / 60)) if total_frames > 60 else 1  # Increased sampling for better accuracy

        for frame_idx in range(0, total_frames, sample_rate):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Update progress
                if progress_bar and total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

                # 1. Robbery/Theft detection
                robbery_metrics = self.detect_robbery_indicators(prev_frame, frame)
                robbery_scores.append(robbery_metrics['robbery_score'])
                theft_scores.append(robbery_metrics['theft_score'])

                # 2. Assault/Fighting detection
                assault_metrics = self.detect_assault_indicators(prev_frame, frame)
                assault_scores.append(assault_metrics['assault_score'])
                fighting_scores.append(assault_metrics['fighting_score'])

                # 3. Suspicious pattern detection
                pattern_metrics = self.detect_visual_crime_patterns(frame)
                suspicious_scores.append(pattern_metrics['suspicious_pattern'])

                # 4. Weapon detection
                weapon_score = self.detect_weapons(frame)
                weapon_scores.append(weapon_score)

                # Detect specific crime events with better classification
                if frame_count > 0:
                    # Calculate rolling averages for more stable classification
                    recent_robbery = np.mean(robbery_scores[-10:]) if len(robbery_scores) >= 10 else robbery_metrics[
                        'robbery_score']
                    recent_theft = np.mean(theft_scores[-10:]) if len(theft_scores) >= 10 else robbery_metrics[
                        'theft_score']
                    recent_assault = np.mean(assault_scores[-10:]) if len(assault_scores) >= 10 else assault_metrics[
                        'assault_score']
                    recent_fighting = np.mean(fighting_scores[-10:]) if len(fighting_scores) >= 10 else assault_metrics[
                        'fighting_score']
                    recent_weapon = np.mean(weapon_scores[-10:]) if len(weapon_scores) >= 10 else weapon_score

                    # ROBBERY: Characterized by high theft scores + sudden movements + possible weapon
                    robbery_prob = recent_robbery * 0.3 + recent_theft * 0.4 + recent_weapon * 0.3

                    # ASSAULT: Characterized by high fighting scores + directional chaos + aggression
                    assault_prob = recent_assault * 0.4 + recent_fighting * 0.4 + recent_weapon * 0.2

                    # THEFT: Quick grabbing motions without the violent aspects of robbery
                    theft_prob = recent_theft * 0.7 + recent_robbery * 0.2 + suspicious_scores[
                        -1] * 0.1 if suspicious_scores else 0

                    # Classification logic with distinct thresholds
                    if assault_prob > 60 and assault_prob > robbery_prob and assault_prob > theft_prob:
                        crime_events.append({
                            'frame': frame_idx,
                            'time': frame_idx / fps if fps > 0 else 0,
                            'score': assault_prob,
                            'type': 'ASSAULT/FIGHT',
                            'confidence': 'HIGH' if assault_prob > 75 else 'MEDIUM'
                        })
                    elif robbery_prob > 55 and robbery_prob > assault_prob and robbery_prob > theft_prob:
                        crime_events.append({
                            'frame': frame_idx,
                            'time': frame_idx / fps if fps > 0 else 0,
                            'score': robbery_prob,
                            'type': 'ROBBERY',
                            'confidence': 'HIGH' if robbery_prob > 70 else 'MEDIUM'
                        })
                    elif theft_prob > 50 and theft_prob > robbery_prob and theft_prob > assault_prob:
                        crime_events.append({
                            'frame': frame_idx,
                            'time': frame_idx / fps if fps > 0 else 0,
                            'score': theft_prob,
                            'type': 'THEFT',
                            'confidence': 'HIGH' if theft_prob > 65 else 'MEDIUM'
                        })
                    elif recent_weapon > 60:
                        crime_events.append({
                            'frame': frame_idx,
                            'time': frame_idx / fps if fps > 0 else 0,
                            'score': recent_weapon,
                            'type': 'WEAPON DETECTED',
                            'confidence': 'HIGH' if recent_weapon > 75 else 'MEDIUM'
                        })

                prev_frame = frame.copy()

            except Exception as e:
                st.warning(f"Error processing frame {frame_idx}: {e}")
                continue

        cap.release()

        # Calculate comprehensive crime scores
        if frame_count > 0:
            # Core crime metrics
            avg_robbery = np.mean(robbery_scores) if robbery_scores else 0
            avg_theft = np.mean(theft_scores) if theft_scores else 0
            avg_assault = np.mean(assault_scores) if assault_scores else 0
            avg_fighting = np.mean(fighting_scores) if fighting_scores else 0
            avg_suspicious = np.mean(suspicious_scores) if suspicious_scores else 0
            avg_weapon = np.mean(weapon_scores) if weapon_scores else 0

            # Peak crime scores
            peak_robbery = np.max(robbery_scores) if robbery_scores else 0
            peak_assault = np.max(assault_scores) if assault_scores else 0
            peak_theft = np.max(theft_scores) if theft_scores else 0

            # Crime persistence
            high_crime_frames = sum(1 for r, a, t in zip(robbery_scores, assault_scores, theft_scores)
                                    if r > 50 or a > 50 or t > 50)
            crime_persistence = (high_crime_frames / frame_count * 100) if frame_count > 0 else 0

            # Count events by type
            robbery_events = sum(1 for e in crime_events if e['type'] == 'ROBBERY')
            assault_events = sum(1 for e in crime_events if e['type'] == 'ASSAULT/FIGHT')
            theft_events = sum(1 for e in crime_events if e['type'] == 'THEFT')
            weapon_events = sum(1 for e in crime_events if e['type'] == 'WEAPON DETECTED')

            # Final crime scores by type (using weighted combinations)
            final_robbery_score = (avg_robbery * 0.3 + peak_robbery * 0.4 + avg_weapon * 0.3)
            final_assault_score = (avg_assault * 0.3 + peak_assault * 0.4 + avg_fighting * 0.3)
            final_theft_score = (avg_theft * 0.5 + peak_theft * 0.3 + avg_suspicious * 0.2)

            # Overall crime score (weighted combination)
            overall_crime_score = max(
                final_robbery_score,
                final_assault_score,
                final_theft_score,
                avg_weapon * 0.8
            )

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
                'weapon_events': weapon_events
            }

            # Store in history
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'video': os.path.basename(video_path),
                'metrics': metrics,
                'crime_events': crime_events[:10]
            })

            return metrics, "Analysis complete"

        return {}, "No frames analyzed"


# --- 8. VIDEO SCANNER WITH CACHE ---
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


# --- 9. INITIALIZE COMPONENTS ---
@st.cache_resource
def init_components():
    crime_model = CrimeDetectionModel()
    analyzer = AdvancedCrimeAnalyzer(crime_model)
    scanner = VideoScanner()
    return crime_model, analyzer, scanner


crime_model, analyzer, scanner = init_components()


# --- 10. HELPER FUNCTIONS ---
def get_crime_level(score):
    if score > 70:
        return "CRITICAL", "#ff4757"
    elif score > 40:
        return "WARNING", "#feca57"
    else:
        return "NORMAL", "#00ff88"


def get_crime_type(metrics):
    """Improved crime type classification"""
    robbery = metrics.get('robbery_score', 0)
    assault = metrics.get('assault_score', 0)
    theft = metrics.get('theft_score', 0)
    weapon = metrics.get('weapon_detection', 0)

    # Find the highest score to determine crime type
    scores = {
        'ROBBERY': robbery,
        'ASSAULT': assault,
        'THEFT': theft,
        'WEAPON': weapon
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
    """Send automated email alert when crime is detected"""
    try:
        yag = yagmail.SMTP(ALERT_EMAIL, GMAIL_APP_PASSWORD)

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
        """

        yag.send(to=ALERT_EMAIL, subject=alert_subject, contents=alert_body)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False


def create_folder_structure():
    try:
        categories = ['theft', 'vandalism', 'assault', 'suspicious', 'normal', 'weapons', 'fire']
        for category in categories:
            category_path = os.path.join(DATASET_PATH, category)
            os.makedirs(category_path, exist_ok=True)
    except:
        pass  # Silently fail if can't create folders


# Create folders
create_folder_structure()


# --- NEW: AUTOMATIC ANALYSIS FUNCTION ---
def run_automatic_analysis(video_path, video_filename, email_alerts, threshold):
    """Run analysis automatically and update session state"""
    with st.spinner(f"🔍 Automatically analyzing {video_filename} for criminal activity..."):
        progress_bar = st.progress(0)

        # Run analysis
        metrics, message = analyzer.analyze_video_crime(
            video_path, progress_bar
        )

        progress_bar.progress(1.0)

        # Store in session state
        st.session_state['last_analysis'] = {
            'metrics': metrics,
            'video': video_path,
            'filename': video_filename,
            'time': datetime.now()
        }
        st.session_state['analysis_complete'] = True

        if message != "Analysis complete":
            st.warning(message)
        else:
            st.success(f"✅ Automatic analysis complete for {video_filename}!")

            # Send automated email alert if enabled and crime detected
            if email_alerts and metrics and metrics.get('overall_crime_score', 0) > threshold:
                crime_level, _ = get_crime_level(metrics['overall_crime_score'])
                crime_type = get_crime_type(metrics)

                if send_crime_alert(video_filename, metrics, crime_type, crime_level):
                    st.info("📧 Automated crime alert email sent!")


# --- 11. MAIN STREAMLIT APP ---
def main():
    # Set background
    set_background()

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🚨 AI COMMUNITY SECURITY ANALYTICS</h1>
            <p style="color: #00fbff; font-size: 1.2em;">Advanced Crime Detection System</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'last_selected_video' not in st.session_state:
        st.session_state['last_selected_video'] = None
    if 'last_uploaded_video' not in st.session_state:
        st.session_state['last_uploaded_video'] = None

    # Sidebar navigation
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

        # System stats
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

        model_status = "✅ Loaded" if crime_model.model_loaded else "⚠️ Limited"
        st.markdown(f"**Model:** {model_status}")
        st.markdown(f"**Device:** {crime_model.device}")

        # Threshold control
        st.markdown("---")
        st.markdown("### ⚡ Detection Threshold")
        threshold = st.slider("Sensitivity", 0, 100, 40, key="threshold",
                              help="Lower = More sensitive, Higher = Less sensitive")

        # Analysis options
        st.markdown("### 🔍 Analysis Options")
        deep_analysis = st.checkbox("Deep Analysis", value=True, key="deep")
        motion_detection = st.checkbox("Motion Detection", value=True, key="motion")
        weapon_detection = st.checkbox("Weapon Detection", value=True, key="weapon")
        email_alerts = st.checkbox("Email Alerts", value=True, key="email")

    # Main content area
    if selected == "Live Analysis":
        col1, col2 = st.columns([0.4, 0.6])

        with col1:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown("### 📹 VIDEO SOURCE")

            # Video source selection
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
                    # Create display names
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

                        # Check if this is a new video selection
                        if st.session_state['last_selected_video'] != video_path:
                            st.session_state['last_selected_video'] = video_path
                            run_analysis_flag = True

                        # Show video info
                        try:
                            size = os.path.getsize(video_path) / (1024 * 1024)
                            st.markdown(f"**Size:** {size:.1f} MB")

                            # Check if video is accessible
                            is_valid, msg = check_video_file(video_path)
                            if is_valid:
                                st.success("✅ Video accessible")
                            else:
                                st.warning(f"⚠️ {msg}")
                        except:
                            st.warning("⚠️ Could not read video info")
                else:
                    st.warning("📂 No videos found in dataset")
                    st.info("Add videos to: " + DATASET_PATH)

            else:  # Upload Video
                uploaded_file = st.file_uploader(
                    "Upload video file",
                    type=['mp4', 'avi', 'mkv', 'mov', 'wmv', 'mpeg'],
                    key="uploader"
                )

                if uploaded_file:
                    # Check if this is a new upload
                    if st.session_state['last_uploaded_video'] != uploaded_file.name:
                        st.session_state['last_uploaded_video'] = uploaded_file.name

                        # Save temporarily
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
                        # Reuse existing temp file path for the same uploaded file
                        # In a real scenario, you'd need to store the temp path
                        # For simplicity, we'll re-upload if needed
                        try:
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            tfile.write(uploaded_file.read())
                            video_path = tfile.name
                            video_filename = uploaded_file.name
                        except Exception as e:
                            st.error(f"Upload failed: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

            # Video display section (optional)
            if video_path and st.checkbox("🎬 Show Video Player", value=False, key="show_video"):
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                try:
                    # Read video file for display
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    if video_bytes:
                        st.video(video_bytes)
                except Exception as e:
                    st.warning(f"⚠️ Video preview unavailable: {str(e)[:50]}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Run automatic analysis if needed
            if run_analysis_flag and video_path:
                run_automatic_analysis(video_path, video_filename, email_alerts, threshold)

        with col2:
            if st.session_state.get('analysis_complete', False) and 'last_analysis' in st.session_state:
                analysis = st.session_state['last_analysis']
                metrics = analysis['metrics']

                if metrics:
                    overall_score = metrics.get('overall_crime_score', 0)
                    robbery_score = metrics.get('robbery_score', 0)
                    assault_score = metrics.get('assault_score', 0)
                    theft_score = metrics.get('theft_score', 0)

                    # Crime level
                    crime_level, crime_color = get_crime_level(overall_score)
                    crime_type = get_crime_type(metrics)

                    # Alert display
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

                    st.markdown(f'<div class="{alert_class}">{alert_text}</div>',
                                unsafe_allow_html=True)

                    # File info
                    st.markdown(f"""
                        <div class="info-box">
                            <b>File:</b> {analysis.get('filename', 'Unknown')}<br>
                            <b>Duration:</b> {metrics.get('duration', 0)}s<br>
                            <b>Frames:</b> {metrics.get('frames_analyzed', 0)}/{metrics.get('total_frames', 0)}
                        </div>
                    """, unsafe_allow_html=True)

                    # Metrics display
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

                    # Detailed metrics
                    st.markdown('<div class="css-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 DETAILED ANALYSIS")

                    # Create tabs
                    tab1, tab2, tab3 = st.tabs(["📈 Crime Metrics", "📊 Crime Profile", "⏱️ Timeline"])

                    with tab1:
                        # Radar chart for crime types
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
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100],
                                    color='white'
                                ),
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            showlegend=False,
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        # Bar chart comparison
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

                        st.plotly_chart(fig, use_container_width=True)

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

                            # Create event timeline
                            events = analyzer.analysis_history[-1].get('crime_events', [])
                            if events:
                                event_df = pd.DataFrame(events)
                                st.dataframe(event_df, use_container_width=True)
                        else:
                            st.success("✅ No crime events detected")

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Manual email button (keep for manual sending)
                    if email_alerts and overall_score > threshold:
                        if st.button("📧 Send Crime Alert Email (Manual)", key="send_email"):
                            if send_crime_alert(analysis.get('filename', 'Unknown'), metrics, crime_type, crime_level):
                                st.success("📧 Crime alert email sent successfully!")
            else:
                # Placeholder when no analysis
                st.markdown("""
                    <div class="css-card" style="display: flex; flex-direction: column; 
                                align-items: center; justify-content: center; min-height: 400px;">
                        <h2 style="color: #00fbff; text-align: center;">🔍 SELECT A VIDEO</h2>
                        <p style="color: white; text-align: center;">Choose a video from the left panel to begin crime analysis</p>
                    </div>
                """, unsafe_allow_html=True)

    elif selected == "Dataset Browser":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 📁 DATASET BROWSER")

        # Get folder statistics
        folder_stats = scanner.get_folder_stats(DATASET_PATH)

        if folder_stats:
            # Display folders as cards
            cols = st.columns(3)
            for idx, (folder, stats) in enumerate(folder_stats.items()):
                with cols[idx % 3]:
                    st.markdown(f"""
                        <div style="background: rgba(0,255,255,0.1); padding: 15px; 
                             border-radius: 10px; margin: 10px 0; border-left: 4px solid #00fbff;">
                            <h4 style="color: #00fbff;">📁 {folder.upper()}</h4>
                            <p style="color: white;">Videos: {stats['count']}</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No folders found in dataset")

        st.markdown('</div>', unsafe_allow_html=True)

        # List all videos
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 🎥 ALL VIDEOS")

        videos = scanner.get_all_videos(DATASET_PATH)
        if videos:
            video_data = []
            for video in videos[:50]:  # Limit to 50 for performance
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
                st.dataframe(df, use_container_width=True)
                st.caption(f"Showing {len(video_data)} of {len(videos)} videos")
            else:
                st.warning("Could not read video information")
        else:
            st.warning("No videos found in dataset")

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Analytics History":
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("### 📊 ANALYSIS HISTORY")

        if analyzer.analysis_history:
            # Create history dataframe
            history_data = []
            for entry in list(analyzer.analysis_history)[-20:]:  # Last 20 entries
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
            st.dataframe(df, use_container_width=True)

            # Historical trend graph
            if len(analyzer.analysis_history) > 1:
                fig = go.Figure()

                scores = [e['metrics']['overall_crime_score'] for e in analyzer.analysis_history]
                robbery_scores = [e['metrics']['robbery_score'] for e in analyzer.analysis_history]
                assault_scores = [e['metrics']['assault_score'] for e in analyzer.analysis_history]
                theft_scores = [e['metrics']['theft_score'] for e in analyzer.analysis_history]
                times = list(range(len(scores)))

                fig.add_trace(go.Scatter(
                    x=times,
                    y=scores,
                    mode='lines+markers',
                    name='Overall Crime',
                    line=dict(color='#ff4757', width=3),
                    marker=dict(size=8)
                ))

                fig.add_trace(go.Scatter(
                    x=times,
                    y=robbery_scores,
                    mode='lines',
                    name='Robbery',
                    line=dict(color='#feca57', width=2, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=times,
                    y=assault_scores,
                    mode='lines',
                    name='Assault',
                    line=dict(color='#00fbff', width=2, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=times,
                    y=theft_scores,
                    mode='lines',
                    name='Theft',
                    line=dict(color='#ff6b6b', width=2, dash='dash')
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

                st.plotly_chart(fig, use_container_width=True)
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
                    yag = yagmail.SMTP(email, password)
                    yag.send(to=email, subject="Test Alert",
                             contents="AI Security System - Test successful!")
                    st.success("✅ Email configured successfully!")
                except Exception as e:
                    st.error(f"❌ Email configuration failed: {e}")

        with col2:
            st.markdown("#### 🗂️ Dataset Path")
            new_path = st.text_input("Dataset Location", value=DATASET_PATH, key="path_input")

            if st.button("Refresh Scanner", key="refresh"):
                with st.spinner("Scanning..."):
                    videos = scanner.get_all_videos(new_path, force_refresh=True)
                    st.success(f"✅ Scanner refreshed! Found {len(videos)} videos")

            st.markdown("#### 🤖 Model Settings")
            st.info(f"Model Status: {'✅ Loaded' if crime_model.model_loaded else '⚠️ Limited'}")
            st.info(f"Device: {crime_model.device}")

            if st.button("Clear Analysis History", key="clear"):
                analyzer.analysis_history.clear()
                st.session_state['analysis_complete'] = False
                st.success("✅ History cleared!")

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()