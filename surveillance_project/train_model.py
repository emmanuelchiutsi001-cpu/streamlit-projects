# train_model.py
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = r"C:\Users\emmanuel chiutsi\Documents\dataset-video-split"  # CHANGE THIS TO YOUR PATH
CRIME_CATEGORIES = [
    'abuse', 'arrest', 'arson', 'assault', 'bulglary', 'clapping',
    'explosion', 'fighting', 'meet_and_split', 'normal_videos',
    'roadaccidents', 'robbery', 'shooting', 'shoplifting', 'sitting',
    'standing_still', 'stealing', 'vandalism', 'walking',
    'walking_while_reading_book', 'walking_while_using_phone'
]


def safe_get_video_files(root_path):
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv', '*.flv', '*.m4v', '*.mpeg']
    all_files = []
    if not os.path.exists(root_path):
        return all_files
    for ext in video_extensions:
        pattern = os.path.join(root_path, '**', ext)
        found_files = glob.glob(pattern, recursive=True)
        all_files.extend(found_files)
    return all_files


def extract_single_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
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


class FastVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frame = extract_single_frame(video_path)
        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if self.transform:
            frame = self.transform(frame)
        return frame, label


class OptimizedCrimeClassifier(nn.Module):
    def __init__(self, num_classes=21):
        super(OptimizedCrimeClassifier, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class OptimizedCrimeTrainer:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = CRIME_CATEGORIES
        self.trained = False
        self.performance_metrics = {}
        self.training_info = {}
        self.inverse_class_mapping = {}
        self.effective_class_names = []
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def map_filename_to_category(self, filename):
        name = os.path.splitext(filename)[0].lower()
        for category in self.class_names:
            if category in name or name.startswith(category):
                return category
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
                class_idx = self.class_names.index(category)
                video_paths.append(video)
                labels.append(class_idx)
                class_counts[category] += 1
        return video_paths, labels, class_counts

    def train_fast_model(self, dataset_path, progress_callback=None):
        try:
            train_path = os.path.join(dataset_path, 'train')
            if not os.path.exists(train_path):
                return False

            video_paths, labels, class_counts = self.prepare_dataset_from_folder(train_path)
            if len(video_paths) == 0:
                return False

            unique_classes = sorted(list(set(labels)))
            valid_classes = []
            for class_idx in unique_classes:
                class_name = self.class_names[class_idx]
                if class_counts[class_name] >= 2:
                    valid_classes.append(class_idx)

            if len(valid_classes) < 2:
                return False

            valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
            video_paths = [video_paths[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]

            class_to_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_classes)}
            remapped_labels = [class_to_idx[label] for label in labels]
            effective_class_names = [self.class_names[i] for i in valid_classes]
            effective_num_classes = len(effective_class_names)

            # Transforms
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            full_dataset = FastVideoDataset(video_paths, remapped_labels, train_transform)
            labels_array = np.array(remapped_labels)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(video_paths, labels_array))

            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            val_video_paths = [video_paths[i] for i in val_idx]
            val_labels = [remapped_labels[i] for i in val_idx]
            val_dataset = FastVideoDataset(val_video_paths, val_labels, val_transform)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

            self.model = OptimizedCrimeClassifier(num_classes=effective_num_classes)
            self.model = self.model.to(self.device)

            class_counts_array = np.array([class_counts[self.class_names[c]] for c in valid_classes])
            class_weights = 1.0 / class_counts_array
            class_weights = class_weights / class_weights.sum() * effective_num_classes
            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
            optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.01)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

            num_epochs = 20
            best_val_acc = 0
            best_model_state = None
            patience = 6
            patience_counter = 0

            for epoch in range(num_epochs):
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for images, labels_batch in train_loader:
                    images = images.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels_batch.size(0)
                    train_correct += (predicted == labels_batch).sum().item()

                train_acc = 100 * train_correct / train_total if train_total > 0 else 0
                avg_train_loss = train_loss / len(train_loader)

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
                scheduler.step(val_acc)

                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_acc'].append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if progress_callback:
                    progress_callback((epoch + 1) / num_epochs,
                                      f"Epoch {epoch + 1}/{num_epochs} - Val Acc: {val_acc:.1f}%")

                if patience_counter >= patience:
                    break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            if len(all_preds) > 0:
                self.performance_metrics = {
                    'accuracy': float(accuracy_score(all_labels, all_preds) * 100),
                    'precision': float(
                        precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100),
                    'recall': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100),
                    'f1_score': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100)
                }

            self.inverse_class_mapping = {v: k for k, v in class_to_idx.items()}
            self.effective_class_names = effective_class_names
            self.trained = True
            return True

        except Exception as e:
            print(f"Training error: {e}")
            return False


def train_and_save_model(dataset_path, model_save_path="saved_model.pkl"):
    """Train model and save it to file"""
    print("Starting model training...")
    print(f"Dataset path: {dataset_path}")

    trainer = OptimizedCrimeTrainer()

    def progress_callback(progress, message):
        print(f"{progress * 100:.1f}% - {message}")

    success = trainer.train_fast_model(dataset_path, progress_callback)

    if success:
        print("\n✅ Training completed successfully!")
        print(f"Final Validation Accuracy: {trainer.performance_metrics.get('accuracy', 0):.1f}%")

        # Save model and all necessary components
        model_data = {
            'model_state_dict': trainer.model.state_dict(),
            'class_mapping': trainer.inverse_class_mapping,
            'inverse_class_mapping': trainer.inverse_class_mapping,
            'effective_class_names': trainer.effective_class_names,
            'performance_metrics': trainer.performance_metrics,
            'training_info': trainer.training_info,
            'training_history': trainer.training_history,
            'num_classes': len(trainer.effective_class_names)
        }

        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Model saved to {model_save_path}")
        print(f"File size: {os.path.getsize(model_save_path) / 1024 / 1024:.2f} MB")
        return True
    else:
        print("\n❌ Training failed!")
        return False


if __name__ == "__main__":
    # IMPORTANT: Change this to your actual dataset path
    DATASET_PATH = r"C:\Users\emmanuel chiutsi\Documents\dataset-video-split"

    print("=" * 60)
    print("CRIME DETECTION MODEL TRAINER")
    print("=" * 60)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 60)

    success = train_and_save_model(DATASET_PATH, "saved_model.pkl")

    if success:
        print("\n🎉 You can now deploy this model to Streamlit Cloud!")
        print("Next steps:")
        print("1. Upload 'saved_model.pkl' to your GitHub repository")
        print("2. Upload the app files to GitHub")
        print("3. Deploy on Streamlit Cloud")
    else:
        print("\n⚠️ Please check your dataset path and try again.")