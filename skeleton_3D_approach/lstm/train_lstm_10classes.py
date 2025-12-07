import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from scipy.interpolate import interp1d
import torch.nn.functional as F

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "dataset/mediapipe_skeletons3D"
MAX_FRAMES = 80
NUM_JOINTS = 33
CHANNELS = 3
# Input size: Position (3) + Velocity (3) = 6 per joint
INPUT_SIZE = NUM_JOINTS * CHANNELS * 2  
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EPOCHS = 60           # Reduced slightly as it converges faster now
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 TRAINING DEVICE: {device}")

# ==========================================
# 1. AUGMENTATION & PREPROCESSING
# ==========================================
def normalize_skeleton(data):
    """Centers hips to (0,0,0)"""
    left_hip = data[:, 23, :]
    right_hip = data[:, 24, :]
    hip_center = (left_hip + right_hip) / 2
    return data - hip_center[:, np.newaxis, :]

def augment_rotate(data):
    theta = np.random.uniform(-0.4, 0.4)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    N, J, C = data.shape
    reshaped = data.reshape(-1, 3)
    rotated = np.dot(reshaped, R.T)
    return rotated.reshape(N, J, C)

def augment_scale(data):
    scale = np.random.uniform(0.85, 1.15)
    return data * scale

def augment_time_warp(data):
    N, J, C = data.shape
    speed_factor = np.random.uniform(0.8, 1.2)
    new_length = int(N * speed_factor)
    x = np.arange(N)
    new_x = np.linspace(0, N-1, new_length)
    flat_data = data.reshape(N, -1)
    f = interp1d(x, flat_data, axis=0, kind='linear')
    new_data = f(new_x)
    return new_data.reshape(new_length, J, C)

# ==========================================
# 2. DATASET (Position + Velocity)
# ==========================================
class AdvancedSkeletonDataset(Dataset):
    def __init__(self, X, y, is_training=False):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_training = is_training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.X[idx].copy()
        
        # 1. Normalization
        data = normalize_skeleton(data)

        # 2. Augmentation
        if self.is_training:
            if random.random() < 0.5: data = augment_rotate(data)
            if random.random() < 0.5: data = augment_scale(data)
            if random.random() < 0.5: data = augment_time_warp(data)

        # 3. Pad / Cut
        L, J, C = data.shape
        if L < MAX_FRAMES:
            padding = np.zeros((MAX_FRAMES - L, J, C))
            data = np.vstack((data, padding))
        else:
            start = (L - MAX_FRAMES) // 2
            data = data[start : start + MAX_FRAMES, :, :]

        # 4. VELOCITY CALCULATION
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        
        # 5. Concatenate
        combined = np.concatenate([data, velocity], axis=-1)
        flat_data = combined.reshape(MAX_FRAMES, -1)
        
        return torch.tensor(flat_data, dtype=torch.float32), self.y[idx]

# ==========================================
# 3. ATTENTION MODEL
# ==========================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        context = torch.sum(encoder_outputs * weights, dim=1)
        return context, weights

class TennisAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TennisAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.5,
            bidirectional=True 
        )
        self.attention = SelfAttention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)
        out = self.fc(context_vector)
        return out

# ==========================================
# 4. LOAD & MERGE LABELS (THE FIX)
# ==========================================
def load_and_merge_data():
    X = []
    y = []
    
    # Get raw folders
    raw_classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"📂 Scanning {len(raw_classes)} folders...")
    
    for folder_name in raw_classes:
        files = glob.glob(os.path.join(DATA_PATH, folder_name, "*.npy"))
        
        # --- THE MERGE LOGIC ---
        label = folder_name
        if folder_name in ["flat_service", "kick_service", "slice_service"]:
            label = "service"
        # -----------------------

        for file in files:
            data = np.load(file)
            X.append(data)
            y.append(label)
            
    # Encode the new merged labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    final_classes = list(le.classes_)
    print(f"✅ Merged into {len(final_classes)} Classes: {final_classes}")
    
    return X, y_enc, len(final_classes), le

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    X, y, num_classes, le = load_and_merge_data()
    
    # Stratify split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    train_ds = AdvancedSkeletonDataset(X_train, y_train, is_training=True)
    test_ds = AdvancedSkeletonDataset(X_test, y_test, is_training=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = TennisAttentionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"🚀 Training Merged Model (Targeting 98% Accuracy)...")
    
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_merged_lstm.pth")
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%")

    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")
    np.save("classes_merged.npy", le.classes_)