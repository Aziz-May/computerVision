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

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "dataset/mediapipe_skeletons3D"
MAX_FRAMES = 80
NUM_JOINTS = 33
CHANNELS = 3
INPUT_SIZE = NUM_JOINTS * CHANNELS
HIDDEN_SIZE = 256       # Increased from 128
NUM_LAYERS = 2
EPOCHS = 70
BATCH_SIZE = 32
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 TRAINING DEVICE: {device}")

# ==========================================
# 1. ADVANCED AUGMENTATIONS
# ==========================================
def augment_rotate(data):
    """Rotates the skeleton around the Y-axis (-25 to +25 degrees)"""
    theta = np.random.uniform(-0.4, 0.4) # radians
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    # Reshape to (Frames*Joints, 3) to multiply, then reshape back
    N, J, C = data.shape
    reshaped = data.reshape(-1, 3)
    rotated = np.dot(reshaped, R.T)
    return rotated.reshape(N, J, C)

def augment_scale(data):
    """Makes the player 5% to 15% Taller/Shorter or Wider/Thinner"""
    scale = np.random.uniform(0.85, 1.15)
    return data * scale

def augment_time_warp(data):
    """Simulates faster or slower execution (Speed +/- 20%)"""
    N, J, C = data.shape
    x = np.arange(N)
    
    # Create a random speed factor
    speed_factor = np.random.uniform(0.8, 1.2)
    new_length = int(N * speed_factor)
    
    # Interpolate to new length
    new_x = np.linspace(0, N-1, new_length)
    
    # Reshape for interpolation
    flat_data = data.reshape(N, -1)
    f = interp1d(x, flat_data, axis=0, kind='linear')
    new_data = f(new_x)
    
    return new_data.reshape(new_length, J, C)

def normalize_skeleton(data):
    """Centers hips to (0,0,0)"""
    left_hip = data[:, 23, :]
    right_hip = data[:, 24, :]
    hip_center = (left_hip + right_hip) / 2
    return data - hip_center[:, np.newaxis, :]

# ==========================================
# 2. DATASET WITH ON-THE-FLY AUGMENTATION
# ==========================================
class AugmentedDataset(Dataset):
    def __init__(self, X, y, is_training=False):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_training = is_training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.X[idx].copy() # (Frames, 33, 3)
        
        # 1. Normalization (ALWAYS DO THIS)
        data = normalize_skeleton(data)

        # 2. Augmentation (ONLY FOR TRAINING)
        if self.is_training:
            if random.random() < 0.5: data = augment_rotate(data)
            if random.random() < 0.5: data = augment_scale(data)
            if random.random() < 0.5: data = augment_time_warp(data)

        # 3. Pad / Cut to MAX_FRAMES
        L, J, C = data.shape
        if L < MAX_FRAMES:
            padding = np.zeros((MAX_FRAMES - L, J, C))
            data = np.vstack((data, padding))
        else:
            # Take center crop
            start = (L - MAX_FRAMES) // 2
            data = data[start : start + MAX_FRAMES, :, :]

        # 4. Flatten for LSTM
        flat_data = data.reshape(MAX_FRAMES, -1)
        
        return torch.tensor(flat_data, dtype=torch.float32), self.y[idx]

# ==========================================
# 3. BI-DIRECTIONAL LSTM MODEL
# ==========================================
class TennisBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TennisBiLSTM, self).__init__()
        
        # Bidirectional=True allows the model to see the FUTURE frames
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.5,
            bidirectional=True  # <--- MAGIC CHANGE
        )
        
        # Hidden size * 2 because it's bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Frames, 99)
        out, _ = self.lstm(x)
        
        # Combine the last frame of forward pass and first frame of backward pass
        # Or simpler: Global Average Pooling over time
        out = torch.mean(out, dim=1)
        
        return self.fc(out)

# ==========================================
# 4. LOADING & TRAINING
# ==========================================
def load_raw_data():
    X = []
    y = []
    classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"📂 Loading Raw Data...")
    
    for label in classes:
        files = glob.glob(os.path.join(DATA_PATH, label, "*.npy"))
        for file in files:
            data = np.load(file) # We don't pad here, we pad in Dataset
            X.append(data)
            y.append(label)
            
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, len(classes), le

if __name__ == "__main__":
    X, y, num_classes, le = load_raw_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create Datasets
    train_ds = AugmentedDataset(X_train, y_train, is_training=True)
    test_ds = AugmentedDataset(X_test, y_test, is_training=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Model
    model = TennisBiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Training Bi-LSTM with Augmentations...")
    
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
        
        # Scheduler
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_bilstm.pth")
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%")

    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")
    np.save("classes.npy", le.classes_)