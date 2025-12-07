import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "dataset/mediapipe_skeletons3D"
MAX_FRAMES = 80        # Reduced slightly to focus on the core action
NUM_JOINTS = 33
CHANNELS = 3
INPUT_SIZE = NUM_JOINTS * CHANNELS  # 33 * 3 = 99 features per frame
HIDDEN_SIZE = 128      # Simpler brain
NUM_LAYERS = 2         # Only 2 layers
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 TRAINING DEVICE: {device}")

# ==========================================
# 1. NORMALIZATION (Keep this, it's good)
# ==========================================
def normalize_skeleton(data):
    """Centers the skeleton hips to (0,0,0)"""
    # 23=Left Hip, 24=Right Hip
    left_hip = data[:, 23, :]
    right_hip = data[:, 24, :]
    hip_center = (left_hip + right_hip) / 2
    return data - hip_center[:, np.newaxis, :]

# ==========================================
# 2. LSTM MODEL DEFINITION
# ==========================================
class TennisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TennisLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.4 # High dropout to prevent overfitting
        )
        
        # Simple classifier at the end
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Frames, 99)
        
        # LSTM output: (Batch, Frames, Hidden)
        # _ is the hidden state (h_n, c_n), we don't need it
        out, _ = self.lstm(x)
        
        # We only care about the result of the LAST frame
        # out[:, -1, :] means: "Take all batches, Last Frame, All Features"
        last_frame_feature = out[:, -1, :]
        
        return self.fc(last_frame_feature)

# ==========================================
# 3. DATA LOADING
# ==========================================
def load_data():
    X = []
    y = []
    classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"📂 Loading {len(classes)} classes...")
    
    for label in classes:
        files = glob.glob(os.path.join(DATA_PATH, label, "*.npy"))
        for file in files:
            data = np.load(file) # (Frames, 33, 3)
            
            # 1. Normalize
            data = normalize_skeleton(data)
            
            # 2. Pad/Cut
            L, J, C = data.shape
            if L < MAX_FRAMES:
                padding = np.zeros((MAX_FRAMES - L, J, C))
                data = np.vstack((data, padding))
            else:
                # Take the MIDDLE frames (often where the action is), not just the start
                start = (L - MAX_FRAMES) // 2
                data = data[start : start + MAX_FRAMES, :, :]
            
            # 3. Flatten (Frames, 33, 3) -> (Frames, 99)
            # LSTM needs a flat vector per frame
            flat_data = data.reshape(MAX_FRAMES, -1)
            
            X.append(flat_data)
            y.append(label)
            
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return np.array(X, dtype=np.float32), y_enc, len(classes), le

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    X, y, num_classes, le = load_data()
    
    # Stratify ensure we split evenly across all classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Build Model
    model = TennisLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Training LSTM on {device}...")
    
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        train_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
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
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_lstm.pth")
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%")

    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")
    np.save("classes.npy", le.classes_)