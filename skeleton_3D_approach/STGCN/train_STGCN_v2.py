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
MAX_FRAMES = 100
NUM_JOINTS = 33
CHANNELS = 3
EPOCHS = 60            # Increased epochs slightly
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5     # Increased to fight overfitting (was 0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 TRAINING DEVICE: {device}")

# ==========================================
# 1. NORMALIZATION FUNCTION (THE FIX)
# ==========================================
def normalize_skeleton(data):
    """
    Centers the skeleton so the hips are always at (0,0,0).
    Makes the model invariant to camera position.
    data shape: (Frames, 33, 3)
    """
    # MediaPipe: 23=Left Hip, 24=Right Hip
    # Calculate hip center for every frame
    left_hip = data[:, 23, :]
    right_hip = data[:, 24, :]
    hip_center = (left_hip + right_hip) / 2  # Shape: (Frames, 3)
    
    # Subtract hip center from all joints
    # specific numpy broadcasting: (Frames, 33, 3) - (Frames, 1, 3)
    normalized_data = data - hip_center[:, np.newaxis, :]
    
    return normalized_data

# ==========================================
# 2. GRAPH & MODEL
# ==========================================
def get_adjacency_matrix():
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 9), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), 
        (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), 
        (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), 
        (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
    ]
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in connections:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + np.eye(NUM_JOINTS)
    D = np.sum(A, axis=1)
    D_inv = np.power(D, -1).flatten()
    D_inv[np.isinf(D_inv)] = 0.
    D_mat = np.diag(D_inv)
    return torch.tensor(D_mat.dot(A), dtype=torch.float32).to(device)

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.A = A
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return torch.einsum('vu,nctu->nctv', self.A, x)

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(STGCN_Block, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(DROPOUT_RATE) # Use higher dropout
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)

class STGCN_Model(nn.Module):
    def __init__(self, num_classes, num_joints, in_channels, A):
        super(STGCN_Model, self).__init__()
        # REDUCED COMPLEXITY to prevent memorization
        # Was: 64 -> 128 -> 256
        # Now: 32 -> 64 -> 128
        self.block1 = STGCN_Block(in_channels, 32, A)
        self.block2 = STGCN_Block(32, 64, A, stride=2)
        self.block3 = STGCN_Block(64, 128, A, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) 
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 3. LOAD DATA
# ==========================================
def load_data():
    X = []
    y = []
    classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"📂 Loading Data from {len(classes)} classes...")
    
    for label in classes:
        files = glob.glob(os.path.join(DATA_PATH, label, "*.npy"))
        for file in files:
            data = np.load(file)
            
            # --- APPLY NORMALIZATION HERE ---
            data = normalize_skeleton(data)
            # --------------------------------
            
            L, J, C = data.shape
            if L < MAX_FRAMES:
                padding = np.zeros((MAX_FRAMES - L, J, C))
                data = np.vstack((data, padding))
            else:
                data = data[:MAX_FRAMES, :, :]
            X.append(data)
            y.append(label)
            
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return np.array(X, dtype=np.float32), y_enc, len(classes), le

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    X, y, num_classes, label_encoder = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    A = get_adjacency_matrix()
    model = STGCN_Model(num_classes, NUM_JOINTS, CHANNELS, A).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight decay
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Training V2 (Normalized) on {device}...")
    
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
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
            
        # Validation Loop
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
            torch.save(model.state_dict(), "best_stgcn.pth")
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {100*correct/total:.1f}% | Val Acc: {val_acc:.1f}%")
        
    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")
    np.save("classes.npy", label_encoder.classes_)