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

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "dataset/mediapipe_skeletons3D"
MAX_FRAMES = 100
NUM_JOINTS = 33
CHANNELS = 3
EPOCHS = 80             # Increased epochs
BATCH_SIZE = 32
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 TRAINING DEVICE: {device}")

# ==========================================
# 1. AUGMENTATION FUNCTIONS (THE SECRET SAUCE)
# ==========================================
def random_rotate(skeleton):
    """Rotates the skeleton around the Y-axis (Up-Down axis)"""
    # Angle in radians (between -30 and +30 degrees)
    theta = np.random.uniform(-np.pi/6, np.pi/6)
    c, s = np.cos(theta), np.sin(theta)
    
    # Rotation Matrix for Y-axis
    R = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])
    
    # Apply to all frames and joints: (Frames*Joints, 3) @ (3,3)
    shape = skeleton.shape
    flat_skel = skeleton.reshape(-1, 3)
    rotated = flat_skel @ R.T
    return rotated.reshape(shape)

def add_noise(skeleton, sigma=0.01):
    """Adds tiny jitter to coordinates"""
    noise = np.random.normal(0, sigma, skeleton.shape)
    return skeleton + noise

def normalize_skeleton(data):
    """Centers hips to (0,0,0)"""
    left_hip = data[:, 23, :]
    right_hip = data[:, 24, :]
    hip_center = (left_hip + right_hip) / 2
    return data - hip_center[:, np.newaxis, :]

# ==========================================
# 2. CUSTOM DATASET CLASS
# ==========================================
class SkeletonDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data copy
        data = self.X[idx].copy()
        
        # 1. Normalize (Center Hips)
        data = normalize_skeleton(data)
        
        # 2. Augment (Only for Training Data!)
        if self.augment:
            if random.random() > 0.5:
                data = random_rotate(data)
            if random.random() > 0.5:
                data = add_noise(data)
        
        # Convert to Tensor (C, T, V) format for PyTorch
        # Current: (Frames, Joints, Channels) -> Target: (Channels, Frames, Joints)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.permute(2, 0, 1) # (C, T, V)
        
        return data, self.y[idx]

# ==========================================
# 3. MODEL DEFINITION (ST-GCN)
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
            nn.Dropout(0.5)
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
        self.block1 = STGCN_Block(in_channels, 64, A)
        self.block2 = STGCN_Block(64, 128, A, stride=2)
        self.block3 = STGCN_Block(128, 256, A, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# MAIN EXECUTION
# ==========================================
def load_all_files():
    X = []
    y = []
    classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"📂 Loading Raw Data...")
    
    for label in classes:
        files = glob.glob(os.path.join(DATA_PATH, label, "*.npy"))
        for file in files:
            data = np.load(file)
            # Pad/Cut
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

if __name__ == "__main__":
    # 1. Load raw data
    X, y, num_classes, le = load_all_files()
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Create Datasets with Augmentation
    # NOTE: Augment=True for Train, False for Test
    train_dataset = SkeletonDataset(X_train, y_train, augment=True)
    test_dataset = SkeletonDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 4. Model Setup
    A = get_adjacency_matrix()
    model = STGCN_Model(num_classes, NUM_JOINTS, CHANNELS, A).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Training V3 (Augmented) on {device}...")
    
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
        
        # Step the scheduler (lower LR if Val Acc doesn't improve)
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_stgcn_v3.pth")
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")
    np.save("classes.npy", le.classes_)