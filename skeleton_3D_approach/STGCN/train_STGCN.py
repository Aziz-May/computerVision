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
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 TRAINING DEVICE: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
# ==========================================

def get_adjacency_matrix():
    """Defines the 33-joint graph structure"""
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
        
    # Self-loops + Normalization
    A = A + np.eye(NUM_JOINTS)
    D = np.sum(A, axis=1)
    D_inv = np.power(D, -1).flatten()
    D_inv[np.isinf(D_inv)] = 0.
    D_mat = np.diag(D_inv)
    
    A_norm = D_mat.dot(A)
    return torch.tensor(A_norm, dtype=torch.float32).to(device)

class GraphConv(nn.Module):
    """Spatial Graph Convolution: Output = A * X * W"""
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.A = A
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (N, C, T, V) where V=Vertices/Joints
        
        # 1. Feature Transform (Conv 1x1 acts as Matrix Mult W)
        x = self.conv(x)
        
        # 2. Spatial Aggregation (A * X)
        # Einsum: 'vu' is adjacency (V,V), 'nctu' is input (N,C,T,V) -> Output (N,C,T,V)
        x = torch.einsum('vu,nctu->nctv', self.A, x)
        
        return x

class STGCN_Block(nn.Module):
    """One block of Spatial-Temporal Convolution"""
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(STGCN_Block, self).__init__()
        
        # Spatial Graph Conv
        self.gcn = GraphConv(in_channels, out_channels, A)
        
        # Temporal Conv (9x1 kernel looks at 9 frames)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3)
        )
        
        # Residual connection (if channels change)
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
        self.A = A
        
        # Input has 3 channels (x,y,z)
        self.block1 = STGCN_Block(in_channels, 64, A)
        self.block2 = STGCN_Block(64, 128, A, stride=2) # Downsample time by 2
        self.block3 = STGCN_Block(128, 256, A, stride=2) # Downsample time by 2
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input x: (N, T, V, C) -> Need (N, C, T, V) for PyTorch Conv2d
        x = x.permute(0, 3, 1, 2) 
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.pool(x) # (N, 256, 1, 1)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

def load_data():
    X = []
    y = []
    classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    print(f"📂 Loading {len(classes)} classes...")
    for label in classes:
        files = glob.glob(os.path.join(DATA_PATH, label, "*.npy"))
        for file in files:
            data = np.load(file) # (Frames, 33, 3)
            
            # Pad/Cut
            L, J, C = data.shape
            if L < MAX_FRAMES:
                padding = np.zeros((MAX_FRAMES - L, J, C))
                data = np.vstack((data, padding))
            else:
                data = data[:MAX_FRAMES, :, :]
                
            X.append(data)
            y.append(label)
            
    # Convert labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    return np.array(X, dtype=np.float32), y_enc, len(classes)

if __name__ == "__main__":
    # 1. Load Data
    X, y, num_classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Create Tensors
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # 3. Build Model
    A = get_adjacency_matrix()
    model = STGCN_Model(num_classes, NUM_JOINTS, CHANNELS, A).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Starting training on {device} for {EPOCHS} epochs...")
    
    # 4. Training Loop
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
            
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        
    # 5. Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    print(f"\n🏆 Final Test Accuracy: {100 * correct / total:.2f}%")
    
    # Save Model
    torch.save(model.state_dict(), "stgcn_tennis_pytorch.pth")
    print("💾 Model saved as 'stgcn_tennis_pytorch.pth'")