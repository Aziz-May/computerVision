import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from train_lstm_v3 import TennisAttentionLSTM, AdvancedSkeletonDataset, normalize_skeleton, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "dataset/mediapipe_skeletons3D"
MODEL_PATH = "best_attention_lstm.pth"
CLASSES_FILE = "classes.npy"
MAX_FRAMES = 80
NUM_JOINTS = 33
CHANNELS = 3
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data(classes):
    X = []
    y = []
    
    # Load all data again to find the test set (Deterministically)
    # We use the same seed '42' so we get the exact same Test Split as training
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    print("📂 Reloading Data to reconstruct Test Set...")
    
    # 1. Load Raw
    raw_X = []
    raw_y = []
    for label in classes:
        files = glob.glob(os.path.join(DATA_PATH, label, "*.npy"))
        for file in files:
            data = np.load(file)
            raw_X.append(data)
            raw_y.append(label)
            
    # 2. Encode
    le = LabelEncoder()
    le.fit(classes) # Force order from file
    y_enc = le.transform(raw_y)
    
    # 3. Split (Must match training seed!)
    _, X_test, _, y_test = train_test_split(raw_X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    
    return X_test, y_test

def evaluate():
    # 1. Load Class Names
    if not os.path.exists(CLASSES_FILE):
        print("❌ Error: classes.npy not found. Did you train the model?")
        return
    class_names = np.load(CLASSES_FILE)
    num_classes = len(class_names)
    print(f"✅ Loaded {num_classes} classes: {class_names}")

    # 2. Load Data
    X_test, y_test = load_test_data(class_names)
    
    # 3. Create Dataset/Loader
    test_ds = AdvancedSkeletonDataset(X_test, y_test, is_training=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 4. Load Model
    model = TennisAttentionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("✅ Model Loaded.")

    # 5. Run Inference
    all_preds = []
    all_labels = []
    
    print("🚀 Running Inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Generate Metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 7. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Tennis Action Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("💾 Saved Confusion Matrix to 'confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    evaluate()