import os
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "dataset/mediapipe_skeletons3D"
# ==========================================

def audit_labels():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Path not found: {DATA_PATH}")
        return

    # 1. Get folder names (The raw labels)
    # We sort them because LabelEncoder sorts alphabetically by default
    folder_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    if len(folder_names) == 0:
        print("⚠️ No folders found!")
        return

    print(f"📂 Found {len(folder_names)} Classes in '{DATA_PATH}'")
    print("="*60)
    print(f"{'ID':<5} | {'CLASS NAME (Label)':<25} | {'SAMPLES'}")
    print("-" * 60)

    # 2. Simulate the Label Encoding
    le = LabelEncoder()
    le.fit(folder_names)
    
    total_samples = 0
    
    for class_name in folder_names:
        # Get the Integer ID assigned by the encoder
        class_id = le.transform([class_name])[0]
        
        # Count files
        class_dir = os.path.join(DATA_PATH, class_name)
        files = glob.glob(os.path.join(class_dir, "*.npy"))
        count = len(files)
        total_samples += count
        
        print(f"{class_id:<5} | {class_name:<25} | {count}")

    print("="*60)
    print(f"TOTAL SAMPLES: {total_samples}")
    
    # 3. Double Check a random file's label
    print("\n🔍 RANDOM CHECK:")
    import random
    random_class = random.choice(folder_names)
    class_id = le.transform([random_class])[0]
    print(f"If the model predicts ID [{class_id}], it means --> '{random_class}'")

if __name__ == "__main__":
    audit_labels()