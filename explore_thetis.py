import os
import pandas as pd
import cv2
import json

# ---------------------------
# 1️⃣ PATH CONFIGURATION
# ---------------------------
THETIS_PATH = r"./dataset/"   # ← change this to your dataset folder

# ---------------------------
# 2️⃣ CHECK FOLDERS
# ---------------------------
print("📁 Exploring THETIS dataset...\n")
print("Folders found:", os.listdir(THETIS_PATH))

# Expected: rgb, depth, skeleton, labels.csv
rgb_folder = os.path.join(THETIS_PATH, "rgb")
depth_folder = os.path.join(THETIS_PATH, "depth")
skeleton_folder = os.path.join(THETIS_PATH, "skeleton")

# ---------------------------
# 3️⃣ LOAD LABELS
# ---------------------------
labels_path = os.path.join(THETIS_PATH, "labels.csv")
if os.path.exists(labels_path):
    df = pd.read_csv(labels_path)
    print("\n✅ Labels file loaded successfully!\n")
    print(df.head())
    
    print("\n🎾 Number of clips per class:")
    print(df['class'].value_counts())
    
    if 'subject_id' in df.columns:
        print("\n👤 Number of subjects:", df['subject_id'].nunique())
else:
    print("\n⚠️ No labels.csv file found!")

# ---------------------------
# 4️⃣ CHECK ONE VIDEO
# ---------------------------
if os.path.exists(rgb_folder):
    videos = [v for v in os.listdir(rgb_folder) if v.endswith(('.mp4', '.avi'))]
    if videos:
        sample_video = os.path.join(rgb_folder, videos[0])
        print(f"\n🎥 Opening video: {videos[0]}")
        
        cap = cv2.VideoCapture(sample_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Frames in this video: {frame_count}")
        cap.release()
    else:
        print("\n⚠️ No video files found in RGB folder.")
else:
    print("\n⚠️ RGB folder not found!")

# ---------------------------
# 5️⃣ CHECK ONE SKELETON FILE
# ---------------------------
if os.path.exists(skeleton_folder):
    files = [f for f in os.listdir(skeleton_folder) if f.endswith(('.json', '.txt'))]
    if files:
        sample_skeleton = os.path.join(skeleton_folder, files[0])
        print(f"\n🦴 Checking skeleton file: {files[0]}")

        with open(sample_skeleton, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"Frames: {len(data)}")
                    print(f"Joints per frame: {len(data[0]) if len(data) > 0 else 0}")
                    print(f"First joint example: {data[0][0]}")
                else:
                    print(f"JSON keys: {list(data.keys())}")
            except json.JSONDecodeError:
                print("⚠️ Skeleton file not valid JSON format.")
    else:
        print("\n⚠️ No skeleton files found.")
else:
    print("\n⚠️ Skeleton folder not found!")

print("\n✅ Dataset exploration complete!")
