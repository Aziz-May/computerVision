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
classes = [ 'backhand','backhand_slice' , 'backhand_volley' , 'backhand2hands', 'flat_service' , 'forehand_flat', 'forehand_openstands', 'forehand_slice', 'forehand_volley', 'kick_service', 'slice_service', 'smash']
rgb_folder = os.path.join(THETIS_PATH, "VIDEO_RGB")
depth_folder = os.path.join(THETIS_PATH, "VIDEO_Depth")
mask_folder = os.path.join(THETIS_PATH, "VIDEO_Mask")
skel2d_folder = os.path.join(THETIS_PATH, "VIDEO_Skelet2D")
skel3d_folder = os.path.join(THETIS_PATH, "VIDEO_Skelet3D")

# ---------------------------
# 3️⃣ LOAD LABELS
# ---------------------------
""" labels_path = os.path.join(THETIS_PATH, "labels.csv")
if os.path.exists(labels_path):
    df = pd.read_csv(labels_path)
    print("\n✅ Labels file loaded successfully!\n")
    print(df.head())
    
    print("\n🎾 Number of clips per class:")
    print(df['class'].value_counts())
    
    if 'subject_id' in df.columns:
        print("\n👤 Number of subjects:", df['subject_id'].nunique())
else:
    print("\n⚠️ No labels.csv file found!")"""

# ---------------------------
# 4️⃣ CHECK ONE VIDEO
# ---------------------------
if os.path.exists(rgb_folder):
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(rgb_folder, class_name)
        if os.path.exists(class_folder):
            videos = [v for v in os.listdir(class_folder) if v.endswith(('.mp4', '.avi'))]
            if videos:
                sample_video = os.path.join(class_folder, videos[0])
                print(f"\n🎥 Opening video: {class_name}/{videos[0]}")
                
                cap = cv2.VideoCapture(sample_video)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Frames in this video: {frame_count}")
                cap.release()
                break
    else:
        print("\n⚠️ No video files found in RGB folder.")
else:
    print("\n⚠️ RGB folder not found!")

# ---------------------------
# 5️⃣ CHECK ONE SKELETON FILE (2D)
# ---------------------------
if os.path.exists(skel2d_folder):
    for class_name in classes:
        class_folder = os.path.join(skel2d_folder, class_name)
        if os.path.exists(class_folder):
            files = [f for f in os.listdir(class_folder) if f.endswith(('.avi', '.mp4'))]
            if files:
                sample_skeleton = os.path.join(class_folder, files[0])
                print(f"\n🦴 Checking 2D skeleton video: {class_name}/{files[0]}")

                cap = cv2.VideoCapture(sample_skeleton)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Frames: {frame_count}, Resolution: {width}x{height}")
                cap.release()
                break
    else:
        print("\n⚠️ No skeleton 2D files found.")
else:
    print("\n⚠️ Skeleton 2D folder not found!")

# ---------------------------
# 6️⃣ CHECK DEPTH VIDEOS
# ---------------------------
if os.path.exists(depth_folder):
    for class_name in classes:
        class_folder = os.path.join(depth_folder, class_name)
        if os.path.exists(class_folder):
            files = [f for f in os.listdir(class_folder) if f.endswith(('.avi', '.mp4'))]
            if files:
                sample_video = os.path.join(class_folder, files[0])
                print(f"\n📊 Checking Depth video: {class_name}/{files[0]}")
                
                cap = cv2.VideoCapture(sample_video)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Frames: {frame_count}, Resolution: {width}x{height}")
                cap.release()
                break
    else:
        print("\n⚠️ No depth videos found.")
else:
    print("\n⚠️ Depth folder not found!")

# ---------------------------
# 7️⃣ CHECK MASK VIDEOS
# ---------------------------
if os.path.exists(mask_folder):
    for class_name in classes:
        class_folder = os.path.join(mask_folder, class_name)
        if os.path.exists(class_folder):
            files = [f for f in os.listdir(class_folder) if f.endswith(('.avi', '.mp4'))]
            if files:
                sample_video = os.path.join(class_folder, files[0])
                print(f"\n🎭 Checking Mask video: {class_name}/{files[0]}")
                
                cap = cv2.VideoCapture(sample_video)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Frames: {frame_count}, Resolution: {width}x{height}")
                cap.release()
                break
    else:
        print("\n⚠️ No mask videos found.")
else:
    print("\n⚠️ Mask folder not found!")

# ---------------------------
# 8️⃣ CHECK 3D SKELETON VIDEOS
# ---------------------------
if os.path.exists(skel3d_folder):
    for class_name in classes:
        class_folder = os.path.join(skel3d_folder, class_name)
        if os.path.exists(class_folder):
            files = [f for f in os.listdir(class_folder) if f.endswith(('.avi', '.mp4'))]
            if files:
                sample_video = os.path.join(class_folder, files[0])
                print(f"\n🦴 Checking 3D skeleton video: {class_name}/{files[0]}")
                
                cap = cv2.VideoCapture(sample_video)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Frames: {frame_count}, Resolution: {width}x{height}")
                cap.release()
                break
    else:
        print("\n⚠️ No 3D skeleton videos found.")
else:
    print("\n⚠️ 3D skeleton folder not found!")

# ---------------------------
# 9️⃣ DATASET SUMMARY
# ---------------------------
print("\n" + "="*50)
print("📊 DATASET SUMMARY")
print("="*50)
for class_name in classes:
    rgb_path = os.path.join(rgb_folder, class_name)
    if os.path.exists(rgb_path):
        video_count = len([f for f in os.listdir(rgb_path) if f.endswith(('.avi', '.mp4'))])
        print(f"{class_name:20s} - {video_count:3d} videos")
    
   

print("\n✅ Dataset exploration complete!")
