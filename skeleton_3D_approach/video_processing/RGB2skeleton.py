import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# =========================================================================
# CONFIGURATION
# =========================================================================
# 👇 CHANGE THIS to the folder containing 'VIDEO_RGB'
DATASET_ROOT_PATH = "dataset" 

INPUT_FOLDER_NAME = "VIDEO_RGB"
OUTPUT_FOLDER_NAME = "mediapipe_skeletons3D"

# Determine number of CPU cores to use (leave 1 or 2 free for the OS)
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
# =========================================================================

def process_single_video(args):
    """
    Worker function to process a single video file.
    Args tuple: (video_path, output_path)
    """
    video_path, save_path = args

    # Skip if already exists
    if os.path.exists(save_path):
        return "Skipped"

    # Initialize MediaPipe INSIDE the process (Crucial for multiprocessing)
    mp_pose = mp.solutions.pose
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Cannot open"

    frames_data = []

    # Configure Pose for this specific process
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            try:
                results = pose.process(image_rgb)
                
                if results.pose_world_landmarks:
                    frame_joints = []
                    for landmark in results.pose_world_landmarks.landmark:
                        frame_joints.append([landmark.x, landmark.y, landmark.z])
                    frames_data.append(frame_joints)
                else:
                    frames_data.append(np.zeros((33, 3)))
            except Exception:
                frames_data.append(np.zeros((33, 3)))

    cap.release()

    # Save data
    if len(frames_data) > 0:
        np.save(save_path, np.array(frames_data))
        return "Success"
    else:
        return "Empty"

def main():
    input_root = os.path.join(DATASET_ROOT_PATH, INPUT_FOLDER_NAME)
    output_root = os.path.join(DATASET_ROOT_PATH, OUTPUT_FOLDER_NAME)

    if not os.path.exists(input_root):
        print(f"❌ Error: Could not find input: {input_root}")
        return

    # Prepare Task List
    tasks = []
    
    # 1. Scan folders and build a list of work to do
    print("🔍 Scanning files...")
    class_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]

    for class_name in class_folders:
        input_class_path = os.path.join(input_root, class_name)
        output_class_path = os.path.join(output_root, class_name)

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        video_files = glob.glob(os.path.join(input_class_path, "*.avi"))
        
        for video_path in video_files:
            filename = os.path.basename(video_path)
            filename_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(output_class_path, filename_no_ext + ".npy")
            
            # Add to queue
            tasks.append((video_path, save_path))

    print(f"🚀 Found {len(tasks)} videos to process.")
    print(f"🔥 Starting Multiprocessing with {NUM_WORKERS} CPU cores.")
    print("   (Your computer will get hot and fans will spin!)")

    # 2. Run in Parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_video, task) for task in tasks]
        
        # Monitor progress
        for _ in tqdm(as_completed(futures), total=len(tasks), unit="vid"):
            pass

    print("\n" + "="*50)
    print("✅ BATCH PROCESSING COMPLETE")
    print(f"Data saved to: {output_root}")
    print("="*50)

if __name__ == "__main__":
    # Windows requires this guard for multiprocessing
    multiprocessing.freeze_support()
    main()