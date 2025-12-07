import os
import glob
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your main dataset folder
DATASET_ROOT = "dataset" 

INPUT_FOLDER = "VIDEO_RGB"
OUTPUT_FOLDER = "mediapipe_skeletons3D"
# ==========================================

def check_dataset_integrity():
    input_path = os.path.join(DATASET_ROOT, INPUT_FOLDER)
    output_path = os.path.join(DATASET_ROOT, OUTPUT_FOLDER)

    if not os.path.exists(output_path):
        print(f"❌ Error: Output folder not found at {output_path}")
        return

    print(f"📊 Verifying Dataset: {DATASET_ROOT}")
    print(f"{'CLASS NAME':<25} | {'VIDEOS (.avi)':<15} | {'SKELETONS (.npy)':<15} | {'STATUS':<10}")
    print("-" * 75)

    total_videos = 0
    total_skeletons = 0
    missing_files = []

    # Get list of action classes (subfolders)
    if os.path.exists(input_path):
        classes = sorted([d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))])
    else:
        # If input folder moved, just check output folder
        classes = sorted([d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))])

    for action_class in classes:
        # Paths
        class_in_path = os.path.join(input_path, action_class)
        class_out_path = os.path.join(output_path, action_class)

        # Count files
        # Check input videos
        if os.path.exists(class_in_path):
            videos = glob.glob(os.path.join(class_in_path, "*.avi"))
            n_videos = len(videos)
        else:
            n_videos = 0

        # Check output skeletons
        if os.path.exists(class_out_path):
            skeletons = glob.glob(os.path.join(class_out_path, "*.npy"))
            n_skeletons = len(skeletons)
        else:
            n_skeletons = 0

        # Update totals
        total_videos += n_videos
        total_skeletons += n_skeletons

        # Determine Status
        if n_videos == n_skeletons and n_videos > 0:
            status = "✅ OK"
        elif n_videos == 0:
             status = "⚠️ No Input"
        else:
            status = f"❌ MISSING {n_videos - n_skeletons}"
            missing_files.append(action_class)

        print(f"{action_class:<25} | {n_videos:<15} | {n_skeletons:<15} | {status}")

    print("-" * 75)
    print(f"{'TOTAL':<25} | {total_videos:<15} | {total_skeletons:<15} |")
    print("-" * 75)

    # Final Result
    if total_videos == total_skeletons:
        print("\n🎉 SUCCESS: All videos have been processed successfully!")
        
        # Optional: Quick Integrity Check on first file
        print("\n🔍 Performing quick random integrity check...")
        try:
            first_class = classes[0]
            first_file = os.listdir(os.path.join(output_path, first_class))[0]
            full_path = os.path.join(output_path, first_class, first_file)
            data = np.load(full_path)
            print(f"   Test load: {first_file}")
            print(f"   Shape: {data.shape} (Frames, Joints, XYZ)")
            if data.shape[1:] == (33, 3):
                print("   Structure: ✅ Valid (33 joints, 3D)")
            else:
                print("   Structure: ❌ Invalid shape")
        except Exception as e:
            print(f"   Check failed: {e}")

    else:
        print(f"\n⚠️ WARNING: You are missing {total_videos - total_skeletons} skeleton files.")
        print("Classes with missing files:", missing_files)

if __name__ == "__main__":
    check_dataset_integrity()