import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import random
import mediapipe as mp

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT = "dataset/mediapipe_skeletons3D"
NUM_VIDEOS_TO_SHOW = 5
# ==========================================

# MediaPipe connection list (to draw lines between joints)
# We use the official MediaPipe topology
mp_pose = mp.solutions.pose
CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

def get_random_files(root_path, count=5):
    """Finds all .npy files and picks random ones."""
    all_files = []
    
    # Walk through all subfolders
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".npy"):
                all_files.append(os.path.join(root, file))
    
    if len(all_files) < count:
        return all_files
    return random.sample(all_files, count)

def play_skeleton_animation(file_path):
    """Loads an .npy file and plays it as a 3D animation."""
    
    # 1. Load Data
    # Shape: (Frames, 33, 3)
    data = np.load(file_path)
    frames, num_joints, dims = data.shape
    
    filename = os.path.basename(file_path)
    action_class = os.path.basename(os.path.dirname(file_path))
    
    print(f"🎬 Playing: {action_class} / {filename}")
    print(f"   Frames: {frames}")

    # 2. Setup Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate limits to keep camera steady (Get min/max of all frames)
    # We add a little padding
    x_min, x_max = data[:, :, 0].min(), data[:, :, 0].max()
    y_min, y_max = data[:, :, 1].min(), data[:, :, 1].max()
    z_min, z_max = data[:, :, 2].min(), data[:, :, 2].max()

    def update(frame_idx):
        ax.clear()
        
        # Get current frame joints
        current_frame = data[frame_idx] # Shape (33, 3)
        xs = current_frame[:, 0]
        ys = current_frame[:, 1]
        zs = current_frame[:, 2]

        # 1. Draw Joints (Dots)
        # MediaPipe coords: Y points down usually, Z is depth
        # We negate Y to make the person stand upright in the plot
        ax.scatter(xs, zs, -ys, c='r', marker='o', s=20)

        # 2. Draw Connections (Lines)
        for connection in CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            # Get start and end points
            start_point = current_frame[start_idx]
            end_point = current_frame[end_idx]
            
            # Draw line
            ax.plot(
                [start_point[0], end_point[0]], 
                [start_point[2], end_point[2]], # Swap Y and Z for better 3D view
                [-start_point[1], -end_point[1]], # Invert Y
                c='black', linewidth=2
            )

        # 3. Set Fixed Limits (Stop camera from shaking)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max) # Swap Y/Z limits logic
        ax.set_zlim(-y_max, -y_min) # Invert Y limits
        
        ax.set_title(f"{action_class}: {filename}\nFrame {frame_idx}/{frames}")
        ax.set_xlabel('X')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Height)')

    # Create Animation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)
    
    plt.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Error: Folder not found: {DATASET_ROOT}")
    else:
        files = get_random_files(DATASET_ROOT, NUM_VIDEOS_TO_SHOW)
        
        print(f"🔍 Found {len(files)} files. Showing {NUM_VIDEOS_TO_SHOW}...")
        
        for i, f in enumerate(files):
            print(f"\n--- Video {i+1} of {len(files)} ---")
            play_skeleton_animation(f)
            # The code pauses here until you close the window