import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ---------------------------
# PATH CONFIGURATION
# ---------------------------
THETIS_PATH = r"./dataset/"
rgb_folder = os.path.join(THETIS_PATH, "VIDEO_RGB")

classes = ['backhand', 'backhand_slice', 'backhand_volley', 'backhand2hands', 
           'flat_service', 'forehand_flat', 'forehand_openstands', 'forehand_slice', 
           'forehand_volley', 'kick_service', 'slice_service', 'smash']

# ---------------------------
# SELECT RANDOM VIDEO
# ---------------------------
def get_random_video():
    """Pick a random video from the dataset"""
    random_class = random.choice(classes)
    class_folder = os.path.join(rgb_folder, random_class)
    
    if os.path.exists(class_folder):
        videos = [v for v in os.listdir(class_folder) if v.endswith('.avi')]
        if videos:
            random_video = random.choice(videos)
            video_path = os.path.join(class_folder, random_video)
            return video_path, random_class, random_video
    return None, None, None

# ---------------------------
# KEEP ONLY LARGEST OBJECT
# ---------------------------
def get_largest_component(mask):
    """Keep only the largest connected component (the player)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    
    if num_labels <= 1:
        return np.zeros_like(mask)
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    
    return largest_mask

# ---------------------------
# 3D HARRIS CORNER (STIP) DETECTION
# ---------------------------
def detect_harris_3d_stip(frame_buffer, fg_masks, sigma_spatial=2.0, sigma_temporal=1.5, threshold=0.0001):
    """
    Detect Spatio-Temporal Interest Points using 3D Harris corner detection
    """
    if len(frame_buffer) < 3:
        return []
    
    # Stack frames to create 3D volume
    volume = np.stack(frame_buffer, axis=2).astype(np.float32)
    combined_mask = fg_masks[len(fg_masks)//2]
    
    # Compute spatial derivatives
    Ix = cv2.Sobel(volume, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(volume, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute temporal derivative
    It = np.zeros_like(volume)
    for t in range(1, volume.shape[2]):
        It[:, :, t] = volume[:, :, t] - volume[:, :, t-1]
    
    # Compute structure tensor elements
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    It2 = It * It
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It
    
    # Apply Gaussian smoothing
    ksize_spatial = int(6 * sigma_spatial + 1)
    if ksize_spatial % 2 == 0:
        ksize_spatial += 1
    
    Ix2_smooth = np.zeros_like(Ix2)
    Iy2_smooth = np.zeros_like(Iy2)
    It2_smooth = np.zeros_like(It2)
    IxIy_smooth = np.zeros_like(IxIy)
    IxIt_smooth = np.zeros_like(IxIt)
    IyIt_smooth = np.zeros_like(IyIt)
    
    for t in range(volume.shape[2]):
        Ix2_smooth[:, :, t] = cv2.GaussianBlur(Ix2[:, :, t], (ksize_spatial, ksize_spatial), sigma_spatial)
        Iy2_smooth[:, :, t] = cv2.GaussianBlur(Iy2[:, :, t], (ksize_spatial, ksize_spatial), sigma_spatial)
        It2_smooth[:, :, t] = cv2.GaussianBlur(It2[:, :, t], (ksize_spatial, ksize_spatial), sigma_spatial)
        IxIy_smooth[:, :, t] = cv2.GaussianBlur(IxIy[:, :, t], (ksize_spatial, ksize_spatial), sigma_spatial)
        IxIt_smooth[:, :, t] = cv2.GaussianBlur(IxIt[:, :, t], (ksize_spatial, ksize_spatial), sigma_spatial)
        IyIt_smooth[:, :, t] = cv2.GaussianBlur(IyIt[:, :, t], (ksize_spatial, ksize_spatial), sigma_spatial)
    
    t_mid = volume.shape[2] // 2
    response = np.zeros((volume.shape[0], volume.shape[1]))
    
    for y in range(volume.shape[0]):
        for x in range(volume.shape[1]):
            if combined_mask[y, x] == 0:
                continue
            
            M = np.array([
                [Ix2_smooth[y, x, t_mid], IxIy_smooth[y, x, t_mid], IxIt_smooth[y, x, t_mid]],
                [IxIy_smooth[y, x, t_mid], Iy2_smooth[y, x, t_mid], IyIt_smooth[y, x, t_mid]],
                [IxIt_smooth[y, x, t_mid], IyIt_smooth[y, x, t_mid], It2_smooth[y, x, t_mid]]
            ])
            
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            response[y, x] = det_M - 0.04 * (trace_M ** 2)
    
    response_thresh = response.copy()
    response_thresh[response < threshold * response.max()] = 0
    
    kernel_size = 5
    dilated = cv2.dilate(response_thresh, np.ones((kernel_size, kernel_size)))
    local_maxima = (response_thresh == dilated) & (response_thresh > 0)
    
    stips = []
    y_coords, x_coords = np.where(local_maxima)
    for y, x in zip(y_coords, x_coords):
        stips.append((x, y, t_mid, response[y, x]))
    
    return stips

# ---------------------------
# EXTRACT ALL STIPS FROM VIDEO
# ---------------------------
def extract_all_stips(video_path, class_name, video_name, crop_percent=0.15, temporal_window=5):
    """
    Extract all STIPs from the entire video and store them with time coordinate
    """
    print(f"🎥 Processing: {class_name}/{video_name}")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Total frames: {total_frames}, FPS: {fps}")
    print(f"🔍 Extracting STIPs from entire video...")
    
    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    frame_buffer = []
    fg_mask_buffer = []
    frame_count = 0
    
    all_stips = []  # Store all STIPs with (x, y, t, response)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        h, w = frame.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        cropped_frame = frame[crop_h:h-crop_h, crop_w:w-crop_w]
        
        fg_mask = mog.apply(cropped_frame)
        fg_mask_clean = get_largest_component(fg_mask)
        
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        
        frame_buffer.append(gray)
        fg_mask_buffer.append(fg_mask_clean)
        
        if len(frame_buffer) > temporal_window:
            frame_buffer.pop(0)
            fg_mask_buffer.pop(0)
        
        if len(frame_buffer) == temporal_window:
            stips = detect_harris_3d_stip(frame_buffer, fg_mask_buffer)
            # Add global time coordinate
            for x, y, t_local, response in stips:
                all_stips.append((x, y, frame_count, response))
        
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames, STIPs found: {len(all_stips)}")
    
    cap.release()
    print(f"\n✅ Total STIPs detected: {len(all_stips)}")
    
    return all_stips, total_frames

# ---------------------------
# VISUALIZE 3D POINT CLOUD
# ---------------------------
def visualize_3d_stip_cloud(all_stips, total_frames, class_name):
    """
    Create an interactive 3D point cloud visualization of STIPs
    """
    if len(all_stips) == 0:
        print("⚠️ No STIPs to visualize!")
        return
    
    # Extract coordinates
    xs = [s[0] for s in all_stips]
    ys = [s[1] for s in all_stips]
    ts = [s[2] for s in all_stips]
    responses = [s[3] for s in all_stips]
    
    # Normalize responses for color mapping (invert for darker = less movement)
    responses_norm = np.array(responses)
    if responses_norm.max() > 0:
        responses_norm = (responses_norm - responses_norm.min()) / (responses_norm.max() - responses_norm.min())
        # Invert: 1 - normalized value (darker = less movement)
        responses_norm = 1 - responses_norm
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Plot point cloud with blue gradient only (no white)
    scatter = ax.scatter(xs, ys, ts, c=responses_norm, cmap='Blues_r', s=10, alpha=0.8, vmin=0, vmax=1)
    
    ax.set_xlabel('X (spatial)', fontsize=12, color='white')
    ax.set_ylabel('Y (spatial)', fontsize=12, color='white')
    ax.set_zlabel('Time (frames)', fontsize=12, color='white')
    ax.set_title(f'3D STIP Cloud - {class_name}\nTotal Points: {len(all_stips)}', 
                 fontsize=14, fontweight='bold', color='white')
    
    # Change axis colors to white
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Movement Strength (darker = less)', rotation=270, labelpad=20, color='blue')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('stip_3d_cloud.png', dpi=150, bbox_inches='tight')
    print("💾 Saved 3D cloud as 'stip_3d_cloud.png'")
    plt.show()
    
    # Create animated rotating view
    print("\n🔄 Creating rotating animation...")
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    def update_view(frame):
        ax2.clear()
        ax2.scatter(xs, ys, ts, c=responses_norm, cmap='Blues_r', s=10, alpha=0.6, vmin=0, vmax=1)
        ax2.set_xlabel('X (spatial)', fontsize=12)
        ax2.set_ylabel('Y (spatial)', fontsize=12)
        ax2.set_zlabel('Time (frames)', fontsize=12)
        ax2.set_title(f'3D STIP Cloud - {class_name} (Rotating)', fontsize=14, fontweight='bold')
        ax2.view_init(elev=20, azim=frame)
        return ax2,
    
    anim = FuncAnimation(fig2, update_view, frames=np.arange(0, 360, 2), interval=50)
    anim.save('stip_3d_cloud_rotating.gif', writer='pillow', fps=20)
    print("💾 Saved rotating animation as 'stip_3d_cloud_rotating.gif'")
    plt.show()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    video_path, class_name, video_name = get_random_video()
    
    if video_path:
        # Extract all STIPs
        all_stips, total_frames = extract_all_stips(video_path, class_name, video_name)
        
        # Visualize 3D cloud
        if len(all_stips) > 0:
            visualize_3d_stip_cloud(all_stips, total_frames, class_name)
    else:
        print("⚠️ No video found!")
