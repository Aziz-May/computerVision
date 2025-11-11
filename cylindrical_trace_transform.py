import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    """Detect Spatio-Temporal Interest Points using 3D Harris corner detection"""
    if len(frame_buffer) < 3:
        return []
    
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
    """Extract all STIPs from the entire video"""
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
    
    all_stips = []
    
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
            for x, y, t_local, response in stips:
                all_stips.append((x, y, frame_count, response))
        
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames, STIPs found: {len(all_stips)}")
    
    cap.release()
    print(f"\n✅ Total STIPs detected: {len(all_stips)}")
    
    return all_stips, total_frames

# ---------------------------
# 3D CYLINDRICAL TRACE TRANSFORM
# ---------------------------
def cylindrical_trace_transform_3d(all_stips, num_angles=36, num_radii=50):
    """
    Apply 3D Cylindrical Trace Transform to STIPs
    
    The cylindrical transform maps 3D points (x, y, t) to cylindrical coordinates (r, θ, z)
    where z is time, and we compute traces (functional values) along cylindrical shells.
    """
    if len(all_stips) == 0:
        print("⚠️ No STIPs to transform!")
        return None
    
    print(f"\n🔄 Applying 3D Cylindrical Trace Transform...")
    print(f"   Angles: {num_angles}, Radii: {num_radii}")
    
    # Extract coordinates
    xs = np.array([s[0] for s in all_stips])
    ys = np.array([s[1] for s in all_stips])
    ts = np.array([s[2] for s in all_stips])
    responses = np.array([s[3] for s in all_stips])
    
    # Center the spatial coordinates
    center_x = (xs.max() + xs.min()) / 2
    center_y = (ys.max() + ys.min()) / 2
    
    xs_centered = xs - center_x
    ys_centered = ys - center_y
    
    # Convert to cylindrical coordinates (r, θ, z)
    # r = distance from center axis (parallel to time)
    # θ = angle around the axis
    # z = time coordinate
    radii = np.sqrt(xs_centered**2 + ys_centered**2)
    angles = np.arctan2(ys_centered, xs_centered)  # [-pi, pi]
    
    # Normalize angles to [0, 2π]
    angles = (angles + np.pi) % (2 * np.pi)
    
    # Create trace transform matrix
    max_radius = radii.max()
    trace_matrix = np.zeros((num_angles, num_radii))
    trace_counts = np.zeros((num_angles, num_radii))
    
    # Bin STIPs into (angle, radius) cells and accumulate responses
    for i in range(len(all_stips)):
        angle_idx = int((angles[i] / (2 * np.pi)) * num_angles) % num_angles
        radius_idx = int((radii[i] / max_radius) * (num_radii - 1))
        
        trace_matrix[angle_idx, radius_idx] += responses[i]
        trace_counts[angle_idx, radius_idx] += 1
    
    # Average the responses in each bin
    mask = trace_counts > 0
    trace_matrix[mask] /= trace_counts[mask]
    
    print(f"✅ Transform complete!")
    print(f"   Max trace value: {trace_matrix.max():.6f}")
    print(f"   Non-zero cells: {np.sum(trace_counts > 0)} / {num_angles * num_radii}")
    
    return trace_matrix, radii, angles, max_radius

# ---------------------------
# VISUALIZE CYLINDRICAL TRACE TRANSFORM
# ---------------------------
def visualize_cylindrical_transform(trace_matrix, class_name):
    """Visualize the 3D Cylindrical Trace Transform as a 2D heatmap"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('black')
    
    # Heatmap representation
    im1 = axes[0].imshow(trace_matrix, aspect='auto', cmap='Blues_r', origin='lower')
    axes[0].set_xlabel('Radius', fontsize=12, color='white')
    axes[0].set_ylabel('Angle (bins)', fontsize=12, color='white')
    axes[0].set_title(f'Cylindrical Trace Transform - {class_name}', 
                      fontsize=14, fontweight='bold', color='white')
    axes[0].set_facecolor('black')
    axes[0].tick_params(colors='white')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Trace Value', rotation=270, labelpad=20, color='white')
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
    
    # Polar representation
    ax_polar = plt.subplot(122, projection='polar')
    ax_polar.set_facecolor('black')
    
    # Create polar mesh
    num_angles, num_radii = trace_matrix.shape
    theta = np.linspace(0, 2*np.pi, num_angles)
    r = np.linspace(0, 1, num_radii)
    Theta, R = np.meshgrid(theta, r)
    
    # Plot
    im2 = ax_polar.contourf(Theta, R, trace_matrix.T, levels=20, cmap='Blues_r')
    ax_polar.set_title(f'Polar View - {class_name}', fontsize=14, fontweight='bold', 
                       color='white', pad=20)
    ax_polar.tick_params(colors='white')
    ax_polar.spines['polar'].set_color('white')
    ax_polar.grid(color='gray', linestyle='--', linewidth=0.5)
    
    cbar2 = plt.colorbar(im2, ax=ax_polar, pad=0.1)
    cbar2.set_label('Trace Value', rotation=270, labelpad=20, color='white')
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    plt.savefig('cylindrical_trace_transform.png', dpi=150, bbox_inches='tight', facecolor='black')
    print("💾 Saved transform as 'cylindrical_trace_transform.png'")
    plt.show()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    video_path, class_name, video_name = get_random_video()
    
    if video_path:
        # Extract all STIPs
        all_stips, total_frames = extract_all_stips(video_path, class_name, video_name)
        
        if len(all_stips) > 0:
            # Apply cylindrical trace transform
            result = cylindrical_trace_transform_3d(all_stips)
            
            if result is not None:
                trace_matrix, radii, angles, max_radius = result
                
                # Visualize
                visualize_cylindrical_transform(trace_matrix, class_name)
        else:
            print("⚠️ No STIPs detected!")
    else:
        print("⚠️ No video found!")
