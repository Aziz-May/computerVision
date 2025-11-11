import os
import cv2
import numpy as np
import random

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
    
    Args:
        frame_buffer: List of consecutive grayscale frames (temporal dimension)
        fg_masks: List of corresponding foreground masks
        sigma_spatial: Spatial scale for Gaussian smoothing
        sigma_temporal: Temporal scale for Gaussian smoothing
        threshold: Threshold for corner response
    
    Returns:
        stips: List of (x, y, t) coordinates of detected STIPs
    """
    if len(frame_buffer) < 3:
        return []
    
    # Stack frames to create 3D volume (height, width, time)
    volume = np.stack(frame_buffer, axis=2).astype(np.float32)
    
    # Create combined mask (only detect on player region)
    combined_mask = fg_masks[len(fg_masks)//2]  # Use middle frame mask
    
    # Compute spatial derivatives (Ix, Iy)
    Ix = cv2.Sobel(volume, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(volume, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute temporal derivative (It)
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
    
    # Apply Gaussian smoothing to structure tensor
    ksize_spatial = int(6 * sigma_spatial + 1)
    if ksize_spatial % 2 == 0:
        ksize_spatial += 1
    
    # Smooth spatially for each time step
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
    
    # Compute Harris corner response for middle frame
    t_mid = volume.shape[2] // 2
    
    # Build 3x3 structure tensor for each pixel
    response = np.zeros((volume.shape[0], volume.shape[1]))
    
    for y in range(volume.shape[0]):
        for x in range(volume.shape[1]):
            if combined_mask[y, x] == 0:  # Skip background pixels
                continue
            
            # 3x3 structure tensor
            M = np.array([
                [Ix2_smooth[y, x, t_mid], IxIy_smooth[y, x, t_mid], IxIt_smooth[y, x, t_mid]],
                [IxIy_smooth[y, x, t_mid], Iy2_smooth[y, x, t_mid], IyIt_smooth[y, x, t_mid]],
                [IxIt_smooth[y, x, t_mid], IyIt_smooth[y, x, t_mid], It2_smooth[y, x, t_mid]]
            ])
            
            # Harris corner response: det(M) - k * trace(M)^2
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            response[y, x] = det_M - 0.04 * (trace_M ** 2)
    
    # Find local maxima
    response_thresh = response.copy()
    response_thresh[response < threshold * response.max()] = 0
    
    # Non-maximum suppression
    kernel_size = 5
    dilated = cv2.dilate(response_thresh, np.ones((kernel_size, kernel_size)))
    local_maxima = (response_thresh == dilated) & (response_thresh > 0)
    
    # Extract STIP coordinates
    stips = []
    y_coords, x_coords = np.where(local_maxima)
    for y, x in zip(y_coords, x_coords):
        stips.append((x, y, t_mid, response[y, x]))
    
    return stips

# ---------------------------
# PROCESS VIDEO WITH 3D HARRIS STIP
# ---------------------------
def process_video_with_stip(video_path, class_name, video_name, crop_percent=0.15, temporal_window=5):
    """
    Process entire video with 3D Harris STIP detection
    """
    print(f"🎥 Processing: {class_name}/{video_name}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Total frames: {total_frames}, FPS: {fps}")
    print(f"✂️ Cropping {crop_percent*100}% from each edge")
    print(f"⏱️  Temporal window: {temporal_window} frames")
    
    # Create MOG2 background subtractor
    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    frame_buffer = []
    fg_mask_buffer = []
    frame_count = 0
    
    print("\n▶️ Playing video with STIP detection... Press 'q' to quit, 'p' to pause/resume\n")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video finished!")
                break
            
            frame_count += 1
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Crop to center region
            crop_h = int(h * crop_percent)
            crop_w = int(w * crop_percent)
            cropped_frame = frame[crop_h:h-crop_h, crop_w:w-crop_w]
            
            # Apply background subtraction
            fg_mask = mog.apply(cropped_frame)
            fg_mask_clean = get_largest_component(fg_mask)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            
            # Maintain temporal buffer
            frame_buffer.append(gray)
            fg_mask_buffer.append(fg_mask_clean)
            
            if len(frame_buffer) > temporal_window:
                frame_buffer.pop(0)
                fg_mask_buffer.pop(0)
            
            # Detect STIPs when buffer is full
            stips = []
            if len(frame_buffer) == temporal_window:
                stips = detect_harris_3d_stip(frame_buffer, fg_mask_buffer)
            
            # Visualize STIPs
            stip_frame = cropped_frame.copy()
            for x, y, t, response in stips:
                # Draw STIP with smaller size
                size = int(2 + min(response * 500000, 3))
                cv2.circle(stip_frame, (x, y), size, (0, 255, 0), -1)
                cv2.circle(stip_frame, (x, y), size + 1, (0, 255, 255), 1)
            
            # Convert mask to BGR
            fg_mask_bgr = cv2.cvtColor(fg_mask_clean, cv2.COLOR_GRAY2BGR)
            
            # Create display
            display = np.hstack([cropped_frame, fg_mask_bgr, stip_frame])
            
            # Add text overlay
            cv2.putText(display, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"{class_name} - STIPs: {len(stips)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Original | Foreground | 3D Harris STIPs', display)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️ Stopped by user")
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("⏸️ Paused - Press 'p' to resume")
            else:
                print("▶️ Resumed")
    
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    video_path, class_name, video_name = get_random_video()
    
    if video_path:
        process_video_with_stip(video_path, class_name, video_name)
    else:
        print("⚠️ No video found!")
