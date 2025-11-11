import os
import cv2
import numpy as np
import random
from scipy.spatial import Delaunay

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
    """
    Keep only the largest connected component (the player), remove smaller objects
    """
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    
    if num_labels <= 1:  # Only background
        return np.zeros_like(mask)
    
    # Find the largest component (excluding background which is label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only the largest component
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    
    return largest_mask

# ---------------------------
# DETECT HARRIS SKELETON WITH FOREGROUND MASK
# ---------------------------
def detect_harris_skeleton(frame, fg_mask, max_corners=50, quality_level=0.01, min_distance=20):
    """
    Detect Harris corners only in the foreground region and connect them to form a skeleton structure
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Keep only the largest object (the player)
    fg_mask_clean = get_largest_component(fg_mask)
    
    # Use goodFeaturesToTrack (Harris) only on the foreground
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=fg_mask_clean,
        useHarrisDetector=True,
        k=0.04
    )
    
    skeleton_frame = frame.copy()
    
    if corners is not None and len(corners) > 3:
        corners = corners.reshape(-1, 2)
        
        # Draw the corner points
        for corner in corners:
            x, y = corner.astype(int)
            cv2.circle(skeleton_frame, (x, y), 5, (0, 255, 0), -1)
        
        # Create Delaunay triangulation to connect nearby points
        try:
            tri = Delaunay(corners)
            
            # Draw the triangulation edges
            for simplex in tri.simplices:
                for i in range(3):
                    p1 = corners[simplex[i]].astype(int)
                    p2 = corners[simplex[(i+1)%3]].astype(int)
                    
                    # Calculate distance between points
                    dist = np.linalg.norm(p1 - p2)
                    
                    # Only draw lines for relatively close points (to avoid long lines)
                    if dist < 100:
                        cv2.line(skeleton_frame, tuple(p1), tuple(p2), (0, 255, 255), 2)
        except:
            pass
        
        num_points = len(corners)
    else:
        num_points = 0
    
    return skeleton_frame, fg_mask_clean, num_points

# ---------------------------
# PROCESS VIDEO WITH HARRIS SKELETON
# ---------------------------
def process_video_with_harris_skeleton(video_path, class_name, video_name, crop_percent=0.15):
    """
    Process entire video with background subtraction + Harris skeleton detection focused on center
    """
    print(f"🎥 Processing: {class_name}/{video_name}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Total frames: {total_frames}, FPS: {fps}")
    print(f"✂️ Cropping {crop_percent*100}% from each edge to focus on center")
    
    # Create MOG2 background subtractor
    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    frame_count = 0
    
    print("\n▶️ Playing video... Press 'q' to quit, 'p' to pause/resume")
    print("   Use '+' to increase corners, '-' to decrease corners\n")
    
    paused = False
    max_corners = 50
    
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
            
            # Apply background subtraction to get foreground mask
            fg_mask = mog.apply(cropped_frame)
            
            # Detect Harris skeleton only on foreground (largest object)
            skeleton_frame, fg_mask_clean, num_points = detect_harris_skeleton(cropped_frame, fg_mask, max_corners=max_corners)
            
            # Convert mask to BGR for display
            fg_mask_bgr = cv2.cvtColor(fg_mask_clean, cv2.COLOR_GRAY2BGR)
            
            # Create three-panel display: Original | Foreground Mask | Skeleton
            display = np.hstack([cropped_frame, fg_mask_bgr, skeleton_frame])
            
            # Add text overlay
            cv2.putText(display, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"{class_name} - Points: {num_points} (max: {max_corners})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the result
            cv2.imshow('Original | Foreground Mask | Harris Skeleton', display)
        
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
        elif key == ord('+') or key == ord('='):
            max_corners = min(max_corners + 10, 200)
            print(f"🔼 Max corners: {max_corners}")
        elif key == ord('-') or key == ord('_'):
            max_corners = max(max_corners - 10, 10)
            print(f"🔽 Max corners: {max_corners}")
    
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    video_path, class_name, video_name = get_random_video()
    
    if video_path:
        process_video_with_harris_skeleton(video_path, class_name, video_name)
    else:
        print("⚠️ No video found!")
