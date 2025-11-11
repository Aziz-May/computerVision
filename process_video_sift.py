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
# PROCESS VIDEO WITH SIFT
# ---------------------------
def process_video_with_sift(video_path, class_name, video_name, crop_percent=0.25):
    """
    Process entire video with SIFT keypoint detection focused on center
    """
    print(f"🎥 Processing: {class_name}/{video_name}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Total frames: {total_frames}, FPS: {fps}")
    print(f"✂️ Cropping {crop_percent*100}% from each edge to focus on center")
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    frame_count = 0
    
    print("\n▶️ Playing video... Press 'q' to quit, 'p' to pause/resume\n")
    
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
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect SIFT keypoints and descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            # Draw keypoints on the frame
            frame_with_keypoints = cv2.drawKeypoints(
                cropped_frame, 
                keypoints, 
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(0, 255, 0)
            )
            
            # Create side-by-side display
            display = np.hstack([cropped_frame, frame_with_keypoints])
            
            # Add text overlay
            cv2.putText(display, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"{class_name} - SIFT Keypoints: {len(keypoints)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the result
            cv2.imshow('Original (cropped) | SIFT Keypoints', display)
        
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
        process_video_with_sift(video_path, class_name, video_name)
    else:
        print("⚠️ No video found!")
