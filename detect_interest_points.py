import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------
# PATH CONFIGURATION
# ---------------------------
THETIS_PATH = r"./dataset/"
rgb_folder = os.path.join(THETIS_PATH, "VIDEO_RGB")

classes = ['backhand', 'backhand_slice', 'backhand_volley', 'backhand2hands', 
           'flat_service', 'forehand_flat', 'forehand_openstands', 'forehand_slice', 
           'forehand_volley', 'kick_service', 'slice_service', 'smash']

# ---------------------------
# HARRIS CORNER DETECTION
# ---------------------------
def detect_harris_corners(frame, block_size=2, ksize=3, k=0.04, crop_percent=0.2):
    """
    Detect Harris corners in a frame with center cropping
    """
    h, w = frame.shape[:2]
    
    # Crop to center region (remove borders)
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    cropped_frame = frame[crop_h:h-crop_h, crop_w:w-crop_w]
    
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # Harris corner detection
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Dilate for marking the corners
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value
    frame_with_corners = cropped_frame.copy()
    frame_with_corners[dst > 0.01 * dst.max()] = [0, 0, 255]  # Red dots for corners
    
    return frame_with_corners, dst, cropped_frame

# ---------------------------
# MOG BACKGROUND SUBTRACTION
# ---------------------------
def apply_mog_background_subtraction(video_path, frame_number=30, crop_percent=0.2):
    """
    Apply MOG2 background subtraction to detect moving objects (returns binary mask)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Create MOG2 background subtractor
    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    original_frame = None
    fg_mask = None
    
    # Process frames to build background model
    for i in range(frame_number + 10):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop to center
        h, w = frame.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        cropped_frame = frame[crop_h:h-crop_h, crop_w:w-crop_w]
        
        # Apply background subtraction
        fg_mask = mog.apply(cropped_frame)
        
        # Save the frame we want to analyze
        if i == frame_number:
            original_frame = cropped_frame.copy()
    
    cap.release()
    
    return original_frame, fg_mask

# ---------------------------
# PROCESS ONE VIDEO PER CLASS
# ---------------------------
def process_dataset():
    """
    Process one video from each class
    """
    results = []
    
    print("🔍 Processing videos - Harris Corner Detection + MOG Background Subtraction\n")
    
    for class_name in classes:
        class_folder = os.path.join(rgb_folder, class_name)
        
        if os.path.exists(class_folder):
            videos = [v for v in os.listdir(class_folder) if v.endswith('.avi')]
            
            if videos:
                # Take the first video
                video_path = os.path.join(class_folder, videos[0])
                print(f"📹 Processing: {class_name}/{videos[0]}")
                
                # Read a frame from the middle of the video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid_frame = total_frames // 2
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Apply Harris corner detection (with cropping)
                    frame_harris, harris_response, cropped_original = detect_harris_corners(frame)
                    
                    # Apply MOG background subtraction (with cropping)
                    frame_mog, fg_mask = apply_mog_background_subtraction(video_path, mid_frame)
                    
                    # Count corners
                    corner_count = np.sum(harris_response > 0.01 * harris_response.max())
                    
                    # Count foreground pixels
                    fg_count = np.sum(fg_mask > 0)
                    
                    results.append({
                        'class': class_name,
                        'video': videos[0],
                        'frame': cropped_original,
                        'harris': frame_harris,
                        'harris_response': harris_response,
                        'mog_frame': frame_mog,
                        'fg_mask': fg_mask,
                        'corner_count': corner_count,
                        'fg_pixel_count': fg_count
                    })
                    
                    print(f"  ✓ Harris corners detected: {corner_count}")
                    print(f"  ✓ Foreground pixels (MOG): {fg_count}\n")
    
    return results

# ---------------------------
# VISUALIZE RESULTS
# ---------------------------
def visualize_results(results):
    """
    Display each result in a separate window
    """
    for result in results:
        # Get the images
        frame = result['frame']
        harris = result['harris']
        fg_mask = result['fg_mask']
        
        # Convert mask to BGR for stacking
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        # Stack horizontally: Original | Harris | MOG Binary Mask
        combined = np.hstack([frame, harris, fg_mask_bgr])
        
        # Display in separate window
        window_name = f"{result['class']} - Corners: {result['corner_count']:.0f} | FG: {result['fg_pixel_count']:.0f}"
        cv2.imshow(window_name, combined)
    
    print("\n✅ Displaying all 12 windows. Press any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    if os.path.exists(rgb_folder):
        results = process_dataset()
        
        if results:
            print("\n" + "="*60)
            print("📊 SUMMARY")
            print("="*60)
            for result in results:
                print(f"{result['class']:20s} | Corners: {result['corner_count']:6.0f} | FG Pixels: {result['fg_pixel_count']:8.0f}")
            
            # Visualize all results
            visualize_results(results)
        else:
            print("⚠️ No results to display")
    else:
        print("⚠️ RGB folder not found!")
