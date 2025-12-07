THETIS Action Recognition: RGB to 3D Skeletons
📝 Project Overview
This project focuses on modernizing the THETIS dataset (Tennis Shots) for Human Action Recognition.
Instead of relying on the obsolete Kinect v1 skeleton data provided with the dataset (circa 2010), we generated high-quality 3D Skeletons directly from the RGB videos using Google MediaPipe.
What we achieved:
Data Generation: We converted the raw RGB video dataset into mathematical 3D coordinate matrices.
Performance: We implemented parallel processing to handle thousands of videos efficiently.
Verification: We ensured data integrity and visualization capabilities.
⚙️ Installation & Dependencies
Crucial: MediaPipe requires a specific version of NumPy to work with OpenCV. Run this exact command:
code
Bash
pip install "numpy<2.0.0" "opencv-python==4.9.0.80" "opencv-contrib-python==4.9.0.80" mediapipe matplotlib tqdm
📂 Project Structure
Your project folder is organized as follows:
code
Text
/Computer_vision_tennis
│
├── /dataset
│   ├── /VIDEO_RGB               <-- Original THETIS videos (Input)
│   │   ├── /backhand
│   │   ├── /smash
│   │   └── ... (12 classes)
│   │
│   └── /mediapipe_skeletons3D   <-- Generated Data (Output)
│       ├── /backhand            <-- Contains .npy files
│       └── ...
│
├── rgb2skeleton.py              <-- The Generator (RGB Video -> .npy)
├── check_dataset.py             <-- The Auditor (Verifies file counts)
├── visualize_skeleton.py        <-- The Viewer (Plays 3D animations)
└── README.md
📜 Description of Scripts
1. rgb2skeleton.py (The Engine)
Function: This is the core script. It reads every RGB video from the dataset, runs MediaPipe Pose Estimation, and extracts 33-point 3D skeletons.
Key Feature: Uses Multiprocessing (utilizing multiple CPU cores) to process the dataset 5x-10x faster than standard execution.
Output: Saves .npy (NumPy) files containing coordinate matrices of shape (Frames, 33, 3).
Usage:
code
Bash
python rgb2skeleton.py
2. check_dataset.py (The Auditor)
Function: Verifies the integrity of the generated dataset. It compares the number of input videos against the number of generated skeleton files.
Key Feature: Identifies missing files or processing errors, ensuring the dataset is clean before training starts.
Usage:
code
Bash
python check_dataset.py
3. visualize_skeleton.py (The Viewer)
Function: Since .npy files are just mathematical coordinates, they cannot be opened in a video player. This script loads random files and reconstructs them as interactive 3D animations.
Key Feature: Allows you to rotate the camera in 3D space to inspect the quality of the extracted skeletons.
Usage:
code
Bash
python visualize_skeleton.py
🧠 Theory: Why did we ignore the original THETIS skeletons?
We chose to generate our own data using rgb2skeleton.py because:
Incompatible Topology: The original THETIS skeletons were recorded with Kinect v1 (20 joints), whereas modern AI (MediaPipe) tracks 33 joints with higher precision.
Real-World Application: To test this system on standard camera footage (like a phone recording), the training data must come from RGB video analysis (MediaPipe), not infrared depth sensors (Kinect).
🚀 Next Steps
Now that the mediapipe_skeletons3D dataset is ready:
Feature Extraction: Feed the .npy files into the Cylindrical Trace Transform (CTT) algorithm.
Training: Use the CTT features to train a classifier (SVM) to distinguish between tennis moves (e.g., Backhand vs. Smash).