# Real-Time Traffic Surveillance & Analysis System

A real-time computer vision system built to detect, categorize, track, and count vehicles passing through critical highway checkpoints. The pipeline features dynamic region-of-interest (ROI) extraction designed to isolate vehicle bumper zones for license plate reading workflows.

## 🚀 Key Features
* **Vehicle Detection & Grouping:** Powered by a YOLOv5 architecture optimized for high-density tracking.
* **Intelligent Local Classification:** Implements custom threshold logic to classify incoming streams into uniform local vehicle groups (*Vehicle*, *Heavy Vehicle*, *Two-Wheeler*).
* **Surveillance Counting Line:** Tracks and increments total volumetric flow smoothly using cross-boundary centerpoint logic.
* **Automated Bumper Extraction:** Crops out the bottom half of localized vehicle blocks to extract high-resolution regions containing license plates.

## 🛠️ System Stack
* **Language:** Python 3.x
* **Core Libraries:** OpenCV (`cv2`), PyTorch (`torch`), EasyOCR, Pandas
*
