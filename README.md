# ArUCO Tag Detection and Pose Estimation

A comprehensive Python implementation of ArUCO tag detection and relative pose estimation built from scratch, without relying on built-in OpenCV ArUCO functions.

## 🎯 Project Overview

This project implements a complete ArUCO tag detection and pose estimation system that includes:

- **Custom ArUCO Detection**: Corner detection, thresholding, contour analysis
- **Camera Calibration**: From-scratch implementation for enhanced accuracy
- **Pose Estimation**: Homography transformations, perspective transformation, and triangulation
- **Real-time Tracking**: Live camera feed processing with pose visualization

## 🏗️ Architecture

```
aruco_pose_estimation/
├── src/
│   ├── camera_calibration.py     # Camera calibration module
│   ├── aruco_detection.py        # ArUCO marker detection
│   ├── pose_estimation.py        # Pose estimation algorithms
│   └── aruco_tracker.py          # Main tracking system
├── examples/
│   ├── basic_detection.py        # Basic detection demo
│   ├── camera_calibration_example.py
│   ├── full_pose_estimation.py   # Complete system demo
│   └── generate_markers.py       # ArUCO marker generator
├── data/                         # Calibration and output data
├── calibration_images/           # Camera calibration images
└── README.md
```

## 🚀 Features

### Core Implementations

1. **Camera Calibration from Scratch**

   - Custom checkerboard corner detection
   - Subpixel refinement algorithms
   - Distortion parameter estimation
   - Calibration accuracy metrics

2. **ArUCO Detection Pipeline**

   - Adaptive thresholding for robust detection
   - Custom contour analysis and filtering
   - Quadrilateral approximation algorithms
   - Corner ordering and refinement

3. **Pose Estimation Algorithms**

   - Homography-based pose estimation
   - Perspective-n-Point (PnP) solving
   - Iterative pose refinement
   - Triangulation for stereo setups

4. **Real-time Tracking System**
   - Temporal smoothing for stable pose
   - Performance optimization
   - Interactive calibration interface
   - Data logging and visualization

## 📋 Requirements

```bash
pip install numpy opencv-python scipy
```

### System Requirements

- Python 3.7+
- OpenCV 4.0+
- NumPy 1.19+
- SciPy 1.5+
- Webcam or camera device

## 🔧 Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd aruco_pose_estimation
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Generate test markers:**

```bash
cd examples
python generate_markers.py
```

## 🎮 Quick Start

### 1. Generate ArUCO Markers

```bash
cd examples
python generate_markers.py
```

This creates printable ArUCO markers in `data/generated_markers/`.

### 2. Basic Detection Test

```bash
python basic_detection.py
```

Tests marker detection without pose estimation.

### 3. Camera Calibration

```bash
python camera_calibration_example.py
```

Interactive calibration process:

1. Print a 9x6 checkerboard pattern
2. Capture 15-20 images from different angles
3. Automatic calibration parameter computation

### 4. Full Pose Estimation

```bash
python full_pose_estimation.py
```

Complete system with real-time pose estimation and visualization.

## 📖 Detailed Usage

### Camera Calibration

```python
from camera_calibration import CameraCalibration

# Initialize calibrator
calibrator = CameraCalibration(checkerboard_size=(9, 6), square_size=2.5)

# Perform calibration
if calibrator.calibrate_from_images("calibration_images/"):
    calibrator.save_calibration("camera_calibration.json")
```

### ArUCO Detection

```python
from aruco_detection import ArUCODetector

# Initialize detector
detector = ArUCODetector()

# Detect markers in image
corners_list, ids_list = detector.detect_markers(image)

# Draw results
result_image = detector.draw_detected_markers(image, corners_list, ids_list)
```

### Pose Estimation

```python
from pose_estimation import PoseEstimator

# Initialize with camera parameters
pose_estimator = PoseEstimator(camera_matrix, distortion_coeffs)

# Estimate pose for detected marker
rvec, tvec = pose_estimator.estimate_pose_single_marker(corners, marker_size)

# Draw coordinate axes
result_image = pose_estimator.draw_coordinate_axes(image, rvec, tvec)
```

### Complete Tracking System

```python
from aruco_tracker import ArUCOTracker

# Initialize tracker
tracker = ArUCOTracker(
    calibration_file="camera_calibration.json",
    marker_size=0.05  # 5cm markers
)

# Run live tracking
tracker.run_live_tracking(camera_index=0)
```

## ⚙️ Configuration

### Detection Parameters

Adjust detection sensitivity in `aruco_detection.py`:

```python
self.min_marker_area = 100        # Minimum marker area
self.max_marker_area = 50000      # Maximum marker area
self.min_contour_length = 16      # Minimum contour perimeter
```

### Pose Estimation Settings

Configure pose estimation in `pose_estimation.py`:

```python
# Marker size in meters
marker_size = 0.05  # 5cm markers

# Pose refinement iterations
max_iterations = 10
```

### Tracking Parameters

Modify tracking behavior in `aruco_tracker.py`:

```python
self.max_history_length = 5       # Temporal smoothing window
self.marker_size = 0.05           # Real-world marker size
```

## 📊 Performance Metrics

The system provides several accuracy metrics:

1. **Calibration Accuracy**: Reprojection error in pixels
2. **Detection Rate**: Markers detected per frame
3. **Pose Accuracy**: Reprojection error for pose estimation
4. **Frame Rate**: Real-time processing speed

### Expected Performance

- **Calibration Error**: < 0.5 pixels (good), < 0.3 pixels (excellent)
- **Detection Range**: 0.1m - 5m depending on marker size
- **Pose Accuracy**: ±2mm translation, ±1° rotation (optimal conditions)
- **Frame Rate**: 15-30 FPS on standard hardware

## 🔬 Algorithm Details

### Camera Calibration Process

1. **Corner Detection**: Harris corner detection with subpixel refinement
2. **Pattern Recognition**: Checkerboard pattern validation
3. **Parameter Estimation**: Intrinsic matrix and distortion coefficients
4. **Optimization**: Non-linear refinement using Levenberg-Marquardt

### ArUCO Detection Pipeline

1. **Preprocessing**: Adaptive thresholding and noise reduction
2. **Contour Detection**: External contour extraction with area filtering
3. **Shape Analysis**: Quadrilateral approximation and validation
4. **Corner Refinement**: Subpixel corner localization
5. **Pattern Extraction**: Perspective transformation and bit extraction
6. **ID Recognition**: Dictionary matching with rotation handling

### Pose Estimation Methods

1. **PnP Solver**: Perspective-n-Point problem solution
2. **Homography Decomposition**: Alternative pose extraction method
3. **Iterative Refinement**: Pose optimization using reprojection error
4. **Temporal Smoothing**: Multi-frame pose stabilization

## 🎯 Applications

This implementation is suitable for:

- **Robotics**: Robot localization and navigation
- **Augmented Reality**: Object tracking and overlay
- **Industrial Automation**: Part positioning and quality control
- **Research**: Computer vision algorithm development
- **Education**: Learning pose estimation concepts

## 🐛 Troubleshooting

### Common Issues

1. **Poor Detection Rate**

   - Ensure good lighting conditions
   - Check marker print quality
   - Adjust detection thresholds
   - Verify camera focus

2. **Calibration Failure**

   - Capture more calibration images (15-20 minimum)
   - Use different angles and distances
   - Ensure checkerboard is flat and high-contrast
   - Check checkerboard size parameters

3. **Unstable Pose Estimation**

   - Increase temporal smoothing window
   - Improve camera calibration
   - Use larger markers
   - Reduce camera motion

4. **Low Frame Rate**
   - Reduce image resolution
   - Optimize detection parameters
   - Use faster hardware
   - Disable unnecessary features

### Debug Mode

Enable debug output by modifying the source files:

```python
# Add to any module for detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Technical References

This implementation is based on the following research:

1. **ArUCO Markers**: Garrido-Jurado et al. "Automatic generation and detection of highly reliable fiducial markers under occlusion"
2. **Camera Calibration**: Zhang, Z. "A flexible new technique for camera calibration"
3. **Pose Estimation**: Lepetit et al. "EPnP: An accurate O(n) solution to the PnP problem"
4. **Homography**: Hartley & Zisserman "Multiple View Geometry in Computer Vision"

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Enhanced marker dictionaries
- Robust pose estimation algorithms
- Performance optimizations
- Additional calibration patterns
- Stereo vision support

## 📄 License

This project is provided for educational and research purposes. Please cite appropriately if used in academic work.

## 🔗 Related Projects

- OpenCV ArUCO module (reference implementation)
- AprilTag detection library
- SLAM frameworks using fiducial markers

---

**Note**: This implementation prioritizes educational value and algorithmic understanding over pure performance. For production applications, consider using optimized libraries like OpenCV's ArUCO module.
