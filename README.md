# ArUco Pose Estimation

This document provides an overview of the ArUco Pose Estimation project, a tool designed for real-time tracking and pose estimation of ArUco markers. The system is built with Python and leverages computer vision techniques to deliver accurate and efficient performance.

### Project Purpose

The primary goal of this project is to offer a straightforward and effective solution for tracking ArUco markers and estimating their 3D pose relative to a camera. This has applications in robotics, augmented reality, and various other fields where precise object tracking is required.

### Key Features

- **ArUco Marker Generation**: Utility for creating custom ArUco markers for testing and application.
- **Camera Calibration**: A guided process to calibrate your camera, ensuring accurate pose estimation by correcting lens distortion.
- **Real-time Marker Detection**: High-speed detection of ArUco markers in a live video stream.
- **3D Pose Estimation**: Calculation of the 3D position and orientation of each detected marker.
- **Modular and Extensible**: The codebase is organized into distinct modules for easy understanding and extension.

### Dependencies

The project requires the following Python libraries:

- `numpy`
- `opencv-python`
- `scipy`

These dependencies are listed in the `requirements.txt` file.

### Installation

To get started, clone the repository and install the required packages:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/UpayanChatterjee/aruco-pose-estimation.git
   cd aruco-pose-estimation
   ```
2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

### Usage Guide

The project includes several examples to demonstrate its functionality.

#### 1. Generating ArUco Markers

You can generate ArUco markers for printing and use in your scenes. The `generate_markers.py` script creates individual markers and a sheet with multiple markers.

```sh
python examples/generate_markers.py
```

The generated markers will be saved in the `data/generated_markers` directory.

#### 2. Camera Calibration

Accurate pose estimation requires a calibrated camera. The `camera_calibration_example.py` script guides you through this process. You will need a checkerboard pattern to perform the calibration.

```sh
python examples/camera_calibration_example.py
```

Follow the on-screen instructions to capture images of the checkerboard at various angles. The calibration parameters will be saved to `data/camera_calibration.json`.

#### 3. Basic Marker Detection

To test basic marker detection without pose estimation, run the `basic_detection.py` script. This will open a camera feed and highlight any detected ArUco markers.

```sh
python examples/basic_detection.py
```

#### 4. Full Pose Estimation

The `full_pose_estimation.py` script runs the complete pipeline, including real-time marker detection and 3D pose estimation. It will first attempt to load the camera calibration file. If the file is not found, it will initiate the interactive calibration process.

```sh
python examples/full_pose_estimation.py
```

The script will display a live video feed with coordinate axes drawn on each detected marker, indicating its 3D position and orientation.

### Project Structure

The repository is organized as follows:

- `src/`: Contains the core source code for marker detection, camera calibration, and pose estimation.
- `examples/`: Includes example scripts demonstrating how to use the different components of the project.
- `data/`: Stores data files, such as camera calibration parameters and generated markers.
- `calibration_images/`: A directory for storing images used in the camera calibration process.
- `requirements.txt`: A list of Python dependencies for the project.

This structure separates the core logic from the example implementations, making the project easier to navigate and maintain.
