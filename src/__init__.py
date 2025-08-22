"""
ArUCO Pose Estimation Project

This package provides a complete implementation of ArUCO tag detection
and pose estimation built from scratch.
"""

__version__ = "1.0.0"
__author__ = "Upayan Chatterjee"
__email__ = "upayanc9@gmail.com"

from .camera_calibration import CameraCalibration
from .aruco_detection import ArUCODetector
from .pose_estimation import PoseEstimator
from .aruco_tracker import ArUCOTracker

__all__ = [
    'CameraCalibration',
    'ArUCODetector', 
    'PoseEstimator',
    'ArUCOTracker'
]