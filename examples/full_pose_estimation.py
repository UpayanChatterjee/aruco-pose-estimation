"""
Example: Full Pose Estimation

This example demonstrates complete ArUCO pose estimation with all features.
"""

import cv2
import numpy as np
import sys
import os
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from aruco_tracker import ArUCOTracker


def main():
    """Full pose estimation example."""
    print("ArUCO Pose Estimation Example")
    print("=" * 50)
    
    # Configuration
    calibration_file = "../data/camera_calibration.json"
    marker_size = 0.05  # 5cm markers
    
    # Check if calibration file exists
    if not os.path.exists(calibration_file):
        print(f"Calibration file not found: {calibration_file}")
        print("Starting interactive calibration process...")
        
        tracker = ArUCOTracker(marker_size=marker_size)
        if not tracker.calibrate_camera_interactive():
            print("Calibration failed. Exiting...")
            return
    else:
        print(f"Loading calibration from: {calibration_file}")
        tracker = ArUCOTracker(calibration_file=calibration_file, marker_size=marker_size)
    
    # Display instructions
    print("\nInstructions:")
    print("1. Print ArUCO markers (4x4 dictionary)")
    print("2. Hold markers in front of the camera")
    print("3. Watch real-time pose estimation")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save detection data")
    print("  'c' - Recalibrate camera")
    
    # Start tracking
    tracker.run_live_tracking(camera_index=0, save_video=False)


if __name__ == "__main__":
    main()