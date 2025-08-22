"""
Example: Camera Calibration

This example demonstrates the camera calibration process.
"""

import cv2
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from camera_calibration import CameraCalibration, create_calibration_images


def main():
    """Camera calibration example."""
    print("Camera Calibration Example")
    print("=" * 50)
    
    # Create calibration object
    calibrator = CameraCalibration(checkerboard_size=(9, 6), square_size=2.5)
    
    # Option 1: Capture calibration images
    choice = input("Do you want to capture new calibration images? (y/n): ")
    
    if choice.lower() == 'y':
        print("\nCapturing calibration images...")
        print("Instructions:")
        print("1. Print a 9x6 checkerboard pattern")
        print("2. Hold it in front of the camera at different angles")
        print("3. Press SPACE to capture, ESC when done")
        
        calibration_dir = "../calibration_images"
        create_calibration_images(calibration_dir, num_images=20)
    
    # Option 2: Use existing images
    calibration_dir = "../calibration_images"
    
    if not os.path.exists(calibration_dir):
        print(f"Error: Calibration directory {calibration_dir} not found!")
        print("Please capture calibration images first.")
        return
    
    # Perform calibration
    print(f"\nPerforming calibration using images from {calibration_dir}...")
    
    if calibrator.calibrate_from_images(calibration_dir):
        # Save calibration results
        output_file = "../data/camera_calibration.json"
        calibrator.save_calibration(output_file)
        
        print(f"\nCalibration successful!")
        print(f"Results saved to: {output_file}")
        
        # Display calibration parameters
        camera_matrix, dist_coeffs = calibrator.get_calibration_params()
        print(f"\nCamera Matrix:")
        print(camera_matrix)
        print(f"\nDistortion Coefficients:")
        print(dist_coeffs.flatten())
        print(f"\nReprojection Error: {calibrator.calibration_error:.3f} pixels")
        
    else:
        print("Calibration failed!")


if __name__ == "__main__":
    main()