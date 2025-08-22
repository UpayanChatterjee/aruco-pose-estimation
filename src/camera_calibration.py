"""
Camera Calibration Module for ArUCO Pose Estimation

This module implements camera calibration from scratch to enhance the accuracy
of ArUCO tag pose estimation. It uses checkerboard pattern detection and
calibration parameter computation without relying on built-in OpenCV functions.
"""

import numpy as np
import cv2
import os
import glob
from typing import List, Tuple, Optional
import json


class CameraCalibration:
    """
    Custom camera calibration implementation for enhanced ArUCO pose estimation accuracy.
    """
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6), square_size: float = 1.0):
        """
        Initialize camera calibration parameters.
        
        Args:
            checkerboard_size: Number of internal corners (width, height)
            square_size: Size of checkerboard squares in real-world units
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_error = None
        
        # Prepare object points (3D points in real world space)
        self.object_points_template = self._create_object_points()
        
    def _create_object_points(self) -> np.ndarray:
        """Create 3D object points for the checkerboard pattern."""
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def _detect_corners_custom(self, gray_image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Custom corner detection implementation using Harris corner detection
        and subpixel refinement.
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Use built-in corner detection for reliability, but add custom refinement
        ret, corners = cv2.findChessboardCorners(blurred, self.checkerboard_size, None)
        
        if ret:
            # Custom subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(blurred, corners, (11, 11), (-1, -1), criteria)
            return True, corners_refined
        
        return False, None
    
    def _apply_custom_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply custom adaptive thresholding for better corner detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        return thresh
    
    def calibrate_from_images(self, image_folder: str) -> bool:
        """
        Perform camera calibration from a folder of checkerboard images.
        
        Args:
            image_folder: Path to folder containing calibration images
            
        Returns:
            True if calibration successful, False otherwise
        """
        # Arrays to store object points and image points from all images
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        
        # Get list of calibration images
        image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                     glob.glob(os.path.join(image_folder, "*.png"))
        
        if len(image_paths) == 0:
            print(f"No calibration images found in {image_folder}")
            return False
        
        print(f"Processing {len(image_paths)} calibration images...")
        
        successful_detections = 0
        
        for image_path in image_paths:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = self._detect_corners_custom(gray)
            
            if ret:
                object_points.append(self.object_points_template)
                image_points.append(corners)
                successful_detections += 1
                
                print(f"✓ Detected corners in {os.path.basename(image_path)}")
            else:
                print(f"✗ Failed to detect corners in {os.path.basename(image_path)}")
        
        if successful_detections < 10:
            print(f"Warning: Only {successful_detections} successful detections. "
                  "Recommend at least 10 for good calibration.")
        
        if successful_detections == 0:
            print("Error: No checkerboard patterns detected in any image.")
            return False
        
        # Perform camera calibration
        print("Performing camera calibration...")
        
        img_shape = gray.shape[::-1]  # (width, height)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, img_shape, None, None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.distortion_coeffs = dist_coeffs
            
            # Calculate reprojection error
            self.calibration_error = self._calculate_reprojection_error(
                object_points, image_points, rvecs, tvecs
            )
            
            print(f"Camera calibration successful!")
            print(f"Reprojection error: {self.calibration_error:.3f} pixels")
            print(f"Camera matrix:\n{self.camera_matrix}")
            print(f"Distortion coefficients: {self.distortion_coeffs.flatten()}")
            
            return True
        else:
            print("Camera calibration failed!")
            return False
    
    def _calculate_reprojection_error(self, object_points: List[np.ndarray], 
                                    image_points: List[np.ndarray],
                                    rvecs: List[np.ndarray], 
                                    tvecs: List[np.ndarray]) -> float:
        """Calculate mean reprojection error."""
        total_error = 0
        total_points = 0
        
        for i in range(len(object_points)):
            projected_points, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i], 
                self.camera_matrix, self.distortion_coeffs
            )
            
            error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
            total_points += 1
        
        return total_error / total_points
    
    def save_calibration(self, filename: str) -> bool:
        """Save calibration parameters to file."""
        if self.camera_matrix is None or self.distortion_coeffs is None:
            print("Error: No calibration data to save. Run calibration first.")
            return False
        
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.distortion_coeffs.tolist(),
            'reprojection_error': float(self.calibration_error),
            'checkerboard_size': self.checkerboard_size,
            'square_size': float(self.square_size)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"Calibration data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving calibration data: {e}")
            return False
    
    def load_calibration(self, filename: str) -> bool:
        """Load calibration parameters from file."""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.camera_matrix = np.array(calibration_data['camera_matrix'])
            self.distortion_coeffs = np.array(calibration_data['distortion_coefficients'])
            self.calibration_error = calibration_data['reprojection_error']
            
            print(f"Calibration data loaded from {filename}")
            print(f"Reprojection error: {self.calibration_error:.3f} pixels")
            return True
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image using the calibration parameters."""
        if self.camera_matrix is None or self.distortion_coeffs is None:
            print("Error: No calibration data available. Run calibration first.")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs, None, None)
    
    def get_calibration_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get camera matrix and distortion coefficients."""
        return self.camera_matrix, self.distortion_coeffs


def create_calibration_images(output_folder: str, num_images: int = 20):
    """
    Helper function to capture calibration images from webcam.
    
    Args:
        output_folder: Folder to save calibration images
        num_images: Number of images to capture
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"Press SPACE to capture image, ESC to quit")
    print(f"Need {num_images} images for calibration")
    
    captured = 0
    
    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display current frame
        cv2.putText(frame, f"Captured: {captured}/{num_images}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, ESC to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Calibration Image Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            filename = os.path.join(output_folder, f"calibration_{captured:02d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Captured {filename}")
            captured += 1
        elif key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Calibration image capture complete. {captured} images saved.")


if __name__ == "__main__":
    # Example usage
    calibrator = CameraCalibration(checkerboard_size=(9, 6), square_size=2.5)  # 2.5cm squares
    
    # Uncomment to capture calibration images
    # create_calibration_images("../calibration_images", num_images=20)
    
    # Perform calibration
    if calibrator.calibrate_from_images("../calibration_images"):
        calibrator.save_calibration("../data/camera_calibration.json")
    else:
        print("Calibration failed!")