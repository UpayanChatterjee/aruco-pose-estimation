"""
Real-time ArUCO Tag Tracking and Pose Estimation System

This module integrates all components for real-time ArUCO tag detection and pose estimation,
combining camera calibration, marker detection, and pose estimation algorithms.
"""

import numpy as np
import cv2
import time
import json
import os
from typing import List, Tuple, Optional, Dict
import argparse

from camera_calibration import CameraCalibration
from aruco_detection import ArUCODetector
from pose_estimation import PoseEstimator


class ArUCOTracker:
    """
    Main class for real-time ArUCO tag tracking and pose estimation.
    """
    
    def __init__(self, calibration_file: Optional[str] = None, marker_size: float = 0.05):
        """
        Initialize the ArUCO tracker.
        
        Args:
            calibration_file: Path to camera calibration file
            marker_size: Real-world size of ArUCO markers in meters
        """
        self.marker_size = marker_size
        self.calibration_file = calibration_file
        
        # Initialize components
        self.camera_calibration = CameraCalibration()
        self.aruco_detector = ArUCODetector()
        self.pose_estimator = None
        
        # Tracking state
        self.is_calibrated = False
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection history for smoothing
        self.detection_history = {}
        self.max_history_length = 5
        
        # Load calibration if available
        if calibration_file and os.path.exists(calibration_file):
            self.load_camera_calibration(calibration_file)
    
    def load_camera_calibration(self, calibration_file: str) -> bool:
        """
        Load camera calibration from file.
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            True if successful, False otherwise
        """
        if self.camera_calibration.load_calibration(calibration_file):
            self.camera_matrix, self.distortion_coeffs = self.camera_calibration.get_calibration_params()
            
            if self.camera_matrix is not None and self.distortion_coeffs is not None:
                self.pose_estimator = PoseEstimator(self.camera_matrix, self.distortion_coeffs)
                self.is_calibrated = True
                print("Camera calibration loaded successfully!")
                return True
        
        print("Failed to load camera calibration!")
        return False
    
    def calibrate_camera_interactive(self, camera_index: int = 0) -> bool:
        """
        Interactive camera calibration process.
        
        Args:
            camera_index: Camera device index
            
        Returns:
            True if successful, False otherwise
        """
        print("Starting interactive camera calibration...")
        print("Instructions:")
        print("1. Print a checkerboard pattern (9x6 internal corners)")
        print("2. Hold the checkerboard in front of the camera")
        print("3. Press SPACE to capture calibration images")
        print("4. Capture at least 15-20 images from different angles")
        print("5. Press ESC when done capturing")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        calibration_images = []
        captured_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display current frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captured: {captured_count} images", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: Capture, ESC: Done, Need 15+ images", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Try to detect checkerboard
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_chess, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            
            if ret_chess:
                cv2.drawChessboardCorners(display_frame, (9, 6), corners, ret_chess)
                cv2.putText(display_frame, "Checkerboard detected!", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ret_chess:  # Space key and checkerboard detected
                calibration_images.append(frame.copy())
                captured_count += 1
                print(f"Captured image {captured_count}")
            elif key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_count < 10:
            print(f"Error: Only {captured_count} images captured. Need at least 10.")
            return False
        
        # Save calibration images to temporary directory
        temp_dir = "../calibration_images_temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, img in enumerate(calibration_images):
            cv2.imwrite(os.path.join(temp_dir, f"calib_{i:02d}.jpg"), img)
        
        # Perform calibration
        print("Performing camera calibration...")
        if self.camera_calibration.calibrate_from_images(temp_dir):
            # Save calibration
            calibration_file = "../data/camera_calibration.json"
            self.camera_calibration.save_calibration(calibration_file)
            
            # Load calibration parameters
            self.camera_matrix, self.distortion_coeffs = self.camera_calibration.get_calibration_params()
            self.pose_estimator = PoseEstimator(self.camera_matrix, self.distortion_coeffs)
            self.is_calibrated = True
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            print("Camera calibration completed successfully!")
            return True
        else:
            print("Camera calibration failed!")
            return False
    
    def _smooth_detection(self, marker_id: int, corners: np.ndarray, 
                         rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Smooth detection using temporal filtering.
        
        Args:
            marker_id: Marker ID
            corners: Detected corners
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            Smoothed (corners, rvec, tvec)
        """
        if marker_id not in self.detection_history:
            self.detection_history[marker_id] = {
                'corners': [],
                'rvecs': [],
                'tvecs': []
            }
        
        history = self.detection_history[marker_id]
        
        # Add current detection to history
        history['corners'].append(corners)
        history['rvecs'].append(rvec)
        history['tvecs'].append(tvec)
        
        # Limit history length
        if len(history['corners']) > self.max_history_length:
            history['corners'].pop(0)
            history['rvecs'].pop(0)
            history['tvecs'].pop(0)
        
        # Calculate smoothed values using weighted average
        weights = np.linspace(0.5, 1.0, len(history['corners']))
        weights /= np.sum(weights)
        
        # Smooth corners
        smoothed_corners = np.zeros_like(corners)
        for i, (c, w) in enumerate(zip(history['corners'], weights)):
            smoothed_corners += w * c
        
        # Smooth rotation and translation vectors
        smoothed_rvec = np.zeros_like(rvec)
        smoothed_tvec = np.zeros_like(tvec)
        
        for i, (r, t, w) in enumerate(zip(history['rvecs'], history['tvecs'], weights)):
            smoothed_rvec += w * r
            smoothed_tvec += w * t
        
        return smoothed_corners, smoothed_rvec, smoothed_tvec
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for ArUCO detection and pose estimation.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detection_results)
        """
        self._update_fps()
        
        # Undistort frame if calibrated
        if self.is_calibrated:
            frame = self.camera_calibration.undistort_image(frame)
        
        # Detect ArUCO markers
        corners_list, ids_list = self.aruco_detector.detect_markers(frame)
        
        # Draw detected markers
        result_frame = self.aruco_detector.draw_detected_markers(frame, corners_list, ids_list)
        
        detection_results = []
        
        # Estimate pose for each detected marker
        if self.is_calibrated and len(corners_list) > 0:
            for corners, marker_id in zip(corners_list, ids_list):
                # Estimate pose
                rvec, tvec = self.pose_estimator.estimate_pose_single_marker(corners, self.marker_size)
                
                if rvec is not None and tvec is not None:
                    # Apply smoothing
                    smoothed_corners, smoothed_rvec, smoothed_tvec = self._smooth_detection(
                        marker_id, corners, rvec, tvec
                    )
                    
                    # Draw coordinate axes
                    result_frame = self.pose_estimator.draw_coordinate_axes(
                        result_frame, smoothed_rvec, smoothed_tvec, length=self.marker_size
                    )
                    
                    # Calculate pose accuracy
                    object_points = self.pose_estimator._create_3d_marker_points(self.marker_size)
                    accuracy = self.pose_estimator.calculate_pose_accuracy(
                        smoothed_rvec, smoothed_tvec, object_points, corners
                    )
                    
                    # Get pose information
                    pose_info = self.pose_estimator.get_pose_info(smoothed_rvec, smoothed_tvec)
                    pose_info['marker_id'] = marker_id
                    pose_info['accuracy'] = accuracy
                    pose_info['corners'] = smoothed_corners.tolist()
                    
                    detection_results.append(pose_info)
                    
                    # Display pose information on frame
                    center = np.mean(corners, axis=0).astype(int)
                    distance = pose_info['distance']
                    
                    cv2.putText(result_frame, f"ID: {marker_id}", 
                               tuple(center - [50, 40]), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    cv2.putText(result_frame, f"Dist: {distance:.2f}m", 
                               tuple(center - [50, 20]), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    cv2.putText(result_frame, f"Acc: {accuracy:.1f}px", 
                               tuple(center - [50, 0]), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
        
        # Draw status information
        status_color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
        status_text = "Calibrated" if self.is_calibrated else "Not Calibrated"
        
        cv2.putText(result_frame, f"Status: {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(result_frame, f"FPS: {self.current_fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Detected: {len(corners_list)} markers", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame, detection_results
    
    def run_live_tracking(self, camera_index: int = 0, save_video: bool = False):
        """
        Run live ArUCO tracking from camera.
        
        Args:
            camera_index: Camera device index
            save_video: Whether to save output video
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('../data/aruco_tracking.avi', fourcc, 30.0, (640, 480))
        
        print("ArUCO Real-time Tracking Started")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Calibrate camera (if not calibrated)")
        print("  's' - Save current detection data")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result_frame, detection_results = self.process_frame(frame)
                
                # Save video frame
                if save_video and video_writer is not None:
                    video_writer.write(result_frame)
                
                # Display frame
                cv2.imshow('ArUCO Real-time Tracking', result_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and not self.is_calibrated:
                    cap.release()
                    cv2.destroyAllWindows()
                    if self.calibrate_camera_interactive(camera_index):
                        cap = cv2.VideoCapture(camera_index)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                elif key == ord('s') and len(detection_results) > 0:
                    # Save detection data
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"../data/detection_data_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(detection_results, f, indent=2)
                    print(f"Detection data saved to {filename}")
        
        except KeyboardInterrupt:
            print("\nTracking interrupted by user")
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            print("Tracking session ended")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='ArUCO Real-time Tracking System')
    parser.add_argument('--calibration', '-c', type=str, 
                       default='../data/camera_calibration.json',
                       help='Path to camera calibration file')
    parser.add_argument('--marker-size', '-m', type=float, default=0.05,
                       help='Real-world size of ArUCO markers in meters')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ArUCOTracker(
        calibration_file=args.calibration,
        marker_size=args.marker_size
    )
    
    # Run live tracking
    tracker.run_live_tracking(
        camera_index=args.camera,
        save_video=args.save_video
    )


if __name__ == "__main__":
    main()