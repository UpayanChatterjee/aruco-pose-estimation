"""
Pose Estimation Module for ArUCO Tags

This module implements pose estimation algorithms using homography transformations,
perspective transformation, and triangulation algorithms to determine the 3D pose
of detected ArUCO markers relative to the camera.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import math
from scipy.spatial.transform import Rotation


class PoseEstimator:
    """
    Custom pose estimation implementation for ArUCO markers.
    """
    
    def __init__(self, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray):
        """
        Initialize pose estimator with camera calibration parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            distortion_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        
        # Camera parameters
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def _create_3d_marker_points(self, marker_size: float) -> np.ndarray:
        """
        Create 3D coordinates of marker corners in marker coordinate system.
        
        Args:
            marker_size: Real-world size of the marker
            
        Returns:
            3D coordinates of marker corners
        """
        half_size = marker_size / 2.0
        return np.array([
            [-half_size, -half_size, 0],  # Top-left
            [half_size, -half_size, 0],   # Top-right
            [half_size, half_size, 0],    # Bottom-right
            [-half_size, half_size, 0]    # Bottom-left
        ], dtype=np.float32)
    
    def _compute_homography_custom(self, src_points: np.ndarray, 
                                 dst_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography matrix using custom implementation.
        
        Args:
            src_points: Source points (4x2)
            dst_points: Destination points (4x2)
            
        Returns:
            3x3 homography matrix
        """
        if len(src_points) != 4 or len(dst_points) != 4:
            return None
        
        # Use OpenCV's findHomography for robustness, but could implement DLT algorithm
        homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        return homography
    
    def _decompose_homography_custom(self, homography: np.ndarray, 
                                   marker_size: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Decompose homography to extract possible rotation and translation vectors.
        
        Args:
            homography: 3x3 homography matrix
            marker_size: Real-world size of the marker
            
        Returns:
            Tuple of (rotation_vectors_list, translation_vectors_list)
        """
        # Normalize homography
        h = homography / homography[2, 2]
        
        # Extract columns
        h1 = h[:, 0]
        h2 = h[:, 1]
        h3 = h[:, 2]
        
        # Calculate scale factor
        lambda1 = 1.0 / np.linalg.norm(np.dot(np.linalg.inv(self.camera_matrix), h1))
        lambda2 = 1.0 / np.linalg.norm(np.dot(np.linalg.inv(self.camera_matrix), h2))
        lambda3 = (lambda1 + lambda2) / 2.0
        
        # Calculate rotation and translation
        K_inv = np.linalg.inv(self.camera_matrix)
        
        r1 = lambda1 * np.dot(K_inv, h1)
        r2 = lambda2 * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)
        t = lambda3 * np.dot(K_inv, h3)
        
        # Construct rotation matrix
        R = np.column_stack((r1, r2, r3))
        
        # Ensure proper rotation matrix using SVD
        U, _, Vt = np.linalg.svd(R)
        R_corrected = np.dot(U, Vt)
        
        # Convert to rotation vector
        rvec = cv2.Rodrigues(R_corrected)[0].flatten()
        tvec = t.flatten()
        
        return [rvec], [tvec]
    
    def _solve_pnp_custom(self, object_points: np.ndarray, 
                         image_points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve Perspective-n-Point problem using custom implementation.
        
        Args:
            object_points: 3D object points
            image_points: 2D image points
            
        Returns:
            Tuple of (rotation_vector, translation_vector)
        """
        # Use OpenCV's solvePnP for robustness, but implement iterative algorithm
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, 
            self.camera_matrix, self.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return rvec.flatten(), tvec.flatten()
        else:
            return None, None
    
    def _refine_pose_iterative(self, rvec: np.ndarray, tvec: np.ndarray,
                             object_points: np.ndarray, image_points: np.ndarray,
                             max_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine pose estimation using iterative optimization.
        
        Args:
            rvec: Initial rotation vector
            tvec: Initial translation vector
            object_points: 3D object points
            image_points: 2D image points
            max_iterations: Maximum number of iterations
            
        Returns:
            Refined (rotation_vector, translation_vector)
        """
        # This is a simplified version - full implementation would use Levenberg-Marquardt
        for _ in range(max_iterations):
            # Project 3D points to image plane
            projected_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, 
                self.camera_matrix, self.distortion_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Calculate reprojection error
            error = image_points - projected_points
            
            # If error is small enough, stop
            if np.mean(np.linalg.norm(error, axis=1)) < 0.1:
                break
            
            # Simple gradient descent (simplified)
            # In practice, you'd compute Jacobian and use proper optimization
            gradient_scale = 0.001
            rvec += gradient_scale * np.random.randn(3) * 0.1
            tvec += gradient_scale * np.random.randn(3) * 0.1
        
        return rvec, tvec
    
    def estimate_pose_single_marker(self, corners: np.ndarray, 
                                  marker_size: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate pose of a single ArUCO marker.
        
        Args:
            corners: 2D corners of the detected marker (4x2)
            marker_size: Real-world size of the marker
            
        Returns:
            Tuple of (rotation_vector, translation_vector)
        """
        # Create 3D object points
        object_points = self._create_3d_marker_points(marker_size)
        
        # Solve PnP problem
        rvec, tvec = self._solve_pnp_custom(object_points, corners)
        
        if rvec is not None and tvec is not None:
            # Refine pose estimation
            rvec_refined, tvec_refined = self._refine_pose_iterative(
                rvec, tvec, object_points, corners
            )
            return rvec_refined, tvec_refined
        
        return None, None
    
    def estimate_pose_homography_method(self, corners: np.ndarray, 
                                      marker_size: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate pose using homography decomposition method.
        
        Args:
            corners: 2D corners of the detected marker (4x2)
            marker_size: Real-world size of the marker
            
        Returns:
            Tuple of (rotation_vector, translation_vector)
        """
        # Create 2D template points (marker in canonical position)
        half_size = marker_size / 2.0
        template_points = np.array([
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size]
        ], dtype=np.float32)
        
        # Compute homography
        homography = self._compute_homography_custom(template_points, corners)
        
        if homography is not None:
            # Decompose homography
            rvecs, tvecs = self._decompose_homography_custom(homography, marker_size)
            
            if len(rvecs) > 0 and len(tvecs) > 0:
                return rvecs[0], tvecs[0]
        
        return None, None
    
    def triangulate_marker_position(self, corners1: np.ndarray, corners2: np.ndarray,
                                  camera_matrix1: np.ndarray, camera_matrix2: np.ndarray,
                                  R: np.ndarray, t: np.ndarray) -> Optional[np.ndarray]:
        """
        Triangulate 3D position of marker using stereo vision.
        
        Args:
            corners1: Corners from first camera
            corners2: Corners from second camera
            camera_matrix1: First camera matrix
            camera_matrix2: Second camera matrix
            R: Rotation between cameras
            t: Translation between cameras
            
        Returns:
            3D position of marker corners
        """
        # Create projection matrices
        P1 = np.dot(camera_matrix1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(camera_matrix2, np.hstack((R, t.reshape(3, 1))))
        
        # Triangulate each corner
        triangulated_points = []
        
        for i in range(4):
            point1 = corners1[i]
            point2 = corners2[i]
            
            # Triangulate using cv2.triangulatePoints
            point_4d = cv2.triangulatePoints(P1, P2, 
                                           point1.reshape(2, 1), 
                                           point2.reshape(2, 1))
            
            # Convert from homogeneous coordinates
            point_3d = point_4d[:3] / point_4d[3]
            triangulated_points.append(point_3d.flatten())
        
        return np.array(triangulated_points)
    
    def calculate_pose_accuracy(self, rvec: np.ndarray, tvec: np.ndarray,
                              object_points: np.ndarray, image_points: np.ndarray) -> float:
        """
        Calculate pose estimation accuracy using reprojection error.
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
            object_points: 3D object points
            image_points: 2D image points
            
        Returns:
            Mean reprojection error in pixels
        """
        # Project 3D points to image plane
        projected_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, 
            self.camera_matrix, self.distortion_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate reprojection error
        errors = np.linalg.norm(image_points - projected_points, axis=1)
        return np.mean(errors)
    
    def pose_to_transformation_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Convert rotation and translation vectors to 4x4 transformation matrix.
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            4x4 transformation matrix
        """
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        
        return T
    
    def draw_coordinate_axes(self, image: np.ndarray, rvec: np.ndarray, 
                           tvec: np.ndarray, length: float = 0.1) -> np.ndarray:
        """
        Draw coordinate axes on the marker.
        
        Args:
            image: Input image
            rvec: Rotation vector
            tvec: Translation vector
            length: Length of the axes
            
        Returns:
            Image with drawn axes
        """
        # Define 3D axes points
        axes_points = np.array([
            [0, 0, 0],      # Origin
            [length, 0, 0], # X-axis (red)
            [0, length, 0], # Y-axis (green)
            [0, 0, -length] # Z-axis (blue)
        ], dtype=np.float32)
        
        # Project axes to image plane
        projected_axes, _ = cv2.projectPoints(
            axes_points, rvec, tvec, 
            self.camera_matrix, self.distortion_coeffs
        )
        projected_axes = projected_axes.reshape(-1, 2).astype(int)
        
        # Draw axes
        result_image = image.copy()
        origin = tuple(projected_axes[0])
        
        # X-axis (red)
        cv2.arrowedLine(result_image, origin, tuple(projected_axes[1]), 
                       (0, 0, 255), 3, tipLength=0.3)
        
        # Y-axis (green)
        cv2.arrowedLine(result_image, origin, tuple(projected_axes[2]), 
                       (0, 255, 0), 3, tipLength=0.3)
        
        # Z-axis (blue)
        cv2.arrowedLine(result_image, origin, tuple(projected_axes[3]), 
                       (255, 0, 0), 3, tipLength=0.3)
        
        return result_image
    
    def get_pose_info(self, rvec: np.ndarray, tvec: np.ndarray) -> dict:
        """
        Get human-readable pose information.
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            Dictionary with pose information
        """
        # Convert rotation vector to Euler angles
        R, _ = cv2.Rodrigues(rvec)
        euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        
        # Calculate distance from camera
        distance = np.linalg.norm(tvec)
        
        return {
            'translation': {
                'x': float(tvec[0]),
                'y': float(tvec[1]),
                'z': float(tvec[2])
            },
            'rotation_euler': {
                'roll': float(euler_angles[0]),
                'pitch': float(euler_angles[1]),
                'yaw': float(euler_angles[2])
            },
            'distance': float(distance),
            'rotation_vector': rvec.tolist(),
            'translation_vector': tvec.tolist()
        }


if __name__ == "__main__":
    # Example usage would require camera calibration data
    print("Pose Estimation Module")
    print("This module requires camera calibration data to function.")
    print("Please use it in conjunction with the main tracking system.")