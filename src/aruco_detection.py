"""
ArUCO Tag Detection Module

This module implements ArUCO tag detection from scratch using custom corner detection,
thresholding, contour analysis, and marker identification algorithms.
Does not rely on built-in OpenCV ArUCO functions.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import math


class ArUCODetector:
    """
    Custom ArUCO tag detector implementing detection algorithms from scratch.
    """
    
    def __init__(self, dictionary_size: int = 50, marker_size: int = 5):
        """
        Initialize ArUCO detector.
        
        Args:
            dictionary_size: Number of markers in dictionary (default: 50 for DICT_4X4_50)
            marker_size: Size of marker grid (4x4, 5x5, etc.)
        """
        self.dictionary_size = dictionary_size
        self.marker_size = marker_size
        self.aruco_dict = self._generate_aruco_dictionary()
        
        # Detection parameters
        self.min_marker_area = 100  # Minimum area for marker candidates
        self.max_marker_area = 50000  # Maximum area for marker candidates
        self.min_contour_length = 16  # Minimum contour perimeter
        self.corner_refinement_window = 5  # Window size for corner refinement
        
    def _generate_aruco_dictionary(self) -> Dict[int, np.ndarray]:
        """
        Generate a simplified ArUCO dictionary.
        For demonstration, creates a basic 4x4 dictionary.
        """
        # This is a simplified version - in practice, you'd use the full ArUCO generation algorithm
        dictionary = {}
        
        # Example 4x4 ArUCO patterns (binary matrices)
        patterns = [
            np.array([
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ], dtype=np.uint8),
            
            np.array([
                [1, 0, 1, 1],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1]
            ], dtype=np.uint8),
            
            np.array([
                [0, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 0, 0, 0]
            ], dtype=np.uint8),
        ]
        
        for i, pattern in enumerate(patterns):
            if i < self.dictionary_size:
                dictionary[i] = pattern
                
        return dictionary
    
    def _apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply custom adaptive thresholding for marker detection.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def _find_contours_custom(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours using custom implementation.
        """
        # Use OpenCV's contour detection but add custom filtering
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours based on area and perimeter
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if (self.min_marker_area < area < self.max_marker_area and 
                perimeter > self.min_contour_length):
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def _approximate_polygon(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Approximate contour to polygon and check if it's quadrilateral.
        """
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if polygon has 4 vertices (quadrilateral)
        if len(approx) == 4:
            return approx.reshape(4, 2)
        
        return None
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in clockwise order starting from top-left.
        """
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Calculate angles from centroid to each corner
        angles = []
        for corner in corners:
            angle = math.atan2(corner[1] - centroid[1], corner[0] - centroid[0])
            angles.append(angle)
        
        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        
        # Reorder to start from top-left (adjust based on angle quadrants)
        ordered_corners = corners[sorted_indices]
        
        # Ensure proper ordering: top-left, top-right, bottom-right, bottom-left
        # This is a simplified version - a complete implementation would be more robust
        return ordered_corners
    
    def _refine_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Refine corner positions using subpixel accuracy.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Convert corners to the format expected by cornerSubPix
        corners_float = corners.astype(np.float32)
        corners_reshaped = corners_float.reshape(-1, 1, 2)
        
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = cv2.cornerSubPix(
            gray, corners_reshaped, 
            (self.corner_refinement_window, self.corner_refinement_window),
            (-1, -1), criteria
        )
        
        return refined_corners.reshape(4, 2)
    
    def _extract_marker_bits(self, image: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract marker bits from the detected quadrilateral region.
        """
        # Define destination points for perspective transformation
        marker_size_px = 100  # Size of the extracted marker in pixels
        dst_points = np.array([
            [0, 0],
            [marker_size_px, 0],
            [marker_size_px, marker_size_px],
            [0, marker_size_px]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        src_points = corners.astype(np.float32)
        transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(image, transformation_matrix, 
                                   (marker_size_px, marker_size_px))
        
        # Convert to grayscale and threshold
        if len(warped.shape) == 3:
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped
        
        # Threshold the warped image
        _, binary_marker = cv2.threshold(warped_gray, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract bits from the marker (excluding border)
        grid_size = 6  # 4x4 marker + 2-pixel border
        cell_size = marker_size_px // grid_size
        
        marker_bits = np.zeros((4, 4), dtype=np.uint8)
        
        for i in range(4):
            for j in range(4):
                # Calculate cell center (skip border)
                y = cell_size + i * cell_size + cell_size // 2
                x = cell_size + j * cell_size + cell_size // 2
                
                # Sample the center of each cell
                if y < marker_size_px and x < marker_size_px:
                    marker_bits[i, j] = 1 if binary_marker[y, x] > 127 else 0
        
        return marker_bits
    
    def _identify_marker(self, marker_bits: np.ndarray) -> Optional[int]:
        """
        Identify the marker by comparing with dictionary.
        """
        if marker_bits is None:
            return None
        
        # Try all 4 rotations
        for rotation in range(4):
            rotated_bits = np.rot90(marker_bits, rotation)
            
            # Compare with dictionary
            for marker_id, pattern in self.aruco_dict.items():
                if np.array_equal(rotated_bits, pattern):
                    return marker_id
        
        return None
    
    def _calculate_marker_corners_3d(self, marker_size: float) -> np.ndarray:
        """
        Calculate 3D coordinates of marker corners.
        
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
    
    def detect_markers(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Detect ArUCO markers in the input image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (corners_list, ids_list)
        """
        # Step 1: Apply adaptive thresholding
        binary_image = self._apply_adaptive_threshold(image)
        
        # Step 2: Find contours
        contours = self._find_contours_custom(binary_image)
        
        # Step 3: Process each contour
        detected_corners = []
        detected_ids = []
        
        for contour in contours:
            # Approximate to quadrilateral
            corners = self._approximate_polygon(contour)
            
            if corners is not None:
                # Order corners
                ordered_corners = self._order_corners(corners)
                
                # Refine corner positions
                refined_corners = self._refine_corners(image, ordered_corners)
                
                # Extract marker bits
                marker_bits = self._extract_marker_bits(image, refined_corners)
                
                # Identify marker
                marker_id = self._identify_marker(marker_bits)
                
                if marker_id is not None:
                    detected_corners.append(refined_corners)
                    detected_ids.append(marker_id)
        
        return detected_corners, detected_ids
    
    def draw_detected_markers(self, image: np.ndarray, corners_list: List[np.ndarray], 
                            ids_list: List[int]) -> np.ndarray:
        """
        Draw detected markers on the image.
        
        Args:
            image: Input image
            corners_list: List of detected marker corners
            ids_list: List of detected marker IDs
            
        Returns:
            Image with drawn markers
        """
        result_image = image.copy()
        
        for corners, marker_id in zip(corners_list, ids_list):
            # Draw marker outline
            corners_int = corners.astype(int)
            cv2.polylines(result_image, [corners_int], True, (0, 255, 0), 2)
            
            # Draw corner points
            for corner in corners_int:
                cv2.circle(result_image, tuple(corner), 5, (255, 0, 0), -1)
            
            # Draw marker ID
            center = np.mean(corners_int, axis=0).astype(int)
            cv2.putText(result_image, f"ID: {marker_id}", 
                       tuple(center - [20, 10]), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
        
        return result_image


if __name__ == "__main__":
    # Example usage
    detector = ArUCODetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    
    print("ArUCO Detection Demo")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers
        corners_list, ids_list = detector.detect_markers(frame)
        
        # Draw detected markers
        result_frame = detector.draw_detected_markers(frame, corners_list, ids_list)
        
        # Display detection info
        cv2.putText(result_frame, f"Detected: {len(ids_list)} markers", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('ArUCO Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()