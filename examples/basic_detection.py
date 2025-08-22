"""
Example: Basic ArUCO Detection

This example demonstrates basic ArUCO marker detection using the custom implementation.
"""

import cv2
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from aruco_detection import ArUCODetector


def main():
    """Basic ArUCO detection example."""
    # Initialize detector
    detector = ArUCODetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Basic ArUCO Detection Example")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers
        corners_list, ids_list = detector.detect_markers(frame)
        
        # Draw detected markers
        result_frame = detector.draw_detected_markers(frame, corners_list, ids_list)
        
        # Display detection count
        cv2.putText(result_frame, f"Detected: {len(ids_list)} markers", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show result
        cv2.imshow('Basic ArUCO Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()