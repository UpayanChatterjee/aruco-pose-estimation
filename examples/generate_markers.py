"""
Utility: ArUCO Marker Generator

This utility generates ArUCO markers for printing and testing.
"""

import cv2
import numpy as np
import os


def generate_aruco_marker(marker_id: int, marker_size: int = 200, 
                         border_bits: int = 1, save_path: str = None):
    """
    Generate a simple ArUCO marker for testing.
    
    Args:
        marker_id: ID of the marker to generate
        marker_size: Size of the marker in pixels
        border_bits: Border size in bits
        save_path: Path to save the marker image
    """
    # Simple 4x4 patterns for demonstration
    patterns = {
        0: np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ], dtype=np.uint8),
        
        1: np.array([
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ], dtype=np.uint8),
        
        2: np.array([
            [0, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 0]
        ], dtype=np.uint8),
    }
    
    if marker_id not in patterns:
        print(f"Warning: Marker ID {marker_id} not available. Using ID 0.")
        marker_id = 0
    
    pattern = patterns[marker_id]
    
    # Calculate dimensions
    total_size = 4 + 2 * border_bits  # 4x4 pattern + border
    cell_size = marker_size // total_size
    
    # Create marker image
    marker_img = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
    
    # Fill pattern (excluding border)
    for i in range(4):
        for j in range(4):
            y_start = (i + border_bits) * cell_size
            y_end = (i + border_bits + 1) * cell_size
            x_start = (j + border_bits) * cell_size
            x_end = (j + border_bits + 1) * cell_size
            
            color = 0 if pattern[i, j] == 1 else 255
            marker_img[y_start:y_end, x_start:x_end] = color
    
    # Add border (black)
    border_thickness = border_bits * cell_size
    marker_img[:border_thickness, :] = 0  # Top
    marker_img[-border_thickness:, :] = 0  # Bottom
    marker_img[:, :border_thickness] = 0  # Left
    marker_img[:, -border_thickness:] = 0  # Right
    
    if save_path:
        cv2.imwrite(save_path, marker_img)
        print(f"Marker {marker_id} saved to: {save_path}")
    
    return marker_img


def generate_marker_sheet(marker_ids: list, sheet_size: tuple = (1200, 800), 
                         marker_size: int = 150, save_path: str = None):
    """
    Generate a sheet with multiple ArUCO markers.
    
    Args:
        marker_ids: List of marker IDs to generate
        sheet_size: Size of the sheet (width, height)
        marker_size: Size of each marker
        save_path: Path to save the sheet
    """
    sheet_img = np.ones(sheet_size[::-1], dtype=np.uint8) * 255
    
    # Calculate grid layout
    cols = sheet_size[0] // (marker_size + 50)
    rows = sheet_size[1] // (marker_size + 100)
    
    current_row = 0
    current_col = 0
    
    for marker_id in marker_ids:
        if current_row >= rows:
            break
        
        # Generate marker
        marker_img = generate_aruco_marker(marker_id, marker_size)
        
        # Calculate position
        x_pos = 25 + current_col * (marker_size + 50)
        y_pos = 50 + current_row * (marker_size + 100)
        
        # Place marker on sheet
        sheet_img[y_pos:y_pos+marker_size, x_pos:x_pos+marker_size] = marker_img
        
        # Add ID label
        cv2.putText(sheet_img, f"ID: {marker_id}", 
                   (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, 0, 2)
        
        # Update position
        current_col += 1
        if current_col >= cols:
            current_col = 0
            current_row += 1
    
    if save_path:
        cv2.imwrite(save_path, sheet_img)
        print(f"Marker sheet saved to: {save_path}")
    
    return sheet_img


def main():
    """Generate ArUCO markers for testing."""
    print("ArUCO Marker Generator")
    print("=" * 30)
    
    # Create output directory
    output_dir = "../data/generated_markers"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate individual markers
    print("Generating individual markers...")
    for marker_id in range(3):
        filename = f"marker_{marker_id:02d}.png"
        filepath = os.path.join(output_dir, filename)
        generate_aruco_marker(marker_id, marker_size=300, save_path=filepath)
    
    # Generate marker sheet
    print("\nGenerating marker sheet...")
    sheet_path = os.path.join(output_dir, "marker_sheet.png")
    generate_marker_sheet([0, 1, 2], save_path=sheet_path)
    
    print(f"\nMarkers generated in: {output_dir}")
    print("Print the markers and use them for testing!")


if __name__ == "__main__":
    main()