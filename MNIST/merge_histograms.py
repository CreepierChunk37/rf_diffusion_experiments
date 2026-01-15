#!/usr/bin/env python3
"""
Script to merge histogram images into one large composite image.
Images will be arranged in a 3x4 grid ordered by tau values.
"""

from PIL import Image
import os
import re

def natural_sort_key(filename):
    """Extract tau value for proper sorting."""
    match = re.search(r'tau_(\d+|Inf)', filename)
    if match:
        value = match.group(1)
        return float('inf') if value == 'Inf' else int(value)
    return 0

def merge_histograms():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # Change to the script's directory
    
    # Get all histogram files
    files = [f for f in os.listdir('.') if f.startswith('hist_dim_2000_tau_') and f.endswith('.jpg')]
    
    # Sort files by tau value
    files.sort(key=natural_sort_key)
    
    print(f"Found {len(files)} histogram files:")
    for f in files:
        print(f"  {f}")
    
    if not files:
        print("No histogram files found! Please ensure the script is in the same directory as the histogram images.")
        return
    
    # Load first image to get dimensions
    first_img = Image.open(files[0])
    img_width, img_height = first_img.size
    
    # Calculate grid layout (3x4 for 11 images)
    cols = 4
    rows = 3
    padding = 10  # Space between images
    
    # Calculate dimensions of the composite image
    composite_width = cols * img_width + (cols - 1) * padding
    composite_height = rows * img_height + (rows - 1) * padding
    
    # Create white background
    composite = Image.new('RGB', (composite_width, composite_height), 'white')
    
    # Paste images in grid
    for idx, filename in enumerate(files):
        if idx >= cols * rows:  # Safety check
            break
            
        img = Image.open(filename)
        
        # Calculate position
        row = idx // cols
        col = idx % cols
        
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        
        composite.paste(img, (x, y))
        print(f"Pasted {filename} at position ({x}, {y}) - Row {row+1}, Col {col+1}")
    
    # Save the composite image
    output_filename = 'merged_histograms_grid.jpg'
    composite.save(output_filename, quality=95)
    print(f"\nComposite image saved as: {output_filename}")
    print(f"Final dimensions: {composite_width} x {composite_height} pixels")

if __name__ == "__main__":
    merge_histograms()