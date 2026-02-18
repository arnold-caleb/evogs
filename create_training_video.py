#!/usr/bin/env python3
"""
Create training progression videos from saved training images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def create_training_video(image_dir, output_path, fps=10):
    """
    Create a video from training progression images.
    
    Args:
        image_dir: Directory containing training images
        output_path: Path for output video
        fps: Frames per second for the video
    """
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} does not exist")
        return
    
    # Get all image files and sort them
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(Path(image_dir).glob(ext))
    
    # Sort by iteration number (first number in filename)
    image_files = sorted(image_files, key=lambda x: int(x.stem.split('_')[0]))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images in {image_dir}")
    print("Creating training progression video from a fixed viewpoint...")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    height, width, layers = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write images to video
    for i, image_file in enumerate(image_files):
        image = cv2.imread(str(image_file))
        if image is not None:
            video_writer.write(image)
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(image_files)} images")
    
    video_writer.release()
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create training progression videos")
    parser.add_argument("--output_dir", type=str, 
                       default="output/sde_baseline/cut_roasted_beef",
                       help="Output directory containing training results")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    
    args = parser.parse_args()
    
    # Find training image directories
    coarse_dir = os.path.join(args.output_dir, "coarsetrain_render", "images")
    fine_dir = os.path.join(args.output_dir, "finetrain_render", "images")
    
    # Create videos with timestamps
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(coarse_dir):
        coarse_video = os.path.join(args.output_dir, f"coarse_training_progression_{timestamp}.mp4")
        print(f"Creating coarse training video...")
        create_training_video(coarse_dir, coarse_video, args.fps)
    
    if os.path.exists(fine_dir):
        fine_video = os.path.join(args.output_dir, f"fine_training_progression_{timestamp}.mp4")
        print(f"Creating fine training video...")
        create_training_video(fine_dir, fine_video, args.fps)
    
    print("Video creation complete!")

if __name__ == "__main__":
    main()
