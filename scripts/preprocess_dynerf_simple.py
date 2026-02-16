#!/usr/bin/env python3
"""
Simple script to extract frames from Dynerf videos without complex imports
"""
import os
import cv2
import argparse
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, fps=None):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame as PNG
        frame_filename = f"{frame_count:04d}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")
    return True

def preprocess_dynerf_dataset(datadir):
    """Preprocess Dynerf dataset by extracting frames from videos"""
    datadir = Path(datadir)
    
    if not datadir.exists():
        print(f"Error: Dataset directory {datadir} does not exist")
        return False
    
    print(f"Processing dataset: {datadir}")
    
    # Find all video files
    video_files = list(datadir.glob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
    success_count = 0
    for video_file in video_files:
        # Create images directory for this camera
        camera_name = video_file.stem  # e.g., "cam00" from "cam00.mp4"
        images_dir = datadir / camera_name / "images"
        
        print(f"Processing {video_file.name} -> {images_dir}")
        
        if extract_frames_from_video(str(video_file), str(images_dir)):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(video_files)} videos")
    return success_count > 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract images from dynerf videos")
    parser.add_argument("--datadir", default='data/dynerf/cut_roasted_beef', type=str)
    args = parser.parse_args()
    
    success = preprocess_dynerf_dataset(args.datadir)
    if success:
        print("✅ Frame extraction completed successfully!")
    else:
        print("❌ Frame extraction failed!")
