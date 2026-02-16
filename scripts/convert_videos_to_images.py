#!/usr/bin/env python3
"""
Convert video files to image sequences for dynamic NeRF datasets.
Extracts frames from each camera video at specified FPS.
"""

import os
import subprocess
import argparse
from pathlib import Path


def convert_video_to_images(video_path, output_dir, fps=30):
    """
    Convert a video file to a sequence of images.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save images
        fps: Frame rate to extract (default: 30)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        '-qscale:v', '1',
        '-qmin', '1',
        '-start_number', '0',
        os.path.join(output_dir, '%04d.png')
    ]
    
    print(f"Converting {video_path} -> {output_dir}")
    subprocess.run(cmd, check=True)
    
    # Count frames
    num_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"Extracted {num_frames} frames")
    return num_frames


def process_dataset(dataset_path):
    """
    Process a dataset directory containing video files.
    
    Expected structure:
    dataset_path/
        cam00.mp4
        cam01.mp4
        ...
        poses_bounds.npy
    """
    dataset_path = Path(dataset_path)
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_path.name}")
    print(f"{'='*60}\n")
    
    # Find all video files
    video_files = sorted(dataset_path.glob('cam*.mp4'))
    
    if not video_files:
        print(f"No video files found in {dataset_path}")
        return
    
    print(f"Found {len(video_files)} camera videos")
    
    # Process each video
    total_frames = 0
    for video_file in video_files:
        camera_name = video_file.stem  # e.g., 'cam00'
        output_dir = dataset_path / camera_name / 'images'
        
        try:
            num_frames = convert_video_to_images(video_file, output_dir)
            total_frames += num_frames
        except subprocess.CalledProcessError as e:
            print(f"Error converting {video_file}: {e}")
            continue
    
    print(f"\n✅ Dataset {dataset_path.name} processed successfully!")
    print(f"   Total frames extracted: {total_frames}")
    print(f"   Average frames per camera: {total_frames // len(video_files)}")


def main():
    parser = argparse.ArgumentParser(description='Convert videos to images for NeRF datasets')
    parser.add_argument('--datasets', nargs='+', help='Dataset paths to process')
    parser.add_argument('--data_dir', type=str, default='data/dynerf',
                       help='Base directory containing datasets')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frame rate to extract (default: 30)')
    
    args = parser.parse_args()
    
    # Process each dataset
    if args.datasets:
        for dataset_name in args.datasets:
            dataset_path = Path(args.data_dir) / dataset_name
            if dataset_path.exists():
                process_dataset(dataset_path)
            else:
                print(f"Dataset not found: {dataset_path}")
    else:
        # Process all datasets in data_dir
        data_dir = Path(args.data_dir)
        datasets = [d for d in data_dir.iterdir() if d.is_dir()]
        
        print(f"Found {len(datasets)} datasets in {data_dir}")
        
        for dataset_path in sorted(datasets):
            process_dataset(dataset_path)
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

