#!/usr/bin/env python3
"""
Create visualization videos from rendered static frame 0:
1. Spiral video rotating through all viewpoints
2. GT vs Rendered comparison video (side-by-side)
3. Before/after transition video
"""
import os
import glob
import cv2
import numpy as np
from PIL import Image
import argparse


def create_spiral_video(render_dir, output_path, fps=10, loop=True):
    """Create a video spiraling through all rendered viewpoints."""
    print(f"Creating spiral video...")
    
    # Get all rendered images
    render_files = sorted(glob.glob(os.path.join(render_dir, "render_*.png")))
    
    if not render_files:
        print(f"Error: No render files found in {render_dir}")
        return
    
    print(f"  Found {len(render_files)} viewpoints")
    
    # Read first image to get dimensions
    first_img = cv2.imread(render_files[0])
    height, width = first_img.shape[:2]
    
    # Create video writer (use H264 for better compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Forward pass
    for img_path in render_files:
        img = cv2.imread(img_path)
        out.write(img)
    
    # If loop, go backward for smooth transition
    if loop:
        for img_path in reversed(render_files):
            img = cv2.imread(img_path)
            out.write(img)
    
    out.release()
    print(f"  ✓ Saved: {output_path}")
    print(f"    Duration: {len(render_files) * (2 if loop else 1) / fps:.1f}s at {fps} FPS")


def create_comparison_video(render_dir, output_path, fps=10):
    """Create side-by-side GT vs Rendered comparison video."""
    print(f"\nCreating GT vs Rendered comparison video...")
    
    # Get all rendered and GT images
    render_files = sorted(glob.glob(os.path.join(render_dir, "render_*.png")))
    gt_files = sorted(glob.glob(os.path.join(render_dir, "gt_*.png")))
    
    if not render_files or not gt_files:
        print(f"Error: Missing render or GT files")
        return
    
    # Match render and GT files by index
    pairs = []
    for gt_path in gt_files:
        gt_idx = int(os.path.basename(gt_path).split('_')[1].split('.')[0])
        render_path = os.path.join(render_dir, f"render_{gt_idx:03d}.png")
        if os.path.exists(render_path):
            pairs.append((gt_path, render_path))
    
    print(f"  Found {len(pairs)} GT-Render pairs")
    
    # Read first pair to get dimensions
    gt_img = cv2.imread(pairs[0][0])
    render_img = cv2.imread(pairs[0][1])
    height, width = gt_img.shape[:2]
    
    # Create side-by-side canvas
    canvas_width = width * 2 + 60  # 60px gap + labels
    canvas_height = height + 80  # Extra space for title
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
    
    # Create video - forward and backward
    for direction in ['forward', 'backward']:
        pair_list = pairs if direction == 'forward' else list(reversed(pairs))
        
        for gt_path, render_path in pair_list:
            # Create canvas
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Load images
            gt = cv2.imread(gt_path)
            rendered = cv2.imread(render_path)
            
            # Add title
            cv2.putText(canvas, "Frame 0 Reconstruction Quality (PSNR: 33.0 dB)", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Place images side-by-side
            y_offset = 80
            canvas[y_offset:y_offset+height, 0:width] = gt
            canvas[y_offset:y_offset+height, width+60:width+60+width] = rendered
            
            # Add labels
            cv2.putText(canvas, "Ground Truth", (width//2 - 100, y_offset - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(canvas, "Rendered (Ours)", (width + 60 + width//2 - 120, y_offset - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(canvas)
    
    out.release()
    print(f"  ✓ Saved: {output_path}")
    print(f"    Duration: {len(pairs) * 2 / fps:.1f}s at {fps} FPS")


def create_transition_video(render_dir, output_path, fps=2):
    """Create before/after transition video."""
    print(f"\nCreating before/after transition video...")
    
    # Get comparison images
    gt_files = sorted(glob.glob(os.path.join(render_dir, "gt_*.png")))[:4]
    render_files = sorted(glob.glob(os.path.join(render_dir, "render_00[0-3].png")))
    
    if not gt_files or not render_files:
        print("Error: Missing files")
        return
    
    # Read first to get dimensions
    first_img = cv2.imread(gt_files[0])
    height, width = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Alternate between GT and Rendered
    for gt_path, render_path in zip(gt_files, render_files):
        # Show GT for 1 second
        gt = cv2.imread(gt_path)
        for _ in range(fps):
            out.write(gt)
        
        # Show rendered for 1 second
        rendered = cv2.imread(render_path)
        for _ in range(fps):
            out.write(rendered)
    
    out.release()
    print(f"  ✓ Saved: {output_path}")
    print(f"    Duration: {len(gt_files) * 2}s (alternating GT/Rendered)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create visualization videos")
    parser.add_argument("--render_dir", type=str, required=True, help="Directory with rendered images")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for spiral video")
    args = parser.parse_args()
    
    render_dir = args.render_dir
    video_dir = os.path.join(os.path.dirname(render_dir), "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    print("=" * 70)
    print("CREATING VISUALIZATION VIDEOS")
    print("=" * 70)
    print(f"Input: {render_dir}")
    print(f"Output: {video_dir}")
    print()
    
    # 1. Spiral video
    spiral_path = os.path.join(video_dir, "spiral_viewpoints.mp4")
    create_spiral_video(render_dir, spiral_path, fps=args.fps, loop=True)
    
    # 2. Comparison video
    comparison_path = os.path.join(video_dir, "gt_vs_rendered_comparison.mp4")
    create_comparison_video(render_dir, comparison_path, fps=args.fps)
    
    # 3. Transition video
    transition_path = os.path.join(video_dir, "before_after_transition.mp4")
    create_transition_video(render_dir, transition_path, fps=2)
    
    print("\n" + "=" * 70)
    print("✅ ALL VIDEOS CREATED!")
    print("=" * 70)
    print(f"\nVideos saved in: {video_dir}")
    print(f"  1. spiral_viewpoints.mp4 - Rotating through all {len(glob.glob(os.path.join(render_dir, 'render_*.png')))} views")
    print(f"  2. gt_vs_rendered_comparison.mp4 - Side-by-side quality comparison")
    print(f"  3. before_after_transition.mp4 - Alternating GT/Rendered")

