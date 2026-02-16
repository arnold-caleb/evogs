#!/usr/bin/env python3
"""
Compare temporal quality between velocity field and displacement field.

Usage:
    python scripts/compare_temporal_quality.py \
        --velocity_video output/sparse_supervision_*/videos/*.mp4 \
        --displacement_video output/sparse_displacement_*/videos/*.mp4 \
        --split_at 150  # Frame where training ends, future begins
"""

import argparse
import subprocess
import json
from pathlib import Path
import numpy as np


def run_evaluation(video_path, start_frame, end_frame, output_suffix):
    """Run temporal evaluation on a video segment."""
    output_json = Path(video_path).parent / f"metrics_{output_suffix}.json"
    
    cmd = [
        'python', 'scripts/evaluate_temporal_quality.py',
        '--video', str(video_path),
        '--start_frame', str(start_frame),
        '--end_frame', str(end_frame),
        '--output', str(output_json)
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    with open(output_json, 'r') as f:
        return json.load(f)


def print_comparison_table(vel_train, vel_future, disp_train, disp_future):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("TEMPORAL QUALITY COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} | {'Training (0-149)':<20} | {'Future (150-299)':<20}")
    print("-"*80)
    
    # Optical Flow Variance
    print(f"{'Optical Flow Variance':<30}")
    print(f"  {'Velocity Field':<28} | {vel_train['optical_flow']['temporal_variance']:>18.6f} | {vel_future['optical_flow']['temporal_variance']:>18.6f}")
    print(f"  {'Displacement Field':<28} | {disp_train['optical_flow']['temporal_variance']:>18.6f} | {disp_future['optical_flow']['temporal_variance']:>18.6f}")
    print()
    
    # Warping Error
    print(f"{'Warping Error':<30}")
    print(f"  {'Velocity Field':<28} | {vel_train['warping']['mean_warping_error']:>18.6f} | {vel_future['warping']['mean_warping_error']:>18.6f}")
    print(f"  {'Displacement Field':<28} | {disp_train['warping']['mean_warping_error']:>18.6f} | {disp_future['warping']['mean_warping_error']:>18.6f}")
    print()
    
    # Temporal LPIPS
    print(f"{'Temporal LPIPS':<30}")
    print(f"  {'Velocity Field':<28} | {vel_train['temporal_lpips']['mean_lpips']:>18.6f} | {vel_future['temporal_lpips']['mean_lpips']:>18.6f}")
    print(f"  {'Displacement Field':<28} | {disp_train['temporal_lpips']['mean_lpips']:>18.6f} | {disp_future['temporal_lpips']['mean_lpips']:>18.6f}")
    print("="*80)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Check if displacement degrades on future
    disp_flow_increase = (disp_future['optical_flow']['temporal_variance'] / 
                          disp_train['optical_flow']['temporal_variance'])
    vel_flow_increase = (vel_future['optical_flow']['temporal_variance'] / 
                         vel_train['optical_flow']['temporal_variance'])
    
    print(f"Displacement field flow variance increase: {disp_flow_increase:.2f}x")
    print(f"Velocity field flow variance increase:     {vel_flow_increase:.2f}x")
    
    if disp_flow_increase > 2.0 and vel_flow_increase < 1.5:
        print("\n✅ SUCCESS: Displacement field fails on future (>2x increase)")
        print("✅ SUCCESS: Velocity field generalizes smoothly (<1.5x increase)")
        print("\n🎉 This proves velocity fields learn TRUE DYNAMICS!")
    elif disp_flow_increase > 2.0:
        print("\n⚠️  Displacement field fails on future, but velocity also struggles")
        print("   Consider longer training or stronger anchor constraints")
    else:
        print("\n⚠️  Displacement field NOT degrading as expected on future frames")
        print("   Check if future frames are actually excluded from training")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--velocity_video', type=str, required=True)
    parser.add_argument('--displacement_video', type=str, required=True)
    parser.add_argument('--split_at', type=int, default=150, 
                       help='Frame where training ends (future begins)')
    parser.add_argument('--total_frames', type=int, default=300)
    args = parser.parse_args()
    
    print("="*80)
    print("VELOCITY FIELD - Training Frames (0-149)")
    print("="*80)
    vel_train = run_evaluation(args.velocity_video, 0, args.split_at - 1, 'velocity_train')
    
    print("\n" + "="*80)
    print("VELOCITY FIELD - Future Frames (150-299)")
    print("="*80)
    vel_future = run_evaluation(args.velocity_video, args.split_at, args.total_frames - 1, 'velocity_future')
    
    print("\n" + "="*80)
    print("DISPLACEMENT FIELD - Training Frames (0-149)")
    print("="*80)
    disp_train = run_evaluation(args.displacement_video, 0, args.split_at - 1, 'displacement_train')
    
    print("\n" + "="*80)
    print("DISPLACEMENT FIELD - Future Frames (150-299)")
    print("="*80)
    disp_future = run_evaluation(args.displacement_video, args.split_at, args.total_frames - 1, 'displacement_future')
    
    # Print comparison
    print_comparison_table(vel_train, vel_future, disp_train, disp_future)


if __name__ == '__main__':
    main()

