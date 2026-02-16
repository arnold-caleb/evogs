#!/usr/bin/env python3
"""
Evaluate temporal/motion quality of rendered videos.

Metrics:
1. Optical Flow Variance (smoothness)
2. Warping Error (motion predictability)
3. Temporal LPIPS (perceptual consistency)
4. PSNR (for comparison)

Usage:
    python scripts/evaluate_temporal_quality.py \
        --video output/sparse_displacement_*/videos/temporal_300frames.mp4 \
        --gt_video data/dynerf/cut_roasted_beef/videos/cam00.mp4 \
        --start_frame 0 --end_frame 299
"""

import argparse
import cv2
import numpy as np
import torch
import lpips
from pathlib import Path
import json


def load_video(video_path, start_frame=0, end_frame=None):
    """Load video frames as numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= start_frame:
            if end_frame is not None and frame_idx > end_frame:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
        
        frame_idx += 1
    
    cap.release()
    return frames


def compute_optical_flow_variance(frames):
    """
    Compute temporal smoothness via optical flow variance.
    Lower = smoother motion.
    """
    flow_variances = []
    flow_magnitudes = []
    
    for i in range(len(frames) - 1):
        frame1 = (frames[i] * 255).astype(np.uint8)
        frame2 = (frames[i+1] * 255).astype(np.uint8)
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Flow magnitude
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitudes.append(np.mean(flow_mag))
        flow_variances.append(np.var(flow_mag))
    
    # Variance of flow magnitudes across frames (temporal consistency)
    temporal_variance = np.var(flow_magnitudes)
    avg_spatial_variance = np.mean(flow_variances)
    
    return {
        'temporal_variance': float(temporal_variance),
        'avg_spatial_variance': float(avg_spatial_variance),
        'flow_magnitudes': [float(m) for m in flow_magnitudes]
    }


def compute_warping_error(frames):
    """
    Warp frame t to frame t+1, measure error.
    Lower = more predictable motion.
    """
    warping_errors = []
    
    for i in range(len(frames) - 1):
        frame1 = (frames[i] * 255).astype(np.uint8)
        frame2 = (frames[i+1] * 255).astype(np.uint8)
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Create remapping coordinates
        h, w = frame1.shape[:2]
        flow_map_x = (np.arange(w, dtype=np.float32) + flow[..., 0])
        flow_map_y = (np.arange(h, dtype=np.float32)[:, np.newaxis] + flow[..., 1])
        
        # Warp frame1 to match frame2
        warped = cv2.remap(
            frame1, flow_map_x, flow_map_y,
            interpolation=cv2.INTER_LINEAR
        )
        
        # Compute error
        error = np.mean((warped.astype(float) - frame2.astype(float))**2)
        warping_errors.append(error)
    
    return {
        'mean_warping_error': float(np.mean(warping_errors)),
        'std_warping_error': float(np.std(warping_errors)),
        'warping_errors': [float(e) for e in warping_errors]
    }


def compute_temporal_lpips(frames, device='cuda'):
    """
    Perceptual motion consistency via LPIPS.
    """
    lpips_model = lpips.LPIPS(net='alex').to(device)
    temporal_distances = []
    
    for i in range(len(frames) - 1):
        frame1 = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).to(device)
        frame2 = torch.from_numpy(frames[i+1]).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Normalize to [-1, 1]
        frame1 = frame1 * 2 - 1
        frame2 = frame2 * 2 - 1
        
        with torch.no_grad():
            distance = lpips_model(frame1, frame2)
        
        temporal_distances.append(distance.item())
    
    return {
        'mean_lpips': float(np.mean(temporal_distances)),
        'std_lpips': float(np.std(temporal_distances)),
        'lpips_distances': [float(d) for d in temporal_distances]
    }


def compute_psnr(frames, gt_frames):
    """Compute PSNR for comparison."""
    psnrs = []
    
    for frame, gt_frame in zip(frames, gt_frames):
        mse = np.mean((frame - gt_frame)**2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnrs.append(psnr)
    
    return {
        'mean_psnr': float(np.mean(psnrs)),
        'std_psnr': float(np.std(psnrs)),
        'psnrs': [float(p) for p in psnrs]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to rendered video')
    parser.add_argument('--gt_video', type=str, default=None, help='Path to ground truth video (for PSNR)')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()
    
    print(f"Loading video: {args.video}")
    frames = load_video(args.video, args.start_frame, args.end_frame)
    print(f"Loaded {len(frames)} frames")
    
    results = {}
    
    # Optical Flow Variance
    print("\n=== Computing Optical Flow Variance ===")
    flow_metrics = compute_optical_flow_variance(frames)
    results['optical_flow'] = flow_metrics
    print(f"  Temporal variance: {flow_metrics['temporal_variance']:.6f}")
    print(f"  Spatial variance:  {flow_metrics['avg_spatial_variance']:.6f}")
    
    # Warping Error
    print("\n=== Computing Warping Error ===")
    warp_metrics = compute_warping_error(frames)
    results['warping'] = warp_metrics
    print(f"  Mean warping error: {warp_metrics['mean_warping_error']:.6f}")
    print(f"  Std warping error:  {warp_metrics['std_warping_error']:.6f}")
    
    # Temporal LPIPS
    print("\n=== Computing Temporal LPIPS ===")
    lpips_metrics = compute_temporal_lpips(frames)
    results['temporal_lpips'] = lpips_metrics
    print(f"  Mean LPIPS: {lpips_metrics['mean_lpips']:.6f}")
    print(f"  Std LPIPS:  {lpips_metrics['std_lpips']:.6f}")
    
    # PSNR (if GT provided)
    if args.gt_video:
        print(f"\n=== Computing PSNR (vs GT) ===")
        gt_frames = load_video(args.gt_video, args.start_frame, args.end_frame)
        
        if len(frames) != len(gt_frames):
            print(f"WARNING: Frame count mismatch ({len(frames)} vs {len(gt_frames)})")
            min_len = min(len(frames), len(gt_frames))
            frames = frames[:min_len]
            gt_frames = gt_frames[:min_len]
        
        psnr_metrics = compute_psnr(frames, gt_frames)
        results['psnr'] = psnr_metrics
        print(f"  Mean PSNR: {psnr_metrics['mean_psnr']:.2f} dB")
        print(f"  Std PSNR:  {psnr_metrics['std_psnr']:.2f} dB")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        video_path = Path(args.video)
        output_path = video_path.parent / f"{video_path.stem}_temporal_metrics.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results saved to: {output_path} ===")
    
    # Summary
    print("\n" + "="*60)
    print("MOTION QUALITY SUMMARY")
    print("="*60)
    print(f"Temporal Flow Variance:  {flow_metrics['temporal_variance']:.6f}")
    print(f"Warping Error:           {warp_metrics['mean_warping_error']:.6f}")
    print(f"Temporal LPIPS:          {lpips_metrics['mean_lpips']:.6f}")
    if args.gt_video:
        print(f"Mean PSNR:               {psnr_metrics['mean_psnr']:.2f} dB")
    print("="*60)


if __name__ == '__main__':
    main()

