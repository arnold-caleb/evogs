"""
Experiment 1: Sample Efficiency
Train on sparse temporal sampling and evaluate on all frames.

This demonstrates that velocity fields learn true dynamics rather than memorizing.
"""

import torch
import numpy as np
from scene.dataset_readers import DatasetReader
from train import render
import argparse
from pathlib import Path

def sample_efficiency_experiment(args):
    """
    Train on subset of frames, evaluate on all frames.
    
    Sampling strategies:
    - Uniform: Every Nth frame (e.g., every 5th = 60 frames)
    - Mid-range: Middle 1/3 of frames (t=0.33 to t=0.67)
    - Sparse: Random 20% of frames
    """
    
    print("=" * 80)
    print("SAMPLE EFFICIENCY EXPERIMENT")
    print("=" * 80)
    
    # Load dataset
    dataset = DatasetReader(args.source_path, args.split, args.resolution)
    
    # Get all frames
    all_frames = len(dataset.train_cameras)
    all_times = [cam.time for cam in dataset.train_cameras]
    
    print(f"\nTotal frames: {all_frames}")
    print(f"Time range: [{min(all_times):.3f}, {max(all_times):.3f}]")
    
    # Experiment 1: Uniform sampling (every Nth frame)
    n_samples = [300, 100, 60, 30]  # Different sampling densities
    
    results = {}
    
    for n_train in n_samples:
        print(f"\n{'='*80}")
        print(f"Experiment: Train on {n_train} frames (every {300//n_train}th frame)")
        print(f"{'='*80}")
        
        # Sample uniformly
        indices = np.linspace(0, all_frames-1, n_train, dtype=int)
        train_times = [all_times[i] for i in indices]
        
        print(f"Training on frames: {indices[:5]} ... {indices[-5:]}")
        print(f"Training times: {train_times[:3]} ... {train_times[-3:]}")
        
        # Train model on subset
        # TODO: Actually train a model here
        
        # Evaluate on ALL frames
        # TODO: Render all frames and compute metrics
        
        # Store results
        results[n_train] = {
            'train_indices': indices,
            'train_times': train_times,
            # 'metrics': ... (will fill after evaluation)
        }
    
    # Compare results
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for n_train, result in results.items():
        coverage = 100 * n_train / all_frames
        print(f"Trained on {n_train} frames ({coverage:.1f}%): TODO metrics")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--resolution", type=int, default=1)
    args = parser.parse_args()
    
    sample_efficiency_experiment(args)

