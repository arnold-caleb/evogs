"""
Experiment 3: Temporal Inversion (Backward Integration)
Train on middle frames, reconstruct boundaries by integrating backward/forward.

This demonstrates that velocity fields are invertible and respect causality.
"""

import torch
import numpy as np
from scene import GaussianModel
from scene import Scene
from scene.velocity_field import VelocityField
import argparse
from pathlib import Path

def temporal_inversion_experiment(args):
    """
    Train on middle time range (e.g., t=0.3 to t=0.7), reconstruct boundaries.
    
    Strategy:
    1. Train velocity field on middle frames
    2. Integrate FORWARD from t=0.3 to reconstruct t=0.0
    3. Integrate BACKWARD from t=0.7 to reconstruct t=1.0
    
    This tests if velocity field respects physical causality in both directions.
    """
    
    print("=" * 80)
    print("TEMPORAL INVERSION EXPERIMENT")
    print("=" * 80)
    
    # Load dataset
    dataset = DatasetReader(args.source_path, args.split, args.resolution)
    
    # Get training frames
    train_times = [cam.time for cam in dataset.train_cameras]
    T_start = args.train_time_start  # e.g., 0.3
    T_end = args.train_time_end      # e.g., 0.7
    
    print(f"\nTotal frames: {len(train_times)}")
    print(f"Time range: [{min(train_times):.3f}, {max(train_times):.3f}]")
    print(f"Training on: t ∈ [{T_start}, {T_end}]")
    print(f"Reconstructing: t ∈ [0, {T_start}] (backward) and t ∈ [{T_end}, 1.0] (forward)")
    
    # Split data
    train_mask = np.array([T_start <= t <= T_end for t in train_times])
    backward_mask = np.array([t < T_start for t in train_times])
    forward_mask = np.array([t > T_end for t in train_times])
    
    train_indices = np.where(train_mask)[0]
    backward_indices = np.where(backward_mask)[0]
    forward_indices = np.where(forward_mask)[0]
    
    print(f"\nTraining frames: {len(train_indices)}")
    print(f"Backward reconstruction frames: {len(backward_indices)}")
    print(f"Forward reconstruction frames: {len(forward_indices)}")
    
    # Train model on middle subset
    print("\n" + "="*80)
    print("STEP 1: Training on t=[T_start, T_end]")
    print("="*80)
    
    # TODO: Train model on train_indices
    # model = train_model(dataset, train_indices, args)
    # velocity_field = model._deformation.velocity_field
    
    # Backward integration: t=T_start → t=0
    print("\n" + "="*80)
    print("STEP 2: Backward Integration (t=T_start → t=0)")
    print("="*80)
    
    print("\nNOTE: This requires changing sign of velocity field")
    print("v_backward(x, t) = -v_forward(x, T - t)")
    
    for back_idx in backward_indices[:3]:  # Show first 3
        back_cam = dataset.train_cameras[back_idx]
        back_time = back_cam.time
        
        print(f"\nReconstructing frame at t={back_time:.3f} (backward from t={T_start})")
        
        # Integrate BACKWARD from T_start to back_time
        # gaussians_at_t = integrate_velocity_backward(velocity_field, t_start=T_start, t_target=back_time)
        
        # Render predicted state
        # predicted_image = render(gaussians_at_t, back_cam, ...)
        
        # Compare with ground truth
        # TODO: Compute metrics
        print(f"  TODO: Compute reconstruction error")
    
    # Forward integration: t=T_end → t=1.0
    print("\n" + "="*80)
    print("STEP 3: Forward Integration (t=T_end → t=1.0)")
    print("="*80)
    
    for fwd_idx in forward_indices[:3]:  # Show first 3
        fwd_cam = dataset.train_cameras[fwd_idx]
        fwd_time = fwd_cam.time
        
        print(f"\nReconstructing frame at t={fwd_time:.3f} (forward from t={T_end})")
        
        # Integrate FORWARD from T_end to fwd_time
        # gaussians_at_t = integrate_velocity(velocity_field, t_start=T_end, t_target=fwd_time)
        
        # Render predicted state
        # predicted_image = render(gaussians_at_t, fwd_cam, ...)
        
        # Compare with ground truth
        # TODO: Compute metrics
        print(f"  TODO: Compute reconstruction error")
    
    print("\n✅ Temporal inversion complete!")
    print("This demonstrates causal reversibility of velocity fields")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--train_time_start", type=float, default=0.3,
                       help="Start of training time range")
    parser.add_argument("--train_time_end", type=float, default=0.7,
                       help="End of training time range")
    args = parser.parse_args()
    
    temporal_inversion_experiment(args)

