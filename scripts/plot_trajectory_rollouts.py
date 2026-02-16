"""
Plot Trajectory Rollouts from Trained Models

This script loads trained models and visualizes trajectory rollouts by integrating
the velocity field over time. It creates 3D plots showing how Gaussians move through
space according to the learned dynamics.

Usage:
    python scripts/plot_trajectory_rollouts.py \
        --model_path output/dnerf/bouncingballs_20251031_162435 \
        --iteration 20000 \
        --n_trajectories 3 \
        --n_steps 200 \
        --t_start 0.0 \
        --t_end 1.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args


def load_model(model_path, iteration, source_path=None):
    """Load trained Gaussian model directly from checkpoint files."""
    print(f"Loading model from {model_path} at iteration {iteration}")
    
    # Parse arguments to get config
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    
    # Set model path
    sys.argv = [
        'plot_trajectory_rollouts.py',
        '--model_path', model_path,
    ]
    if source_path:
        sys.argv.extend(['--source_path', source_path])
    
    args, _ = parser.parse_known_args()
    args = get_combined_args(parser)
    
    # Initialize model
    gaussians = GaussianModel(args.sh_degree, args)
    
    # Handle iteration=-1 (find latest)
    if iteration == -1:
        from utils.system_utils import searchForMaxIteration
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        if not os.path.exists(point_cloud_dir):
            raise FileNotFoundError(f"Point cloud directory not found: {point_cloud_dir}")
        iteration = searchForMaxIteration(point_cloud_dir)
        print(f"Found latest iteration: {iteration}")
    
    # Load checkpoint directly
    checkpoint_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}")
    ply_path = os.path.join(checkpoint_path, "point_cloud.ply")
    
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")
    
    print(f"Loading point cloud from {ply_path}")
    gaussians.load_ply(ply_path)
    
    print(f"Loading model checkpoint from {checkpoint_path}")
    gaussians.load_model(checkpoint_path)
    
    # Set AABB bounds from point cloud if needed
    # Get bounds from loaded positions
    positions = gaussians.get_xyz.detach()
    if positions.numel() > 0:
        xyz_max = positions.max(dim=0)[0].cpu().numpy()
        xyz_min = positions.min(dim=0)[0].cpu().numpy()
        
        # Set AABB for deformation network
        if hasattr(gaussians, '_deformation') and gaussians._deformation is not None:
            if hasattr(gaussians._deformation, 'deformation_net'):
                gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
                print(f"Set AABB: max={xyz_max}, min={xyz_min}")
            elif hasattr(gaussians._deformation, 'set_aabb'):
                gaussians._deformation.set_aabb(xyz_max, xyz_min)
                print(f"Set AABB: max={xyz_max}, min={xyz_min}")
    
    return gaussians, None, args  # Return None for scene since we don't need it


def compute_trajectory_rollout(gaussians, initial_positions, indices=None, t_start=0.0, t_end=1.0, n_steps=200, device='cuda'):
    """
    Compute trajectory rollouts by integrating the velocity field or querying displacement field.
    
    For VELOCITY FIELDS: Integrates the ODE dx/dt = v(x,t) using Euler or RK4 methods.
    This gives true continuous trajectories that respect the learned dynamics.
    
    For DISPLACEMENT FIELDS: Queries the field at each time step independently.
    This shows the learned per-timestep deformations but is not a true integration.
    
    Args:
        gaussians: GaussianModel with velocity or displacement field
        initial_positions: [N, 3] initial positions
        indices: Optional indices for matching Gaussian properties (for displacement fields)
        t_start: Starting time
        t_end: Ending time
        n_steps: Number of integration steps
        device: Device to use
    
    Returns:
        trajectories: List of [N, 3] position arrays, one per timestep
    """
    if not hasattr(gaussians, '_deformation') or gaussians._deformation is None:
        raise ValueError("Model does not have a deformation network!")
    
    deformation_net = gaussians._deformation
    is_velocity_field = hasattr(deformation_net, 'integrate_velocity')
    
    print(f"Computing trajectory rollout from t={t_start} to t={t_end} with {n_steps} steps...")
    if is_velocity_field:
        print(f"  Using VELOCITY FIELD: Integrating ODE dx/dt = v(x,t) (true dynamics)")
    else:
        print(f"  Using DISPLACEMENT FIELD: Querying at each time step (not true integration)")
    
    # Move to device
    initial_positions = initial_positions.to(device)
    deformation_net = deformation_net.to(device)
    deformation_net.eval()  # Set to eval mode
    
    # Time points
    times = np.linspace(t_start, t_end, n_steps)
    
    # Store trajectory
    trajectory = [initial_positions.cpu().numpy()]
    
    with torch.no_grad():
        if is_velocity_field:
            # VELOCITY FIELD: Integrate ODE
            current_positions = initial_positions.clone()
            dt = times[1] - times[0]
            
            for i in range(1, n_steps):
                t_current = times[i-1]
                t_next = times[i]
                
                # Prepare initial state
                initial_state = {
                    'xyz': current_positions,
                }
                
                # Integrate velocity field from t_current to t_next
                try:
                    final_state = deformation_net.integrate_velocity(
                        initial_state,
                        t_current,
                        t_next,
                        method='euler',  # Use Euler for speed, 'rk4' for accuracy
                        steps=1  # Single step per dt
                    )
                    
                    current_positions = final_state['xyz']
                    trajectory.append(current_positions.cpu().numpy())
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Step {i+1}/{n_steps}, t={t_next:.3f}")
                        
                except Exception as e:
                    print(f"  Error at step {i+1}: {e}")
                    # Use simple Euler step as fallback
                    t_tensor = torch.ones(current_positions.shape[0], 1, device=device) * t_current
                    _, velocities = deformation_net.query_velocity(current_positions, t_tensor)
                    current_positions = current_positions + dt * velocities['xyz']
                    trajectory.append(current_positions.cpu().numpy())
        else:
            # DISPLACEMENT FIELD: Query at each time point
            # Get canonical positions (at t=0 or initial time)
            canonical_positions = initial_positions.clone()
            
            # Get other Gaussian properties needed for displacement field
            # Only get properties for the sampled positions
            all_scales = gaussians.get_scaling.detach().to(device)
            all_rotations = gaussians.get_rotation.detach().to(device)
            all_opacity = gaussians.get_opacity.detach().to(device)
            all_shs = gaussians.get_features.detach().to(device)
            
            # Get properties for the sampled Gaussians
            if indices is not None:
                scales = all_scales[indices]
                rotations = all_rotations[indices]
                opacity = all_opacity[indices]
                shs = all_shs[indices]
            elif initial_positions.shape[0] < gaussians.get_xyz.shape[0]:
                # Fallback: use first N (approximate)
                scales = all_scales[:initial_positions.shape[0]]
                rotations = all_rotations[:initial_positions.shape[0]]
                opacity = all_opacity[:initial_positions.shape[0]]
                shs = all_shs[:initial_positions.shape[0]]
            else:
                scales = all_scales
                rotations = all_rotations
                opacity = all_opacity
                shs = all_shs
            
            for i in range(1, n_steps):
                t = times[i]
                t_tensor = torch.ones(canonical_positions.shape[0], 1, device=device) * t
                
                # Query displacement field
                try:
                    # The forward_dynamic method returns (means3D, scales, rotations, opacity, shs)
                    deformed = deformation_net.forward_dynamic(
                        canonical_positions,
                        scales,
                        rotations,
                        opacity,
                        shs,
                        times_sel=t_tensor
                    )
                    
                    # Extract positions (first element of tuple)
                    if isinstance(deformed, tuple):
                        current_positions = deformed[0]  # means3D
                    elif isinstance(deformed, dict):
                        current_positions = deformed['xyz']
                    else:
                        current_positions = deformed
                    
                    trajectory.append(current_positions.cpu().numpy())
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Step {i+1}/{n_steps}, t={t:.3f}")
                        
                except Exception as e:
                    print(f"  Error at step {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: just use canonical positions
                    trajectory.append(canonical_positions.cpu().numpy())
    
    return trajectory


def plot_trajectories(trajectories, output_path, title="Learned Dynamics: Trajectory Rollouts", 
                     linewidth=0.5, view_elev=10, view_azim=45):
    """
    Plot trajectories using the user's plotting style.
    
    Args:
        trajectories: List of [N, 3] numpy arrays, one per trajectory
        output_path: Path to save PDF
        title: Plot title
        linewidth: Width of trajectory lines (default: 0.5 for thinner lines)
        view_elev: Elevation angle for 3D view (default: 10 degrees, side view)
        view_azim: Azimuth angle for 3D view (default: 45 degrees)
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    for traj in trajectories:
        # traj is [n_steps, 3]
        # Use thinner lines for cleaner visualization
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=linewidth, alpha=0.7)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Auto-scale axes based on all trajectories
    all_points = np.vstack(trajectories)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # Set equal aspect ratio so trajectories don't appear distorted
    # This helps show the true shape of trajectories and prevents diagonal/tilted appearance
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    
    # Add 10% padding
    max_range *= 1.1
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle for side view (not top-down)
    # elev: elevation angle (0 = horizontal, 90 = top-down)
    # azim: azimuth angle (rotation around z-axis)
    ax.view_init(elev=view_elev, azim=view_azim)
    
    plt.tight_layout()
    
    # Save high-quality vector PDF for CVPR appendix
    pdf_path = output_path
    plt.savefig(pdf_path, format="pdf", dpi=300)
    print(f"✓ Saved PDF: {pdf_path}")
    
    # Optional: also save PNG for local preview
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300)
    print(f"✓ Saved PNG: {png_path}")
    
    plt.close()
    
    return pdf_path


def filter_center_points(positions, center_radius_ratio=0.3, min_distance_from_center=None):
    """
    Filter out points that are too close to the center of the scene.
    
    Args:
        positions: [N, 3] tensor of positions
        center_radius_ratio: Ratio of scene radius to filter (0.3 = filter inner 30%)
        min_distance_from_center: Minimum distance from center (if None, uses center_radius_ratio)
    
    Returns:
        mask: [N] boolean tensor, True for points to keep
    """
    # Compute center and radius
    center = positions.mean(dim=0)
    distances_from_center = torch.norm(positions - center, dim=1)
    max_distance = distances_from_center.max()
    
    if min_distance_from_center is None:
        min_distance = max_distance * center_radius_ratio
    else:
        min_distance = min_distance_from_center
    
    # Keep points that are far enough from center
    mask = distances_from_center > min_distance
    
    return mask


def sample_edge_points(positions, n_samples, center_radius_ratio=0.3):
    """
    Sample points preferentially from the edges/periphery of the scene.
    
    Args:
        positions: [N, 3] tensor of positions
        n_samples: Number of points to sample
        center_radius_ratio: Ratio of scene radius to consider as "center"
    
    Returns:
        indices: [n_samples] tensor of indices to sample
    """
    # Compute distances from center
    center = positions.mean(dim=0)
    distances_from_center = torch.norm(positions - center, dim=1)
    max_distance = distances_from_center.max()
    threshold = max_distance * center_radius_ratio
    
    # Weight by distance from center (prefer edge points)
    # Use squared distance to emphasize edge points more
    weights = (distances_from_center / max_distance) ** 2
    
    # Boost weights for points beyond threshold
    edge_mask = distances_from_center > threshold
    weights[edge_mask] *= 2.0
    
    # Normalize to probabilities
    weights = weights / weights.sum()
    
    # Sample with replacement using weights
    indices = torch.multinomial(weights, n_samples, replacement=False)
    
    return indices


def main():
    parser = argparse.ArgumentParser(description="Plot trajectory rollouts from trained models")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--source_path', type=str, default=None,
                       help='Path to source data (not required, only used for config parsing)')
    parser.add_argument('--iteration', type=int, default=20000,
                       help='Iteration to load')
    parser.add_argument('--n_trajectories', type=int, default=3,
                       help='Number of trajectories to plot')
    parser.add_argument('--n_steps', type=int, default=500,
                       help='Number of integration steps (increased for longer trajectories)')
    parser.add_argument('--t_start', type=float, default=0.0,
                       help='Starting time')
    parser.add_argument('--t_end', type=float, default=3.0,
                       help='Ending time (extended for long-term visualization)')
    parser.add_argument('--n_samples', type=int, default=300,
                       help='Number of Gaussians to sample for initial positions (reduced)')
    parser.add_argument('--filter_center', action='store_true',
                       help='Filter out points in the center of the scene')
    parser.add_argument('--center_radius_ratio', type=float, default=0.3,
                       help='Ratio of scene radius to filter as center (default: 0.3)')
    parser.add_argument('--sample_edges', action='store_true',
                       help='Sample points preferentially from edges/periphery')
    parser.add_argument('--linewidth', type=float, default=0.5,
                       help='Line width for trajectories (default: 0.5 for thin lines)')
    parser.add_argument('--view_elev', type=float, default=10,
                       help='Elevation angle for 3D view in degrees (default: 10, side view)')
    parser.add_argument('--view_azim', type=float, default=45,
                       help='Azimuth angle for 3D view in degrees (default: 45)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as model_path)')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = args.model_path
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    gaussians, _, model_args = load_model(args.model_path, args.iteration, args.source_path)
    
    # Get Gaussian positions
    positions = gaussians.get_xyz.detach()
    device = positions.device
    print(f"Loaded {positions.shape[0]} Gaussians")
    print(f"Device: {device}")
    
    # Check if model has velocity field
    if not hasattr(gaussians, '_deformation') or gaussians._deformation is None:
        raise ValueError("Model does not have a deformation network!")
    
    if not hasattr(gaussians._deformation, 'integrate_velocity'):
        print("WARNING: Model does not appear to use velocity field.")
        print("This script is designed for velocity field models.")
        print("Attempting to proceed anyway...")
    
    # Sample initial positions for trajectories
    print(f"\nSampling {args.n_samples} Gaussians for trajectory computation")
    
    if args.sample_edges:
        # Sample preferentially from edges
        print(f"  Using edge-biased sampling (center_radius_ratio={args.center_radius_ratio})")
        n_samples = min(args.n_samples, positions.shape[0])
        sample_indices = sample_edge_points(positions.cpu(), n_samples, args.center_radius_ratio)
        sample_indices = sample_indices.to(device)
        initial_positions = positions[sample_indices]
        n_actual_samples = n_samples
    elif args.filter_center:
        # Filter out center points first, then sample
        print(f"  Filtering center points (center_radius_ratio={args.center_radius_ratio})")
        mask = filter_center_points(positions, args.center_radius_ratio)
        filtered_positions = positions[mask]
        n_filtered = filtered_positions.shape[0]
        print(f"  After filtering: {n_filtered} points (from {positions.shape[0]})")
        
        n_samples = min(args.n_samples, n_filtered)
        if n_samples < n_filtered:
            # Get indices of filtered points
            filtered_indices = torch.where(mask)[0]
            sample_indices_filtered = torch.randperm(n_filtered, device=device)[:n_samples]
            sample_indices = filtered_indices[sample_indices_filtered]
            initial_positions = filtered_positions[sample_indices_filtered]
        else:
            filtered_indices = torch.where(mask)[0]
            sample_indices = filtered_indices
            initial_positions = filtered_positions
        n_actual_samples = initial_positions.shape[0]
    else:
        # Regular random sampling
        n_samples = min(args.n_samples, positions.shape[0])
        if n_samples < positions.shape[0]:
            sample_indices = torch.randperm(positions.shape[0], device=device)[:n_samples]
            initial_positions = positions[sample_indices]
        else:
            sample_indices = torch.arange(positions.shape[0], device=device)
            initial_positions = positions
        n_actual_samples = initial_positions.shape[0]
    
    print(f"  Selected {n_actual_samples} Gaussians")
    
    # Group into trajectories (each trajectory uses a subset of Gaussians)
    n_per_traj = max(1, n_actual_samples // args.n_trajectories)
    trajectories = []
    
    print(f"\nComputing {args.n_trajectories} trajectory rollouts...")
    print(f"  Steps per trajectory: {args.n_steps}")
    print(f"  Time range: [{args.t_start}, {args.t_end}] (duration: {args.t_end - args.t_start})")
    print(f"  Gaussians per trajectory: {n_per_traj}")
    
    for i in range(args.n_trajectories):
        start_idx = i * n_per_traj
        end_idx = min((i + 1) * n_per_traj, n_actual_samples)
        traj_positions = initial_positions[start_idx:end_idx]
        traj_indices = sample_indices[start_idx:end_idx] if n_actual_samples < positions.shape[0] else None
        
        print(f"\n  Trajectory {i+1}/{args.n_trajectories} ({traj_positions.shape[0]} Gaussians)...")
        
        # Compute trajectory rollout
        traj = compute_trajectory_rollout(
            gaussians,
            traj_positions,
            indices=traj_indices,
            t_start=args.t_start,
            t_end=args.t_end,
            n_steps=args.n_steps,
            device=device
        )
        
        # Convert to numpy array: [n_steps, n_gaussians, 3]
        traj_array = np.array(traj)
        
        # For plotting, we want to show individual Gaussian trajectories
        # So we transpose to [n_gaussians, n_steps, 3]
        traj_array = traj_array.transpose(1, 0, 2)
        
        # Add each Gaussian's trajectory
        for j in range(traj_array.shape[0]):
            trajectories.append(traj_array[j])  # [n_steps, 3]
    
    print(f"\n✓ Computed {len(trajectories)} individual trajectories")
    
    # Plot trajectories
    print("\n" + "=" * 60)
    print("PLOTTING TRAJECTORIES")
    print("=" * 60)
    
    model_name = os.path.basename(args.model_path.rstrip('/'))
    output_path = os.path.join(output_dir, f"trajectory_rollout_{model_name}_iter{args.iteration}.pdf")
    
    plot_trajectories(
        trajectories,
        output_path,
        title=f"Learned Dynamics: Trajectory Rollouts ({model_name})",
        linewidth=args.linewidth,
        view_elev=args.view_elev,
        view_azim=args.view_azim
    )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

