"""
Adaptive Coherence Predictor for Velocity Field Training - CORRECTED VERSION

CRITICAL FIX: Uses integrated positions (trajectories) instead of instantaneous velocities
to properly capture accumulated divergence over integration spans.

Key Idea:
- Torso (slow, rigid) → High coherence weight
- Hands (fast, articulated) → Low coherence weight
- Automatically learned from data!

Author: Corrected based on compute_trajectory_coherence() logic
Date: November 6, 2025
"""

import torch
import torch.nn as nn


class AdaptiveCoherencePredictor(nn.Module):
    """
    Predicts per-Gaussian coherence weights based on local motion properties.
    
    CORRECTED: Now uses trajectory features (integrated positions) in addition
    to instantaneous velocities for more informed weight prediction.
    
    Inputs:
        - xyz: Canonical position [N, 3]
        - t: Time [N, 1]
        - velocity: Current velocity [N, 3]
        - neighbor_velocities: Velocities of k nearest neighbors [N, k, 3]
        - trajectory_divergence: Divergence after integration [N, k] (NEW!)
        - neighbor_distances: Canonical distances [N, k] (NEW!)
    
    Output:
        - coherence_weight: Per-Gaussian weight in [0, 1] [N, 1]
          - High weight = rigid motion (e.g., torso)
          - Low weight = articulated motion (e.g., fingers)
    """
    
    def __init__(self, input_dim=13, hidden_dim=32):
        """
        Args:
            input_dim: Input feature dimension (default: 13)
                - xyz: 3
                - t: 1
                - speed: 1
                - local_variance: 1
                - neighbor_speed_std: 1
                - neighbor_mean_vel: 3
                - trajectory_divergence: 1 (NEW!)
                - neighbor_dist_mean: 1 (NEW!)
                - neighbor_dist_std: 1 (NEW!)
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize with bias toward medium coherence (0.5)
        # This ensures gradual learning rather than extreme values early on
        with torch.no_grad():
            self.mlp[-2].bias.fill_(0.0)  # Sigmoid(0) = 0.5
    
    def forward(self, xyz, t, velocity, neighbor_velocities, 
                trajectory_divergence=None, neighbor_distances=None):
        """
        Args:
            xyz: [N, 3] canonical positions
            t: [N, 1] or scalar time
            velocity: [N, 3] current velocities
            neighbor_velocities: [N, k, 3] neighbor velocities
            trajectory_divergence: [N, k] OPTIONAL - divergence after integration
            neighbor_distances: [N, k] OPTIONAL - canonical distances to neighbors
        
        Returns:
            coherence_weights: [N, 1] per-Gaussian weights
        """
        N = xyz.shape[0]
        
        # Ensure t is [N, 1]
        if isinstance(t, float):
            t = torch.full((N, 1), t, device=xyz.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(N, 1)
        elif t.shape[0] == 1:
            t = t.expand(N, 1)
        
        # === Velocity-based features (instantaneous) ===
        speed = velocity.norm(dim=-1, keepdim=True)  # [N, 1]
        
        # Local motion variance: how different are neighbors?
        neighbor_mean = neighbor_velocities.mean(dim=1)  # [N, 3]
        velocity_diff = velocity - neighbor_mean  # [N, 3]
        local_variance = velocity_diff.norm(dim=-1, keepdim=True)  # [N, 1]
        
        # Neighbor speed variance: how consistent is motion nearby?
        neighbor_speeds = neighbor_velocities.norm(dim=-1)  # [N, k]
        neighbor_speed_std = neighbor_speeds.std(dim=-1, keepdim=True)  # [N, 1]
        
        # === Trajectory-based features (accumulated) ===
        features_list = [
            xyz,                    # [N, 3] - spatial position
            t,                      # [N, 1] - time
            speed,                  # [N, 1] - motion magnitude
            local_variance,         # [N, 1] - velocity difference from neighbors
            neighbor_speed_std,     # [N, 1] - consistency of nearby motion
            neighbor_mean           # [N, 3] - average neighbor velocity
        ]
        
        if trajectory_divergence is not None:
            # Key insight: If integrated positions diverge a lot, coherence should be LOW
            # (because these Gaussians are naturally articulated)
            traj_div_mean = trajectory_divergence.mean(dim=-1, keepdim=True)  # [N, 1]
            features_list.append(traj_div_mean)
        else:
            # Fallback: use zero feature
            features_list.append(torch.zeros(N, 1, device=xyz.device))
        
        if neighbor_distances is not None:
            # Spatial proximity: closer neighbors should have higher coherence
            dist_mean = neighbor_distances.mean(dim=-1, keepdim=True)  # [N, 1]
            dist_std = neighbor_distances.std(dim=-1, keepdim=True)  # [N, 1]
            features_list.extend([dist_mean, dist_std])
        else:
            # Fallback: zero features
            features_list.extend([
                torch.zeros(N, 1, device=xyz.device),
                torch.zeros(N, 1, device=xyz.device)
            ])
        
        # Concatenate all features
        features = torch.cat(features_list, dim=-1)  # [N, 13]
        
        # Predict coherence weight
        coherence_weights = self.mlp(features)  # [N, 1]
        
        return coherence_weights
    
    def get_coherence_statistics(self, coherence_weights):
        """Helper to analyze learned coherence distribution."""
        with torch.no_grad():
            stats = {
                'mean': coherence_weights.mean().item(),
                'std': coherence_weights.std().item(),
                'min': coherence_weights.min().item(),
                'max': coherence_weights.max().item(),
                'median': coherence_weights.median().item(),
            }
        return stats


def compute_adaptive_coherence_loss(
    gaussians,
    velocity_field,
    coherence_predictor,
    t,
    n_samples=1000,
    n_neighbors=8,
    base_lambda=0.02,
    integrated_positions_cache=None
):
    """
    Compute trajectory coherence loss with learned adaptive weights.
    
    CORRECTED: Now properly uses integrated positions (trajectories) instead of
    instantaneous velocities, matching the original compute_trajectory_coherence() logic.
    
    Args:
        gaussians: GaussianModel
        velocity_field: VelocityField module
        coherence_predictor: AdaptiveCoherencePredictor module
        t: Current time (float)
        n_samples: Number of Gaussians to sample
        n_neighbors: Number of nearest neighbors
        base_lambda: Base coherence weight (scales the learned weights)
        integrated_positions_cache: Optional pre-computed integrated positions from render
    
    Returns:
        total_loss: Weighted coherence loss (scalar)
        stats: Dict with loss breakdown and coherence statistics
    """
    device = gaussians._xyz.device
    N = gaussians._xyz.shape[0]
    canonical_positions_all = gaussians._xyz
    
    # Sample Gaussians
    if N > n_samples:
        sample_idx = torch.randperm(N, device=device)[:n_samples]
        canonical_positions = canonical_positions_all[sample_idx].detach()
    else:
        sample_idx = torch.arange(N, device=device)
        canonical_positions = canonical_positions_all.detach()
        n_samples = N
    
    # Find k-nearest neighbors in canonical space
    with torch.no_grad():
        dist_matrix = torch.cdist(canonical_positions, canonical_positions)
        _, indices = torch.topk(dist_matrix, k=n_neighbors+1, largest=False, dim=1)
        indices = indices[:, 1:]  # Remove self
        
        canonical_dists = dist_matrix.gather(1, indices)  # [n_samples, n_neighbors]
        sigma = canonical_dists.mean() * 0.5
    
    # === CRITICAL: Integrate to get TRAJECTORIES, not just velocities ===
    if integrated_positions_cache is not None:
        # FAST PATH: Reuse from render
        integrated_positions = integrated_positions_cache[sample_idx]
        neighbor_integrated = integrated_positions_cache[sample_idx[indices.flatten()]].reshape(
            n_samples, n_neighbors, 3
        )
    else:
        # SLOW PATH: Compute from scratch
        all_indices = torch.cat([sample_idx.unsqueeze(1), sample_idx[indices]], dim=1)
        all_indices_flat = all_indices.flatten()
        
        positions_to_integrate = canonical_positions_all[all_indices_flat]
        n_total = len(positions_to_integrate)
        t_tensor = torch.ones(n_total, 1, device=device) * t
        
        # Build initial state for integration
        initial_state = {
            'xyz': positions_to_integrate,
            'scale': None,
            'rotation': None,
            'opacity': None,
            'shs': None
        }
        
        t0 = torch.zeros(n_total, 1, device=device)
        integrated_state = velocity_field.integrate_velocity(
            initial_state,
            t0,
            t_tensor,
            method=velocity_field.ode_method_train,
            steps=velocity_field.ode_steps_train
        )
        
        integrated_flat = integrated_state['xyz']  # [n_total, 3]
        
        integrated_all = integrated_flat.reshape(n_samples, 1 + n_neighbors, 3)
        integrated_positions = integrated_all[:, 0, :]
        neighbor_integrated = integrated_all[:, 1:, :]
    
    # Trajectory divergence: accumulated displacement difference
    trajectory_divergence = integrated_positions.unsqueeze(1) - neighbor_integrated  # [n_samples, n_neighbors, 3]
    traj_cost = (trajectory_divergence ** 2).sum(dim=-1)  # [n_samples, n_neighbors]
    
    # === Query velocities for feature extraction ===
    t_tensor = torch.ones(n_samples, 1, device=device) * t
    _, vels_dict = velocity_field.query_velocity(canonical_positions, t_tensor)
    velocities = vels_dict['xyz']  # [n_samples, 3]
    
    # Get neighbor velocities
    neighbor_positions = canonical_positions_all[sample_idx[indices.flatten()]].reshape(
        n_samples, n_neighbors, 3
    )
    t_expanded = t_tensor.unsqueeze(1).expand(n_samples, n_neighbors, 1).reshape(-1, 1)
    _, neighbor_vels_dict = velocity_field.query_velocity(
        neighbor_positions.reshape(-1, 3),
        t_expanded
    )
    neighbor_velocities = neighbor_vels_dict['xyz'].reshape(n_samples, n_neighbors, 3)
    
    # === ADAPTIVE WEIGHTS: Predict per-Gaussian coherence ===
    # Now includes trajectory divergence as input!
    trajectory_divergence_norm = (trajectory_divergence ** 2).sum(dim=-1)  # [n_samples, n_neighbors]
    
    coherence_weights = coherence_predictor(
        xyz=canonical_positions,
        t=t_tensor,
        velocity=velocities,
        neighbor_velocities=neighbor_velocities,
        trajectory_divergence=trajectory_divergence_norm,  # NEW!
        neighbor_distances=canonical_dists  # NEW!
    )  # [n_samples, 1]
    
    # Distance weighting (closer neighbors matter more)
    distance_weights = torch.exp(-canonical_dists / sigma)  # [n_samples, n_neighbors]
    
    # Adaptive weighting (learned per-Gaussian)
    adaptive_weights = coherence_weights * distance_weights
    
    # Weighted trajectory coherence loss
    coherence_loss = (adaptive_weights * traj_cost).sum() / (adaptive_weights.sum() + 1e-8)
    
    # === ANTI-COLLAPSE: Entropy regularization ===
    # Prevent network from outputting all 0s or all 1s
    entropy_loss = -torch.mean(
        coherence_weights * torch.log(coherence_weights + 1e-8) + 
        (1 - coherence_weights) * torch.log(1 - coherence_weights + 1e-8)
    )
    
    # Scale by base lambda
    total_loss = base_lambda * coherence_loss - 0.01 * base_lambda * entropy_loss
    
    # Statistics for logging
    stats = {
        'coherence_loss': coherence_loss.item(),
        'entropy': entropy_loss.item(),
        'total_loss': total_loss.item(),
        'coherence_weights': coherence_predictor.get_coherence_statistics(coherence_weights),
    }
    
    return total_loss, stats


# ============================================================================
# VISUALIZATION: Sanity check that network is learning meaningful patterns
# ============================================================================

def visualize_coherence_weights(gaussians, coherence_predictor, velocity_field, t, save_path='coherence_weights.png'):
    """
    Render a heatmap of learned coherence weights to verify spatial variation.
    
    Expected pattern:
    - Torso/pelvis: High weights (0.8-1.0) → red
    - Arms/legs: Medium weights (0.4-0.6) → yellow
    - Hands/fingers: Low weights (0.0-0.2) → blue
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    N = gaussians._xyz.shape[0]
    positions = gaussians._xyz.detach()
    t_tensor = torch.ones(N, 1, device=positions.device) * t
    
    # Query velocities
    _, vels_dict = velocity_field.query_velocity(positions, t_tensor)
    velocities = vels_dict['xyz']
    
    # Find neighbors
    dist_matrix = torch.cdist(positions, positions)
    _, indices = torch.topk(dist_matrix, k=9, largest=False, dim=1)
    indices = indices[:, 1:]  # Remove self
    neighbor_velocities = velocities[indices]
    neighbor_distances = dist_matrix.gather(1, indices)
    
    # Predict weights
    with torch.no_grad():
        weights = coherence_predictor(
            xyz=positions,
            t=t_tensor,
            velocity=velocities,
            neighbor_velocities=neighbor_velocities,
            neighbor_distances=neighbor_distances
        ).cpu().numpy()
    
    # Plot 3D scatter colored by weight
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    xyz = positions.cpu().numpy()
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                        c=weights[:, 0], cmap='coolwarm', 
                        vmin=0, vmax=1, s=1)
    
    ax.set_title(f'Adaptive Coherence Weights at t={t:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(scatter, label='Coherence Weight')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved coherence visualization to {save_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Coherence Predictor - CORRECTED VERSION")
    print("=" * 70)
    print("\nKey improvements over original:")
    print("  1. ✓ Uses integrated positions (trajectories) not velocities")
    print("  2. ✓ Includes trajectory divergence as input feature")
    print("  3. ✓ Reuses integrated_positions_cache from render (efficient!)")
    print("  4. ✓ Entropy regularization to prevent collapse")
    print("  5. ✓ Visualization tools for debugging")
    print("=" * 70)
    print("\nUsage:")
    print("  from scene.adaptive_coherence import AdaptiveCoherencePredictor, compute_adaptive_coherence_loss")
    print("  predictor = AdaptiveCoherencePredictor().cuda()")
    print("  loss, stats = compute_adaptive_coherence_loss(...)")
    print("=" * 70)
