"""
Velocity Field Utilities

Helper functions for velocity field operations, including:
- Divergence computation for incompressibility
- Trajectory visualization
- Coherence metrics
"""

import torch
import torch.nn.functional as F


def normalize_quaternions(q):
    """
    Normalize quaternions to unit length.
    
    During ODE integration, quaternions can drift from ||q|| = 1.
    This ensures valid rotations.
    
    Args:
        q (Tensor): [N, 4] quaternions
    
    Returns:
        Tensor: [N, 4] normalized quaternions
    """
    return F.normalize(q, dim=-1)


def compute_divergence_loss(velocity_field, xyz, t, n_samples=500):
    """
    Compute divergence of velocity field: ∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
    
    For incompressible flow (volume-preserving): ∇·v ≈ 0
    This prevents Gaussians from expanding (creating gaps) or collapsing (thinning).
    
    Uses finite differences to approximate divergence without requiring
    second-order derivatives through grid_sample.
    
    Args:
        velocity_field (VelocityField): The velocity field
        xyz (Tensor): [N, 3] sample positions
        t (float or Tensor): time value(s)
        n_samples (int): Number of points to sample for efficiency
    
    Returns:
        Tensor: scalar divergence loss (mean squared divergence)
    """
    # Sample for efficiency
    if xyz.shape[0] > n_samples:
        sample_idx = torch.randperm(xyz.shape[0], device=xyz.device)[:n_samples]
        xyz_sample = xyz[sample_idx]
    else:
        xyz_sample = xyz
    
    N = xyz_sample.shape[0]
    device = xyz_sample.device
    
    # Time tensor
    if isinstance(t, float):
        t_tensor = torch.ones(N, 1, device=device) * t
    else:
        t_tensor = t
    
    # Query velocity at sample points
    with torch.no_grad():
        _, velocities_center = velocity_field.query_velocity(xyz_sample, t_tensor)
        v_center = velocities_center['xyz']  # [N, 3]
    
    # Finite difference approximation
    epsilon = 0.01  # Small perturbation for finite differences
    div = torch.zeros(N, device=device)
    
    for i in range(3):  # x, y, z dimensions
        # Perturb position in dimension i
        xyz_plus = xyz_sample.clone()
        xyz_plus[:, i] += epsilon
        
        xyz_minus = xyz_sample.clone()
        xyz_minus[:, i] -= epsilon
        
        # Query velocities at perturbed positions
        with torch.no_grad():
            _, v_plus = velocity_field.query_velocity(xyz_plus, t_tensor)
            _, v_minus = velocity_field.query_velocity(xyz_minus, t_tensor)
        
        # Finite difference: ∂vi/∂xi ≈ (vi(x+ε) - vi(x-ε)) / (2ε)
        dv_dx = (v_plus['xyz'][:, i] - v_minus['xyz'][:, i]) / (2 * epsilon)
        div += dv_dx
    
    # Penalize non-zero divergence
    div_loss = (div ** 2).mean()
    
    return div_loss


def compute_trajectory_rollout(velocity_field, integrator, xyz_init, t_span, n_steps=50):
    """
    Compute trajectory rollout for visualization.
    
    Integrates from t_span[0] to t_span[1] and returns intermediate positions.
    
    Args:
        velocity_field (VelocityField): Velocity field
        integrator (ODEIntegrator): ODE integrator
        xyz_init (Tensor): [N, 3] initial positions
        t_span (tuple): (t_start, t_end)
        n_steps (int): Number of intermediate steps
    
    Returns:
        Tensor: [n_steps, N, 3] trajectory positions
    """
    t_start, t_end = t_span
    t_values = torch.linspace(t_start, t_end, n_steps, device=xyz_init.device)
    
    trajectories = []
    current_state = {'xyz': xyz_init, 'scale': None, 'rotation': None, 
                    'opacity': None, 'shs': None}
    
    for i in range(n_steps - 1):
        # Integrate one step
        next_state = integrator.integrate(current_state, t_values[i].item(), 
                                         t_values[i+1].item())
        trajectories.append(current_state['xyz'])
        current_state = next_state
    
    trajectories.append(current_state['xyz'])
    
    return torch.stack(trajectories, dim=0)


def compute_velocity_magnitude_stats(velocity_field, xyz, t):
    """
    Compute velocity magnitude statistics for analysis.
    
    Args:
        velocity_field (VelocityField): Velocity field
        xyz (Tensor): [N, 3] positions
        t (float or Tensor): time
    
    Returns:
        dict: Statistics including mean, std, max, min velocity magnitudes
    """
    N = xyz.shape[0]
    device = xyz.device
    
    if isinstance(t, float):
        t_tensor = torch.ones(N, 1, device=device) * t
    else:
        t_tensor = t
    
    with torch.no_grad():
        _, velocities = velocity_field.query_velocity(xyz, t_tensor)
        v_xyz = velocities['xyz']
        v_mag = v_xyz.norm(dim=-1)
    
    stats = {
        'mean': v_mag.mean().item(),
        'std': v_mag.std().item(),
        'max': v_mag.max().item(),
        'min': v_mag.min().item(),
        'median': v_mag.median().item(),
    }
    
    return stats


def compute_temporal_smoothness(velocity_field, xyz, t_samples):
    """
    Compute temporal smoothness of velocity field.
    
    Measures how smoothly the velocity changes over time at fixed positions.
    
    Args:
        velocity_field (VelocityField): Velocity field
        xyz (Tensor): [N, 3] fixed positions
        t_samples (Tensor): [T] time samples
    
    Returns:
        Tensor: scalar smoothness loss (mean temporal variation)
    """
    N = xyz.shape[0]
    device = xyz.device
    
    velocities_over_time = []
    for t in t_samples:
        t_tensor = torch.ones(N, 1, device=device) * t
        _, velocities = velocity_field.query_velocity(xyz, t_tensor)
        velocities_over_time.append(velocities['xyz'])
    
    velocities_over_time = torch.stack(velocities_over_time, dim=0)  # [T, N, 3]
    
    # Compute temporal differences
    temporal_diff = velocities_over_time[1:] - velocities_over_time[:-1]
    smoothness_loss = (temporal_diff ** 2).mean()
    
    return smoothness_loss


def visualize_velocity_field_slice(velocity_field, t, bounds, resolution=64, plane='xy'):
    """
    Visualize velocity field on a 2D slice.
    
    Args:
        velocity_field (VelocityField): Velocity field
        t (float): Time to visualize
        bounds (tuple): ((xmin, ymin, zmin), (xmax, ymax, zmax))
        resolution (int): Grid resolution
        plane (str): Slice plane ('xy', 'xz', or 'yz')
    
    Returns:
        tuple: (grid_points, velocities) for visualization
    """
    device = next(velocity_field.parameters()).device
    bounds_min, bounds_max = bounds
    
    # Create grid
    if plane == 'xy':
        x = torch.linspace(bounds_min[0], bounds_max[0], resolution, device=device)
        y = torch.linspace(bounds_min[1], bounds_max[1], resolution, device=device)
        z = torch.tensor([(bounds_min[2] + bounds_max[2]) / 2], device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_z = z.expand_as(grid_x)
    elif plane == 'xz':
        x = torch.linspace(bounds_min[0], bounds_max[0], resolution, device=device)
        z = torch.linspace(bounds_min[2], bounds_max[2], resolution, device=device)
        y = torch.tensor([(bounds_min[1] + bounds_max[1]) / 2], device=device)
        grid_x, grid_z = torch.meshgrid(x, z, indexing='ij')
        grid_y = y.expand_as(grid_x)
    else:  # yz
        y = torch.linspace(bounds_min[1], bounds_max[1], resolution, device=device)
        z = torch.linspace(bounds_min[2], bounds_max[2], resolution, device=device)
        x = torch.tensor([(bounds_min[0] + bounds_max[0]) / 2], device=device)
        grid_y, grid_z = torch.meshgrid(y, z, indexing='ij')
        grid_x = x.expand_as(grid_y)
    
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    t_tensor = torch.ones(grid_points.shape[0], 1, device=device) * t
    
    with torch.no_grad():
        _, velocities = velocity_field.query_velocity(grid_points, t_tensor)
        v_grid = velocities['xyz'].reshape(resolution, resolution, 3)
    
    return grid_points.reshape(resolution, resolution, 3), v_grid

