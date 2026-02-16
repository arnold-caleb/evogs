"""
Backward Compatibility Wrapper for Velocity Field

This file maintains backward compatibility with existing imports.
The actual implementation has been moved to scene/velocity/ for better organization.

Old usage (still works):
    from scene.velocity_field import velocity_network

New usage (recommended):
    from scene.velocity import velocity_network
"""

# Re-export everything from the new velocity module
from scene.velocity import (
    VelocityField,
    ODEIntegrator,
    velocity_network,
    normalize_quaternion,
    compute_divergence_loss,
    compute_trajectory_rollout,
    compute_velocity_magnitude_stats,
    compute_temporal_smoothness,
    visualize_velocity_field_slice
)

# Backward compatibility alias
normalize_quaternions = normalize_quaternion

__all__ = [
    'VelocityField',
    'ODEIntegrator',
    'velocity_network',
    'normalize_quaternion',
    'normalize_quaternions',  # Backward compat
    'compute_divergence_loss',
    'compute_trajectory_rollout',
    'compute_velocity_magnitude_stats',
    'compute_temporal_smoothness',
    'visualize_velocity_field_slice',
]
