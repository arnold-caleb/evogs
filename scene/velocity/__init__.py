"""
EvoGS Velocity Field Module

This module implements velocity-based dynamics for Gaussian Splatting, treating
dynamic scenes as true dynamical systems integrated via ODEs.

Key Contribution:
-----------------
Unlike displacement-based methods (4DGS) that predict Δx = f(x, t), we predict
velocities dx/dt = v(x, t) and integrate via Neural ODE. This ensures:
1. Temporal coherence: trajectories are C¹ continuous
2. Physical plausibility: satisfies dynamical system properties
3. Better extrapolation: can predict beyond training frames

Modules:
--------
- field: Core VelocityField architecture
- integration: ODE integration methods
- network: Complete velocity_network with positional encoding
- utils: Helper functions for analysis and visualization

Example Usage:
--------------
```python
from scene.velocity import velocity_network

# Create velocity-based deformation network
deform_net = velocity_network(args)

# Apply deformation at time t
xyz_deformed, scales, rotations, opacity, shs = deform_net(
    point=xyz_canonical,
    scales=scales_canonical,
    rotations=rotations_canonical,
    opacity=opacity_canonical,
    shs=shs_canonical,
    times_sel=t
)
```

References:
-----------
- 4D Gaussian Splatting (Wu et al. 2023) - displacement-based baseline
- Neural ODE (Chen et al. 2018) - continuous-depth networks
- Neural Flow Maps (Holynski et al. 2023) - velocity-based 2D dynamics
- EvoGS (ours) - velocity-based 3D Gaussian dynamics

Copyright (c) 2024
"""

__version__ = "1.0.0"

from .field import VelocityField
from .integration import ODEIntegrator
from .network import velocity_network
from .quaternion_utils import (
    quaternion_multiply,
    quaternion_derivative,
    angular_velocity_to_quaternion,
    normalize_quaternion,
    quaternion_geodesic_distance,
    quaternion_slerp
)
from .utils import (
    compute_divergence_loss,
    compute_trajectory_rollout,
    compute_velocity_magnitude_stats,
    compute_temporal_smoothness,
    visualize_velocity_field_slice
)

__all__ = [
    # Core components
    'VelocityField',
    'ODEIntegrator',
    'velocity_network',
    
    # Quaternion utilities
    'quaternion_multiply',
    'quaternion_derivative',
    'angular_velocity_to_quaternion',
    'normalize_quaternion',
    'quaternion_geodesic_distance',
    'quaternion_slerp',
    
    # Analysis utilities
    'compute_divergence_loss',
    'compute_trajectory_rollout',
    'compute_velocity_magnitude_stats',
    'compute_temporal_smoothness',
    'visualize_velocity_field_slice',
]

