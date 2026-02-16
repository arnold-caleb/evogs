"""
D-NeRF Velocity Field Configuration

This configuration enables Neural ODE velocity field for dynamic scenes.
Use this for training with continuous dynamics representation.
"""

_base_ = './base.py'

ModelHiddenParams = dict(
    use_velocity_field=True,
    integrate_color_opacity=False,  # Only integrate geometry, keep color/opacity canonical
    time_smoothness_weight=0.00001,  # Reduced to allow fast motion
    multires=[1, 2],
    static_mlp=True,  # Enable static/dynamic mask
    velocity_activation_iter=5000,  # Start velocity at fine stage
)

OptimizationParams = dict(
    batch_size=1,
    iterations=20000,
    coarse_iterations=5000,
    densify_until_iter=4500,
    velocity_reg_weight=0.1,
    temporal_stride=1,  # Train on all frames
)

