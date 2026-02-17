"""
Base configuration for velocity field models on DyNeRF scenes.

This extends base.py with EvoGS-specific settings for continuous dynamics modeling.

Key features:
- Neural ODE integration for smooth motion
- Velocity coherence regularization
- Geometry-focused evolution (freezes opacity to prevent holes)
- Minimal Gaussian updates (velocity field does the work)
"""

_base_ = './base.py'

ModelHiddenParams = dict(
    use_velocity_field = True,  # Enable Neural ODE velocity field
    integrate_color_opacity = False,  # Keep color/opacity canonical
    time_smoothness_weight = 0.00001,
    static_mlp = False,
    velocity_activation_iter = 0,
    # train_time_max = 0.5,
    sparse_supervision = True,
    supervised_frame_stride = 3,  # Train on every 3rd frame
    supervised_frame_offset = 1,  # Start at frame 1
    
    # ODE Integration
    use_adjoint = False,  # Standard odeint (adjoint mode can be unstable)
    
    # Query mode: enables annealing from canonical to current
    query_at_canonical = False,
    
    # Velocity field: GEOMETRY ONLY
    no_dx = False,  # Position velocity
    no_ds = False,  # Scale velocity  
    no_dr = False,  # Rotation velocity
    no_do = True,   # FREEZE opacity (prevents holes!)
    apply_rotation = True,
)

OptimizationParams = dict(
    batch_size = 2,  # DyNeRF scenes typically use smaller batch
    iterations = 7000,
    coarse_iterations = 0,  # Skip coarse stage (loading from static checkpoint)
    
    # Velocity coherence prevents "spiky" artifacts
    lambda_velocity_coherence = 0.01,
    
    # DENSIFICATION: Disabled when using --start_checkpoint with pre-optimized Gaussians.
    # The velocity field handles dynamics; adding/removing points breaks correspondence
    # with anchor Gaussians and introduces noise into an already-converged representation.
    densify_until_iter = 0,         # DISABLED: don't add new Gaussians
    densify_from_iter = 99999999,
    densification_interval = 200,
    densify_grad_threshold_fine_init = 0.0004,
    densify_grad_threshold_after = 0.0004,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    pruning_from_iter = 99999999,   # DISABLED: don't prune Gaussians
    pruning_interval = 200,
    opacity_reset_interval = 99999999,  # DISABLED: fatal with opacity_lr=0!
    add_point = False,
    
    velocity_reg_weight = 0.0,
    temporal_stride = 2,
    
    # DEFORMATION NETWORK LRs
    deformation_lr_init = 0.00053,
    deformation_lr_final = 0.000053,
    grid_lr_init = 0.00053,
    grid_lr_final = 0.000053,
    
    # MINIMAL GAUSSIAN UPDATES (velocity field handles dynamics)
    position_lr_init = 0.00001,
    position_lr_final = 0.000001,
    feature_lr = 0.0,      # FROZEN
    opacity_lr = 0.0,      # FROZEN
    scaling_lr = 0.0001,   # Very small
    rotation_lr = 0.00005, # Very small
)

