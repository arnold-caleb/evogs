"""
Base configuration for future reconstruction experiments on DyNeRF scenes.

Future reconstruction tests the velocity field's ability to extrapolate beyond
observed training data. We train on the first N% of frames (e.g., first 50%)
and evaluate on the remaining frames. This requires learning dynamics that
generalize to unseen future timesteps.

Key challenges:
- Integration drift accumulates over long rollouts
- No direct supervision on future frames
- Requires strong temporal smoothness and coherence

Typical workflow:
1. Train static Gaussians at multiple anchor frames (e.g., frame 0, 75, 150)
2. Train velocity field on first N% of frames with anchor constraints
3. Evaluate by forward-integrating from last training frame into future
"""

_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # Future reconstruction: train on first 50% of frames, predict rest
    future_reconstruction = True,
    train_time_max = 0.5,  # Override this in scene-specific configs if needed
    
    # Multi-anchor constraints help reduce drift
    use_multi_anchor = True,
    forward_only_anchors = True,  # Only use anchors in training window
    lambda_anchor = 1.0,
    
    # Query at canonical for more stable extrapolation
    query_at_canonical = True,
    
    # Conservative integration to prevent drift
    integrate_rotation = False,  # Keep rotations fixed for stability
    
    # Position-only motion (most conservative setting)
    no_dx = False,  # Position velocity (ONLY thing we evolve)
    no_ds = True,   # FREEZE scale
    no_dr = True,   # FREEZE rotation
    no_do = True,   # FREEZE opacity
    apply_rotation = False,
    
    # Dense supervision on the training window (no sparse skipping)
    sparse_supervision = False,
)

OptimizationParams = dict(
    batch_size = 1,  # Small batch for stability
    coarse_iterations = 0,  # Skip coarse stage (we load from static checkpoint)
    iterations = 14000,
    
    # Strong coherence to prevent drift
    lambda_coherence = 0.05,  # Higher than standard velocity training
    lambda_velocity_coherence = 0.02,
    
    # Disable densification (maintain correspondence across time)
    densify_until_iter = 0,
    pruning_from_iter = 99999999,
    
    # Conservative learning rates
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.000016,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.00016,
    
    # Frozen Gaussian parameters (velocity field does all the work)
    position_lr_init = 0.0,
    position_lr_final = 0.0,
    feature_lr = 0.0,
    opacity_lr = 0.0,
    scaling_lr = 0.0,
    rotation_lr = 0.0,
)


