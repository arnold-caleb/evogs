"""
Base configuration for sparse temporal supervision experiments.

Sparse supervision trains on only a subset of frames to test the velocity
field's ability to interpolate across larger temporal gaps. This encourages
the model to learn smooth, temporally coherent dynamics rather than 
memorizing per-frame deformations.
"""

_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # Sparse temporal supervision
    sparse_supervision = True,
    supervised_frame_stride = 3,  # Train on every 3rd frame
    supervised_frame_offset = 1,  # Start at frame 1
    
    # Query at canonical for stable training
    query_at_canonical = True,
    
    # Position-only motion (prevents disappearing Gaussians)
    no_dx = False,  # Learn position velocities
    no_ds = True,   # Keep scales FIXED
    no_dr = True,   # Keep rotations FIXED
    no_do = True,   # Keep opacity FIXED
    apply_rotation = False,
)

OptimizationParams = dict(
    batch_size = 1,
    iterations = 14000,
    
    # Spatial coherence: force nearby Gaussians to move together
    lambda_coherence = 0.02,
    use_adaptive_coherence = False,
    
    # Disable densification (maintain correspondence)
    densify_until_iter = 0,
    pruning_from_iter = 99999999,
    
    # Learning rates
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.000016,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.00016,
    
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
)

