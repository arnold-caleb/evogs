"""
Base configuration for displacement field (non-ODE) experiments.

Displacement field predicts Δx = f(x,t) directly per timestep without ODE
integration. This is the baseline approach used in 4DGaussians.
"""

_base_ = './base.py'

ModelHiddenParams = dict(
    use_velocity_field = False,  # Displacement field mode
    integrate_color_opacity = False,
    integrate_rotation = False,
    
    time_smoothness_weight = 0.00001,
    static_mlp = False,
    velocity_activation_iter = 0,
    
    # ODE settings (if hybrid mode is enabled)
    use_adjoint = False,
    ode_method_train = 'euler',
    ode_method_eval = 'euler',
    ode_steps_train = 4,
    ode_steps_eval = 4,
    
    query_at_canonical = False,
    
    # Full deformation
    no_dx = False,
    no_ds = False,
    no_dr = False,
    no_do = True,
    no_doshs = True,
    apply_rotation = False,  # Additive displacement
)

OptimizationParams = dict(
    batch_size = 2,
    iterations = 3000,
    coarse_iterations = 1,
    
    # Conservative densification
    densify_until_iter = 2500,
    densify_from_iter = 500,
    densification_interval = 200,
    densify_grad_threshold_fine_init = 0.0004,
    densify_grad_threshold_after = 0.0004,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    pruning_from_iter = 500,
    pruning_interval = 200,
    opacity_reset_interval = 5000,
    add_point = False,
    
    temporal_stride = 1,
    
    # Reduced learning rates
    deformation_lr_init = 0.00053,
    deformation_lr_final = 0.000053,
    grid_lr_init = 0.00053,
    grid_lr_final = 0.000053,
    
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
)

