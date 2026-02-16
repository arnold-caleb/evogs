"""
Static Anchor Training Config (Frame 150, 299)

This config is used to train anchor Gaussians by:
1. Loading canonical Gaussians from frame 0
2. Optimizing ONLY positions to fit target frame
3. NO densification/pruning (keeps Gaussian count fixed)

Result: Same Gaussians as canonical, but moved to correct positions for frame 150/299
"""

ModelHiddenParams = dict(
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 1]
    },
    multires = [1, 2, 4],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes = 0,
    no_do = True,
    no_dshs = True,
    no_ds = True,
    no_dr = True,
    no_dx = True,
    no_grid = False,
    grid_pe = 0,
    timebase_pe = 4,
    posebase_pe = 10,
    scale_rotation_pe = 2,
    opacity_pe = 2,
    timenet_width = 64,
    timenet_output = 32,
    bounds = 1.6,
    empty_voxel = False,
    render_process = False,
    static_mlp = False,
    is_static = True,
)

OptimizationParams = dict(
    dataloader = True,
    iterations = 30000,
    batch_size = 2,
    coarse_iterations = 3000,
    
    # CRITICAL: Disable densification to maintain Gaussian correspondence
    densify_from_iter = 500,
    densify_until_iter = 0,  # No densification!
    densification_interval = 100,
    densify_grad_threshold_coarse = 0.0002,
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    
    # CRITICAL: Disable pruning
    pruning_from_iter = 99999999,  # Never prune!
    pruning_interval = 100,
    
    opacity_reset_interval = 30000,  # Disable
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    
    percent_dense = 0.01,
    
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 30000,
    
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
    
    deformation_lr_init = 0.0,
    deformation_lr_final = 0.0,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0,
    grid_lr_final = 0.0,
    
    lambda_dssim = 0.3,
)

