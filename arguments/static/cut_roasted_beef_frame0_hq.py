# High-Quality Static 3D Gaussian Splatting Configuration for Frame 0
# Improved settings for better reconstruction quality
# Note: dataset_type='dynerf_static' should be passed via command line --dataset_type

ModelHiddenParams = dict(
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,  # Higher features (was 16)
        'resolution': [64, 64, 64, 1]  # Single timestep
    },
    multires = [1, 2, 4],  # More resolution levels (was [1])
    defor_depth = 0,  # No deformation MLP
    net_width = 128,  # Larger (was 64)
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes = 0,
    no_do = True,  # No opacity deformation
    no_dshs = True,  # No color deformation  
    no_ds = True,  # No scale deformation
    no_dr = True,  # No rotation deformation
    no_dx = True,  # No position deformation
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
    iterations = 30000,  # Longer training (was 15000)
    batch_size = 2,  # Smaller batch for stability
    coarse_iterations = 3000,  # Longer coarse phase
    densify_from_iter = 500,
    densify_until_iter = 20000,  # Much longer densification
    densify_grad_threshold_coarse = 0.0002,  # Standard threshold for static 3DGS
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    densification_interval = 100,
    pruning_from_iter = 500,
    pruning_interval = 100,
    opacity_reset_interval = 3000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    percent_dense = 0.01,
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 30000,
    # Standard Gaussian parameters
    feature_lr = 0.0025,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
    # No deformation learning rates (set to 0)
    deformation_lr_init = 0.0,
    deformation_lr_final = 0.0,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0,
    grid_lr_final = 0.0,
    # Higher SSIM weight for perceptual quality
    lambda_dssim = 0.3,  # Higher (was 0.2)
)

