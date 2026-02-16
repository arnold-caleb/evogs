# Static 3D Gaussian Splatting Configuration for Frame 0
# This trains ONLY on the first frame (all viewpoints) to get G₀

# No deformation network - this is static 3D GS
# We still need minimal config to avoid errors, but won't train it (lr=0)
ModelHiddenParams = dict(
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,  # Minimal
        'resolution': [32, 32, 32, 1]  # Minimal - single timestep
    },
    multires = [1],  # Minimal single resolution
    defor_depth = 0,  # No deformation MLP
    net_width = 64,  # Minimal
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
    render_process = False,  # Not needed for static
    static_mlp = False,
    is_static = True,  # NEW FLAG: indicates static training
)

OptimizationParams = dict(
    dataloader = True,
    iterations = 15000,  # Shorter - single frame training
    batch_size = 4,  # Can use larger batch since single frame
    coarse_iterations = 1000,
    densify_from_iter = 500,
    densify_until_iter = 8000,  # Densify less
    densify_grad_threshold_coarse = 0.0002,  # Standard threshold
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    densification_interval = 100,
    opacity_reset_interval = 3000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    percent_dense = 0.01,
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 15000,
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
    # SSIM for quality
    lambda_dssim = 0.2,
)

