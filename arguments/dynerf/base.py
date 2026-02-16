"""
Base configuration for all DyNeRF dataset scenes.

DyNeRF dataset: Real-world dynamic scenes captured with iPhone
- Multi-view captures of real scenes (cooking, pouring, etc.)
- Typically 12-20 frames, various camera viewpoints
- Higher resolution than D-NeRF synthetic scenes

This base config defines common settings shared across all DyNeRF scenes.
Individual scene configs inherit from this and override specific parameters.
"""

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1, 2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes = 0.0001,
    no_do = False,
    no_dshs = False,
    no_ds = False,
    empty_voxel = False,
    render_process = False,
    static_mlp = False
)

OptimizationParams = dict(
    dataloader = True,
    iterations = 14000,
    batch_size = 4,
    coarse_iterations = 3000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
)

