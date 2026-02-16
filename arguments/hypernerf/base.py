"""
Base configuration for all HyperNeRF dataset scenes.

HyperNeRF dataset: Captures topological changes and non-rigid deformations
- Uses nerfies data format
- Varied temporal resolutions (typically 80-150 frames)
- Higher deformation complexity than D-NeRF

This base config defines common settings shared across all HyperNeRF scenes.
Individual scene configs inherit from this and override temporal resolution.
"""

ModelParams = dict(
    dataset_type = 'nerfies',  # HyperNeRF uses nerfies format
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]  # Default temporal resolution
    },
    multires = [1, 2, 4],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes = 0.0001,
    render_process = True
)

OptimizationParams = dict(
    iterations = 14_000,
    batch_size = 2,
    coarse_iterations = 3000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 300000,
)

