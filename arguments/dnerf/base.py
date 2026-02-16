"""
D-NeRF Dataset Configuration Base

This module provides base configurations for D-NeRF synthetic datasets.
All D-NeRF scenes use the Blender/NeRF Synthetic format.
"""

ModelParams = dict(
    dataset_type='Blender',  # D-NeRF uses Blender/NeRF Synthetic format
)

OptimizationParams = dict(
    coarse_iterations=3000,
    deformation_lr_init=0.00016,
    deformation_lr_final=0.0000016,
    deformation_lr_delay_mult=0.01,
    grid_lr_init=0.0016,
    grid_lr_final=0.000016,
    iterations=20000,
    pruning_interval=8000,
    percent_dense=0.01,
    render_process=False,
)

ModelHiddenParams = dict(
    multires=[1, 2],
    defor_depth=0,
    net_width=64,
    plane_tv_weight=0.0001,
    time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    weight_decay_iteration=0,
    bounds=1.6,
    train_time_max=0.75,  # Train on first 75% of frames for extrapolation evaluation
)

