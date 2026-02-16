"""D-NeRF Jumping Jacks Scene with Velocity Field (75 frames)"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    kplanes_config={
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 75]
    },
)
