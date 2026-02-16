"""Cut roasted beef - Velocity field with additional regularization"""
_base_ = './base_velocity.py'

OptimizationParams = dict(
    velocity_reg_weight = 0.001,  # Enable velocity magnitude regularization
)
