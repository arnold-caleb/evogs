"""Cut roasted beef - Velocity field with strong coherence regularization"""
_base_ = './base_velocity.py'

OptimizationParams = dict(
    lambda_velocity_coherence = 0.05,  # Very strong coherence (5x base)
)
