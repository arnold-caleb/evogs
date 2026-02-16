"""Cut roasted beef - Enhanced velocity coherence regularization"""
_base_ = './base_velocity.py'

OptimizationParams = dict(
    lambda_velocity_coherence = 0.02,  # Stronger coherence (2x base)
)
