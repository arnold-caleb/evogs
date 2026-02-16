"""Cut roasted beef - Optimized velocity field with tuned hyperparameters"""
_base_ = './base_velocity.py'

OptimizationParams = dict(
    iterations = 10000,  # Extended training
    densify_until_iter = 8000,
    lambda_velocity_coherence = 0.015,  # Fine-tuned coherence
)
