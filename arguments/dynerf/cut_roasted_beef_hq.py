"""Cut roasted beef - High quality training with longer iterations"""
_base_ = './base.py'

OptimizationParams = dict(
    batch_size = 2,
    iterations = 20000,  # Extended training
    densify_until_iter = 15000,
)
