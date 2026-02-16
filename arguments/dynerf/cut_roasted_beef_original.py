"""Cut roasted beef - Original 4DGaussians baseline"""
_base_ = './base.py'

ModelHiddenParams = dict(
    # Original 4DGaussians configuration
    defor_depth = 1,
    multires = [1, 2, 4],
)

OptimizationParams = dict(
    batch_size = 2,
)
