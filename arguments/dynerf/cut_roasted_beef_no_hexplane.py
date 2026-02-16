"""Cut roasted beef - No hexplane grid (MLP-only deformation)"""
_base_ = './base.py'

ModelHiddenParams = dict(
    no_grid = True,  # Disable hexplane, use pure MLP
)

OptimizationParams = dict(
    batch_size = 2,
)
