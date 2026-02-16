"""Cut roasted beef - Static-only training (no deformation)"""
_base_ = './base.py'

ModelHiddenParams = dict(
    static_mlp = True,  # Train static mask
    no_do = True,
    no_dshs = True,
    no_ds = True,
    no_dr = True,
    no_dx = True,
)

OptimizationParams = dict(
    batch_size = 2,
)
