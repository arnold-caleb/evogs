"""Cut roasted beef - Future reconstruction with displacement field"""
_base_ = './base_displacement.py'

ModelHiddenParams = dict(
    future_reconstruction = True,
    train_time_max = 0.5,
    
    use_multi_anchor = True,
    forward_only_anchors = True,
    anchor_checkpoints = {
        # NOTE: Update paths to match your static checkpoints
        0.00: "output/static_4anchors/frame0_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.25: "output/static_4anchors/frame75_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.50: "output/static_4anchors/frame150_YYYYMMDD_HHMMSS/chkpnt30000.pth",
    },
    lambda_anchor = 1.0,
    
    no_ds = True,
    no_dr = True,
)

OptimizationParams = dict(
    batch_size = 1,
    iterations = 14000,
    densify_until_iter = 0,
    pruning_from_iter = 99999999,
)
