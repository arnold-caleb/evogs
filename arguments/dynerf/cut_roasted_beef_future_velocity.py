"""Cut roasted beef - Future reconstruction (train on first half, predict second half)"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # Future reconstruction: train on t ∈ [0, 0.5], test on t ∈ [0.5, 1.0]
    future_reconstruction = True,
    train_time_max = 0.5,
    
    # Use first-half anchors only
    use_multi_anchor = True,
    forward_only_anchors = True,
    anchor_checkpoints = {
        # NOTE: Update paths to match your static checkpoints
        0.00: "output/static_4anchors/frame0_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.25: "output/static_4anchors/frame75_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.50: "output/static_4anchors/frame150_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        # NOT using t=0.75 - that's in the future!
    },
    lambda_anchor = 1.0,
    
    integrate_rotation = False,
    query_at_canonical = True,
    
    no_ds = True,
    no_dr = True,
    no_do = True,
    apply_rotation = False,
)

OptimizationParams = dict(
    batch_size = 1,
    iterations = 14000,
    densify_until_iter = 0,
    pruning_from_iter = 99999999,
)
