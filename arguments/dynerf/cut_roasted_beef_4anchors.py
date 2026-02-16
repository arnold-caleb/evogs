"""Cut roasted beef - 4-anchor velocity field with forward-only integration"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # 4 anchors with forward-only integration (no blending)
    use_multi_anchor = True,
    forward_only_anchors = True,
    anchor_checkpoints = {
        # NOTE: Update paths to match your static checkpoints
        0.00: "output/static_4anchors/frame0_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.25: "output/static_4anchors/frame75_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.50: "output/static_4anchors/frame150_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.75: "output/static_4anchors/frame225_YYYYMMDD_HHMMSS/chkpnt30000.pth",
    },
    lambda_anchor = 1.0,
    
    integrate_rotation = False,
    query_at_canonical = True,
    
    # Position-only motion
    no_ds = True,
    no_dr = True,
    no_do = True,
    apply_rotation = False,
)

OptimizationParams = dict(
    batch_size = 1,
    iterations = 14000,
    
    # Disable densification
    densify_until_iter = 0,
    pruning_from_iter = 99999999,
)
