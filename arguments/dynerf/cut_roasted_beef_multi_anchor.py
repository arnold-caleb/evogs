"""Cut roasted beef - Multi-anchor velocity field (3 anchors at t=0, 0.5, 1.0)"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # Multi-anchor constraints to prevent integration drift
    use_multi_anchor = True,
    anchor_checkpoints = {
        # NOTE: Update paths to match your static checkpoints
        0.0: "output/static_multi_anchor/frame0_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.5: "output/static_multi_anchor/frame150_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        1.0: "output/static_multi_anchor/frame299_YYYYMMDD_HHMMSS/chkpnt30000.pth",
    },
    lambda_anchor = 1.0,
    
    integrate_rotation = False,
    query_at_canonical = True,
    no_do = True,
)

OptimizationParams = dict(
    iterations = 14000,
)
