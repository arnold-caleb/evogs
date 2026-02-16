"""Cut roasted beef - Neural ODE velocity field (EvoGS)"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # Multi-Anchor Training: integrate from nearest waypoint
    use_multi_anchor = True,
    anchor_checkpoints = {
        0.00: "output/static_4anchors/frame0_20251104_103403/chkpnt30000.pth",
        0.25: "output/static_4anchors/frame75_20251104_103403/chkpnt30000.pth",
        0.50: "output/static_4anchors/frame150_20251104_103403/chkpnt30000.pth",
        0.75: "output/static_4anchors/frame225_20251104_103403/chkpnt30000.pth",
    },
)
