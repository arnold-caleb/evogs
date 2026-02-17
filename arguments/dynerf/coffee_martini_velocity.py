"""Coffee martini - Neural ODE velocity field (EvoGS)"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # Multi-Anchor Training: integrate from nearest waypoint
    use_multi_anchor = True,
    anchor_checkpoints = {
        # NOTE: Update paths after running train_anchor.py for this scene
        0.00: "output/static_anchors/coffee_martini_frame0/chkpnt30000.pth",
        0.25: "output/static_anchors/coffee_martini_frame75/chkpnt30000.pth",
        0.50: "output/static_anchors/coffee_martini_frame150/chkpnt30000.pth",
        0.75: "output/static_anchors/coffee_martini_frame225/chkpnt30000.pth",
    },
)
