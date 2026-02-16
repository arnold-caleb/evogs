"""
Coffee martini - Future reconstruction experiment

Train on first 50% of frames, predict the rest.
Demonstrates temporal extrapolation capability of EvoGS.
"""
_base_ = './base_future.py'

ModelHiddenParams = dict(
    # Anchor checkpoints for coffee_martini (300 frames total)
    # Train on t ∈ [0, 0.5] → frames 0-150
    # Test on t ∈ [0.5, 1.0] → frames 150-300
    anchor_checkpoints = {
        # NOTE: Update paths to match your static checkpoints
        0.00: "output/static_anchors/coffee_martini_frame0/chkpnt30000.pth",
        0.25: "output/static_anchors/coffee_martini_frame75/chkpnt30000.pth",
        0.50: "output/static_anchors/coffee_martini_frame150/chkpnt30000.pth",
    },
)

# That's it! Everything else is inherited from base_future.py

