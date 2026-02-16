"""
Cut roasted beef - Future reconstruction (train on first half, predict second half)

This config inherits from base_future.py and only needs to specify the
scene-specific anchor checkpoints. All other settings come from the base.
"""
_base_ = './base_future.py'

ModelHiddenParams = dict(
    # Anchor checkpoints for cut_roasted_beef (300 frames total)
    # Train on t ∈ [0, 0.5] → frames 0-150
    # Test on t ∈ [0.5, 1.0] → frames 150-300
    anchor_checkpoints = {
        # NOTE: Update paths to match your static checkpoints
        0.00: "output/static_4anchors/frame0_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.25: "output/static_4anchors/frame75_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        0.50: "output/static_4anchors/frame150_YYYYMMDD_HHMMSS/chkpnt30000.pth",
        # NOT using t=0.75 (frame 225) - that's in the future!
    },
)
