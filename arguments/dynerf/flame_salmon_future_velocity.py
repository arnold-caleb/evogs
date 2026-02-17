"""Flame salmon - Future reconstruction (train on first half, predict second half)"""
_base_ = './base_future.py'

ModelHiddenParams = dict(
    anchor_checkpoints = {
        # NOTE: Update paths after running train_anchor.py for this scene
        # Only anchors in [0, 0.5] training window — t=0.75 is in the future!
        0.00: "output/static_anchors/flame_salmon_frame0/chkpnt30000.pth",
        0.25: "output/static_anchors/flame_salmon_frame75/chkpnt30000.pth",
        0.50: "output/static_anchors/flame_salmon_frame150/chkpnt30000.pth",
    },
)

