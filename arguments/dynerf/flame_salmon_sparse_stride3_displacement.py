"""Flame salmon - Sparse stride 3 with displacement field"""
_base_ = './base_displacement.py'

ModelHiddenParams = dict(
    sparse_supervision = True,
    supervised_frame_stride = 3,
    supervised_frame_offset = 1,
)

OptimizationParams = dict(
    batch_size = 1,
    lambda_coherence = 0.02,
    iterations = 14000,
)
