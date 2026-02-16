"""Cut roasted beef - Sparse stride 3 with adaptive coherence"""
_base_ = './base_sparse.py'

OptimizationParams = dict(
    use_adaptive_coherence = True,  # Learn spatially-varying weights
)
