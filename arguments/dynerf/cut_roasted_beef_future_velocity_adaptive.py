"""Cut roasted beef - Future reconstruction with adaptive coherence"""
_base_ = './cut_roasted_beef_future_velocity.py'

OptimizationParams = dict(
    lambda_coherence = 0.02,
    use_adaptive_coherence = True,  # Learn spatially-varying weights
)
