"""Cut roasted beef - Velocity field initialized from static checkpoint"""
_base_ = './base_velocity.py'

OptimizationParams = dict(
    iterations = 10000,
    coarse_iterations = 1,  # Skip coarse stage
)

# Note: Requires checkpoint_path to be set in training script:
# checkpoint_path = "output/static_frame0_hq/cut_roasted_beef_YYYYMMDD_HHMMSS/chkpnt30000.pth"
