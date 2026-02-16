"""Cut roasted beef - Velocity field trained at canonical position"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    query_at_canonical = True,  # Query v(x₀,t) instead of v(x(t),t)
)
