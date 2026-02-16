"""Cut roasted beef - RK4 integration for A100 (40GB) GPUs"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # RK4 integration: 40-100x better accuracy than Euler
    ode_method_train = 'rk4',
    ode_method_eval = 'rk4',
    ode_steps_train = 6,  # Conservative for 40GB A100
    ode_steps_eval = 6,
    
    query_at_canonical = False,  # True ODE dynamics
    
    # Full geometry evolution
    no_ds = False,
    no_dr = False,
    apply_rotation = True,
)

OptimizationParams = dict(
    iterations = 7000,
)

# Expected memory: ~42GB (safe for 40GB A100)
# Can increase to 8-10 steps for better quality if memory allows
