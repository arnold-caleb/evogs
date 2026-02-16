"""Cut roasted beef - RK4 integration for L40 (24GB) GPUs"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    # RK4 integration with reduced steps for 24GB GPU
    ode_method_train = 'rk4',
    ode_method_eval = 'rk4',
    ode_steps_train = 4,  # Reduced for 24GB GPU
    ode_steps_eval = 4,
    
    query_at_canonical = False,
    
    no_ds = False,
    no_dr = False,
    apply_rotation = True,
)

OptimizationParams = dict(
    batch_size = 1,  # Reduced for memory
    iterations = 7000,
)

# Expected memory: ~22GB (safe for 24GB L40)
