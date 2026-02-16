"""
Configuration for training evolution operator on cut_roasted_beef scene.

Phase 3: Train operator to evolve G₀ through time.
"""

from argparse import Namespace


def get_operator_config():
    """Configuration for operator training."""
    config = Namespace()
    
    # ========== Paths ==========
    config.source_path = "data/dynerf/cut_roasted_beef"
    config.model_path = "output/operator_training/cut_roasted_beef"
    
    # Grid G₀ from Phase 2
    config.G0_path = "output/static_frame0_hq/cut_roasted_beef_20251017_013347/hexplane_grid/G0_iter30000.pth"
    
    # Static Gaussians from Phase 1 (for decoder initialization)
    config.static_gaussians_path = "output/static_frame0_hq/cut_roasted_beef_20251017_013347/point_cloud/iteration_30000/point_cloud.ply"
    
    # ========== Training ==========
    config.iterations = 30000
    config.save_interval = 5000
    config.test_interval = 2000
    config.checkpoint_interval = 5000
    
    # ========== Evolution Operator ==========
    config.operator_channels = 96  # Match grid feature dimension
    config.operator_depth = 3      # Number of conv layers
    config.operator_kernel_size = 3
    
    # Learning rates
    config.operator_lr = 1e-3      # Operator network
    config.G0_lr = 1e-4            # Fine-tune initial grid
    config.decoder_lr = 5e-4       # Decoder network (grid → Gaussians)
    
    # ========== ODE Integration ==========
    config.ode_method = 'dopri5'   # Adaptive Runge-Kutta
    config.ode_rtol = 1e-3
    config.ode_atol = 1e-4
    config.ode_max_steps = 100     # Prevent infinite loops
    
    # ========== Temporal Sampling ==========
    config.temporal_batch_size = 4  # Number of frames per batch
    config.max_time = 1.0              # Normalize time to [0, 1]
    config.random_time_sampling = True # Random vs sequential
    
    # ========== Loss Weights ==========
    config.lambda_photometric = 1.0  # L1 + SSIM on rendered images
    config.lambda_l1 = 0.8
    config.lambda_ssim = 0.2
    config.lambda_temporal_smooth = 0.01  # Smoothness across time
    config.lambda_grid_reg = 1e-5         # Regularize grid values
    
    # ========== Rendering ==========
    config.white_background = False
    config.render_resolution = 1.0  # Scale factor for rendering
    
    # ========== Optimization ==========
    config.gradient_clip = 1.0     # Prevent exploding gradients
    config.lr_decay_start = 15000
    config.lr_decay_end = 30000
    config.lr_decay_factor = 0.1
    
    # ========== Logging ==========
    config.log_interval = 100
    config.wandb = False  # Enable Weights & Biases logging
    config.wandb_project = "gaussian-splatting-operator"
    
    return config

