"""
Intrinsic Grid Evolution for 4D Gaussian Splatting.

Inspired by CORAL (Serrano et al., NeurIPS 2023), this module implements
self-evolving feature grids where the grid state evolves according to 
learned dynamics rather than being queried as a function of time.

Key idea: 
    Current: features = Grid(x,y,z,t) - time is input coordinate
    Ours:    Grid[t+dt] = Evolve(Grid[t]) - time emerges from evolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class GridEvolutionOperator(nn.Module):
    """
    Lightweight evolution operator: Grid → dGrid/dt
    
    Inspired by CORAL's Neural ODE approach (Section 4.2.1).
    Uses convolutions for local interactions with stability tricks:
    - Spectral normalization (bound Lipschitz constant)
    - Small output scale (prevent explosion)
    - Residual connections (gradient flow)
    
    Args:
        channels: Number of feature channels in grid (default: 32)
        kernel_size: Convolution kernel size (default: 3)
        depth: Number of conv layers (default: 2, keep small for stability)
    """
    
    def __init__(self, channels=32, kernel_size=3, depth=2):
        super().__init__()
        
        self.channels = channels
        self.depth = depth
        
        # Build lightweight conv layers
        layers = []
        for i in range(depth):
            conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
            # Spectral normalization for stability (prevents unbounded growth)
            conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
        
        self.convs = nn.ModuleList(layers)
        
        # Learnable but small output scale (stability trick)
        # Initialize small to prevent initial instability
        self.output_scale = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, t, grid_state):
        """
        Compute dGrid/dt = F(Grid)
        
        This is the evolution operator that defines how the grid evolves.
        Called by ODE solver at each integration step.
        
        Args:
            t: Current time (required by ODE solver, but we don't use it!)
            grid_state: [1, channels, H, W] - current grid state
            
        Returns:
            d_grid: [1, channels, H, W] - rate of change (dGrid/dt)
        """
        h = grid_state
        
        # Apply conv layers with ReLU activations
        for i, conv in enumerate(self.convs[:-1]):
            h = F.relu(conv(h))
        
        # Final conv (no activation to allow negative changes)
        delta = self.convs[-1](h)
        
        # Scale down to keep changes small (stability)
        # This is crucial: prevents ODE from exploding
        return self.output_scale * delta


class MinimalGridEvolution(nn.Module):
    """
    Minimal evolution operator for testing/debugging.
    
    Single conv layer - simplest possible dynamics.
    Use this first to validate the pipeline works!
    """
    
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.scale = 0.01  # Very small scale
        
    def forward(self, t, grid_state):
        """dGrid/dt = small × conv(Grid)"""
        return self.scale * self.conv(grid_state)


class DynamicHexPlaneField(nn.Module):
    """
    Self-evolving HexPlane field with intrinsic dynamics.
    
    Unlike standard HexPlane which queries Grid(x,y,z,t),
    this evolves the grid state: Grid[t] = Evolve(Grid[0], t)
    
    Inspired by CORAL's dynamics modeling (Section 4.2).
    
    Args:
        config: HexPlane configuration
        multires: Multi-resolution multipliers
        evolution_operator: The dynamics operator (default: GridEvolutionOperator)
        ode_solver: ODE integration method (default: 'dopri5' - adaptive)
    """
    
    def __init__(self, config, multires, 
                 evolution_operator=None,
                 ode_solver='dopri5'):
        super().__init__()
        
        self.config = config
        self.multires = multires
        self.ode_solver = ode_solver
        
        # Feature dimension
        self.feat_dim = config['output_coordinate_dim'] * len(multires)
        
        # Initial grid state at t=0 (learnable parameters)
        # This is our "canonical" grid that will evolve
        self.grid_initial = self._initialize_grids(config, multires)
        
        # Evolution operator (the learned dynamics)
        if evolution_operator is None:
            evolution_operator = GridEvolutionOperator(
                channels=config['output_coordinate_dim']
            )
        self.evolution_op = evolution_operator
        
        # Cache evolved states to avoid recomputation
        # Key: time value, Value: evolved grid
        self.evolution_cache = {}
        self.training_mode = True  # Clear cache during training
        
    def _initialize_grids(self, config, multires):
        """Initialize multi-resolution grids for all 6 HexPlanes"""
        grids = nn.ModuleList()
        
        # 6 planes: XY, XZ, YZ, XT, YT, ZT
        for res_mult in multires:
            # Scale spatial dims by resolution multiplier
            resolution = [
                config['resolution'][i] * res_mult if i < 3 
                else config['resolution'][i]
                for i in range(4)
            ]
            
            # Initialize grid parameters for this resolution
            # Following HexPlane initialization
            # We'll create 6 planes here (simplified for now)
            # TODO: Full HexPlane initialization
            
            # For now, create a single plane as example
            H, W = resolution[0], resolution[1]  
            C = config['output_coordinate_dim']
            
            grid = nn.Parameter(
                torch.randn(1, C, H, W) * 0.1
            )
            grids.append(grid)
        
        return grids
    
    def evolve_to_time(self, target_time):
        """
        Evolve grid from t=0 to target_time using Neural ODE.
        
        This is where the magic happens! The grid evolves according to
        learned dynamics, not by querying at a time coordinate.
        
        Args:
            target_time: Target time to evolve to [0, 1]
            
        Returns:
            evolved_grids: List of grid states at target_time
        """
        # Check cache (don't recompute during inference)
        cache_key = float(target_time)
        if not self.training_mode and cache_key in self.evolution_cache:
            return self.evolution_cache[cache_key]
        
        # Evolve each resolution level
        evolved_grids = []
        
        for grid_init in self.grid_initial:
            # Solve ODE: Grid[t] = Grid[0] + ∫_0^t F(Grid[s]) ds
            # Using adaptive solver for stability
            time_span = torch.tensor([0.0, target_time], device=grid_init.device)
            
            # ODE integration (CORAL Section 4.2.1)
            grid_trajectory = odeint(
                self.evolution_op,
                grid_init,
                time_span,
                method=self.ode_solver,  # 'dopri5' = adaptive Runge-Kutta
                rtol=1e-5,  # Relative tolerance
                atol=1e-7,  # Absolute tolerance
            )
            
            # Return grid at target_time (last element)
            evolved_grids.append(grid_trajectory[-1])
        
        # Cache result
        if not self.training_mode:
            self.evolution_cache[cache_key] = evolved_grids
        
        return evolved_grids
    
    def forward(self, xyz, time):
        """
        Query features from evolved grid.
        
        Unlike standard HexPlane: Grid(xyz, time),
        We do: Evolve(Grid, time) then query(xyz)
        
        Args:
            xyz: 3D coordinates [N, 3]
            time: Time value [scalar or [N, 1]]
            
        Returns:
            features: [N, feat_dim]
        """
        # Ensure time is scalar
        if isinstance(time, torch.Tensor):
            if time.numel() > 1:
                time = time[0].item()  # Assume all same time
            else:
                time = time.item()
        
        # Evolve grid to this time
        evolved_grids = self.evolve_to_time(time)
        
        # TODO: Interpolate features from evolved grids
        # For now, return placeholder
        # This needs full HexPlane interpolation logic
        
        batch_size = xyz.shape[0]
        features = torch.zeros(batch_size, self.feat_dim, device=xyz.device)
        
        return features
    
    def clear_cache(self):
        """Clear evolution cache (call when switching to training)"""
        self.evolution_cache = {}
        self.training_mode = True
    
    def set_eval_mode(self):
        """Enable caching for inference"""
        self.training_mode = False


# ============================================
# Testing/Validation Functions
# ============================================

def test_evolution_stability():
    """Test that grid evolution is stable over long time periods"""
    print("=== Testing Grid Evolution Stability ===")
    
    channels = 32
    grid_init = torch.randn(1, channels, 64, 64).cuda()
    
    # Minimal operator
    evolution_op = MinimalGridEvolution(channels).cuda()
    
    # Evolve to different times
    times = torch.linspace(0, 1.0, 11).cuda()  # 0, 0.1, 0.2, ..., 1.0
    
    print(f"Initial grid norm: {grid_init.norm().item():.4f}")
    
    for t in times[1:]:  # Skip t=0
        grid_t = odeint(evolution_op, grid_init, torch.tensor([0.0, t]).cuda())[-1]
        norm = grid_t.norm().item()
        print(f"  t={t:.1f}: grid norm = {norm:.4f}")
        
        if norm > 1000:
            print("  ⚠️ WARNING: Unstable evolution (exploding)")
            break
    
    print("✅ Evolution stability test complete")


if __name__ == '__main__':
    # Run stability test
    test_evolution_stability()

