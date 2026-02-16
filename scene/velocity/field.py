"""
EvoGS Velocity Field

Core contribution: Treats dynamic Gaussian Splatting as a true dynamical system.

Key Difference from Prior Work (4DGS):
- 4DGS: Predicts displacement Δx = f(x, t) → no temporal continuity
- EvoGS: Predicts velocity dx/dt = v(x, t) → coherent motion trajectories via ODE

Architecture:
1. HexPlane spatiotemporal feature grid (Eulerian representation)
2. Feature extraction MLP (shared across all velocity heads)
3. Velocity prediction heads (xyz, scale, rotation, opacity, SHs)
4. ODE integration to obtain final deformed state

Reference:
- 4D Gaussian Splatting (Wu et al. 2023) - displacement-based
- Neural Flow Maps (Holynski et al. 2023) - velocity-based for images
- EvoGS (ours) - velocity-based for 3D Gaussians
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid


class VelocityField(nn.Module):
    """
    Velocity field for dynamic Gaussian Splatting.
    
    Predicts velocities dx/dt for all Gaussian properties and integrates
    via ODE to obtain temporally coherent deformations.
    
    Args:
        D (int): MLP depth
        W (int): MLP width
        input_ch (int): Input feature dimension
        input_ch_time (int): Time feature dimension
        grid_pe (int): Grid positional encoding frequency
        args: Configuration arguments
    """
    
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(VelocityField, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.args = args
        
        # Integration configuration
        self.integrate_color_opacity = getattr(args, 'integrate_color_opacity', True)
        self.integrate_rotation = getattr(args, 'integrate_rotation', True)
        
        # Rotation mode: 'quaternion' (4D velocity) or 'angular' (3D angular velocity)
        # 'angular' is geometrically correct and integrates on SO(3) manifold
        self.rotation_mode = getattr(args, 'rotation_mode', 'quaternion')
        if self.rotation_mode not in ['quaternion', 'angular']:
            raise ValueError(f"rotation_mode must be 'quaternion' or 'angular', got {self.rotation_mode}")
        
        # Spatiotemporal feature grid (HexPlane factorization)
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        
        # Optional: empty voxel masking for efficiency
        if args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        
        # Optional: static region detection
        if args.static_mlp:
            self.static_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.W, self.W),
                nn.ReLU(),
                nn.Linear(self.W, 1)
            )
        
        # Two-stage training: activate velocity field after canonical scene is trained
        self.velocity_activation_iter = getattr(args, 'velocity_activation_iter', 5000)
        self.current_iteration = 0
        
        # Build network architecture
        self.ratio = 0
        self.create_net()
    
    @property
    def get_aabb(self):
        """Get axis-aligned bounding box from grid"""
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        """Set scene bounding box"""
        print(f"VelocityField: Setting AABB [{xyz_min} → {xyz_max}]")
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    def create_net(self):
        """
        Create network architecture.
        
        Structure:
        - Feature extraction network (shared)
        - Separate velocity prediction heads for each property
        """
        mlp_out_dim = 0
        
        # Grid features with optional positional encoding
        if self.grid_pe != 0:
            grid_out_dim = self.grid.feat_dim + (self.grid.feat_dim) * 2
        else:
            grid_out_dim = self.grid.feat_dim
        
        # Feature extraction MLP (shared across all velocity heads)
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W)]
        
        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        
        self.feature_out = nn.Sequential(*self.feature_out)
        
        # Velocity prediction heads
        # Each head predicts the rate of change (dx/dt) for its property
        self.velocity_xyz = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 3)
        )
        self.velocity_scale = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 3)
        )
        
        # Rotation head: 4D (quaternion velocity) or 3D (angular velocity)
        if self.rotation_mode == 'angular':
            # Predict angular velocity ω ∈ ℝ³ (geometrically correct)
            self.velocity_rotation = nn.Sequential(
                nn.ReLU(), nn.Linear(self.W, self.W),
                nn.ReLU(), nn.Linear(self.W, 3)
            )
        else:
            # Predict quaternion velocity dq/dt ∈ ℝ⁴ (legacy)
            self.velocity_rotation = nn.Sequential(
                nn.ReLU(), nn.Linear(self.W, self.W),
                nn.ReLU(), nn.Linear(self.W, 4)
            )
        
        self.velocity_opacity = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 1)
        )
        self.velocity_shs = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W),
            nn.ReLU(), nn.Linear(self.W, 16 * 3)
        )
        
        # Learnable velocity scale factors
        # Different properties have different units/magnitudes
        self.velocity_scale_xyz = nn.Parameter(torch.tensor(10.0))  # Spatial motion
        self.velocity_scale_scale = nn.Parameter(torch.tensor(1.0))  # Log-space scale
        if self.rotation_mode == 'angular':
            self.velocity_scale_rotation = nn.Parameter(torch.tensor(1.0))  # Angular velocity (rad/s)
        else:
            self.velocity_scale_rotation = nn.Parameter(torch.tensor(0.1))  # Quaternion velocity (small changes)
        self.velocity_scale_opacity = nn.Parameter(torch.tensor(1.0))  # Logit-space opacity
    
    def query_velocity(self, xyz, t):
        """
        Query velocity field at positions xyz and time t.
        
        This is the core function that defines the velocity field v(x,t).
        
        Args:
            xyz (Tensor): [N, 3] 3D positions
            t (Tensor): [N, 1] time values in [0, 1]
        
        Returns:
            hidden (Tensor): [N, W] extracted features
            velocities (dict): Dictionary with velocity for each property:
                - 'xyz': [N, 3] spatial velocity
                - 'scale': [N, 3] scale velocity
                - 'rotation': [N, 4] rotation velocity (quaternion mode) OR
                - 'angular_velocity': [N, 3] angular velocity (angular mode)
                - 'opacity': [N, 1] opacity velocity
                - 'shs': [N, 16, 3] spherical harmonics velocity
        """
        if self.no_grid:
            h = torch.cat([xyz, t], -1)
        else:
            # Query HexPlane spatiotemporal grid
            grid_feature = self.grid(xyz, t)
            
            # Optional positional encoding
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            
            h = grid_feature
        
        # Extract shared features
        hidden = self.feature_out(h)
        
        # Predict raw velocities from each head
        v_xyz_raw = self.velocity_xyz(hidden)
        v_scale_raw = self.velocity_scale(hidden)
        v_rot_raw = self.velocity_rotation(hidden)
        v_opacity_raw = self.velocity_opacity(hidden)
        v_shs_raw = self.velocity_shs(hidden)
        
        # Scale velocities by learnable factors
        velocities = {
            'xyz': v_xyz_raw * self.velocity_scale_xyz,
            'scale': v_scale_raw * self.velocity_scale_scale,
            'opacity': v_opacity_raw * self.velocity_scale_opacity,
            'shs': v_shs_raw.reshape([v_shs_raw.shape[0], 16, 3]) * self.velocity_scale_scale
        }
        
        # Rotation velocity depends on mode
        if self.rotation_mode == 'angular':
            # Angular velocity ω ∈ ℝ³ (geometrically correct)
            velocities['angular_velocity'] = v_rot_raw * self.velocity_scale_rotation
            velocities['rotation'] = None  # Not used in angular mode
        else:
            # Quaternion velocity dq/dt ∈ ℝ⁴ (legacy)
            velocities['rotation'] = v_rot_raw * self.velocity_scale_rotation
            velocities['angular_velocity'] = None  # Not used in quaternion mode
        
        return hidden, velocities
    
    def forward_static(self, rays_pts_emb):
        """
        Static scene forward pass (no dynamics).
        
        Args:
            rays_pts_emb (Tensor): [N, 3+] input positions
        
        Returns:
            Tensor: [N, 3] deformed positions (minimal deformation for static)
        """
        grid_feature = self.grid(rays_pts_emb[:, :3])
        dx = self.static_mlp(grid_feature) if self.args.static_mlp else 0
        return rays_pts_emb[:, :3] + dx
    
    @property
    def get_empty_ratio(self):
        """Get ratio of empty voxels (for pruning)"""
        return self.ratio
    
    def get_mlp_parameters(self):
        """Get MLP parameters (excluding grid)"""
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        """Get grid parameters only"""
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list


def initialize_weights(m):
    """Initialize network weights with Xavier uniform"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def poc_fre(input_data, poc_buf):
    """
    Positional encoding with frequency modulation.
    
    Args:
        input_data (Tensor): Input features
        poc_buf (Tensor): Frequency buffer
    
    Returns:
        Tensor: Positionally encoded features
    """
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb

