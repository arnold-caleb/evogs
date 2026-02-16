"""
Velocity Network Wrapper

Combines VelocityField with positional encoding and ODE integration.
This wrapper maintains compatibility with the Scene class and training pipeline.
"""

import torch
import torch.nn as nn
from scene.velocity.field import VelocityField, initialize_weights
from scene.velocity.integration import ODEIntegrator


class velocity_network(nn.Module):
    """
    Complete velocity-based deformation network.
    
    Wraps VelocityField with:
    - Positional encoding for inputs
    - ODE integration for dynamics
    - Compatibility layer for Scene initialization
    
    Args:
        args: Configuration arguments with fields:
            - net_width: MLP width
            - defor_depth: MLP depth
            - timebase_pe, posebase_pe: PE frequencies
            - ode_method: Integration method ('euler', 'rk4')
            - ode_steps: Number of integration steps
    """
    
    def __init__(self, args):
        super(velocity_network, self).__init__()
        
        # Network dimensions
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2 * timebase_pe + 1
        
        # Time encoding network (optional)
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width),
            nn.ReLU(),
            nn.Linear(timenet_width, timenet_output)
        )
        
        # Core velocity field
        self.velocity_field = VelocityField(
            W=net_width,
            D=defor_depth,
            input_ch=(3) + (3 * posbase_pe) * 2,
            grid_pe=grid_pe,
            input_ch_time=timenet_output,
            args=args
        )
        
        # ODE integrator
        ode_method = getattr(args, 'ode_method', 'euler')
        ode_steps = getattr(args, 'ode_steps', 4)
        use_adjoint = getattr(args, 'use_adjoint', False)
        
        self.integrator = ODEIntegrator(
            self.velocity_field,
            method=ode_method,
            steps=ode_steps,
            use_adjoint=use_adjoint
        )
        
        # Positional encoding buffers
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        
        # Initialize weights
        self.apply(initialize_weights)
    
    @property
    def deformation_net(self):
        """Compatibility: return velocity_field as deformation_net for Scene"""
        return self.velocity_field
    
    @property
    def get_aabb(self):
        """Get axis-aligned bounding box"""
        return self.velocity_field.get_aabb
    
    @property
    def get_empty_ratio(self):
        """Get empty voxel ratio"""
        return self.velocity_field.get_empty_ratio
    
    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, 
                times_sel=None, anchor_gaussians=None, freeze_mask=None):
        """
        Forward pass with velocity field integration.
        
        Args:
            point (Tensor): [N, 3] canonical positions
            scales (Tensor): [N, 3] canonical scales
            rotations (Tensor): [N, 4] canonical rotations (quaternions)
            opacity (Tensor): [N, 1] canonical opacity
            shs (Tensor): [N, 16, 3] canonical spherical harmonics
            times_sel (Tensor): [N, 1] target times
            anchor_gaussians (dict): Optional anchor Gaussians for multi-anchor training
            freeze_mask (Tensor): Optional mask for freezing regions
        
        Returns:
            tuple: (means3D, scales, rotations, opacity, shs) at time times_sel
        """
        if times_sel is None:
            # Static scene
            return self.forward_static(point)
        else:
            return self.forward_dynamic(
                point, scales, rotations, opacity, shs, times_sel,
                anchor_gaussians=anchor_gaussians, freeze_mask=freeze_mask
            )
    
    def forward_static(self, points):
        """Static scene (no dynamics)"""
        points = self.velocity_field.forward_static(points)
        return points
    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, 
                       shs=None, times_sel=None, t_start=0.0, 
                       anchor_gaussians=None, freeze_mask=None):
        """
        Dynamic scene with ODE integration.
        
        Process:
        1. Apply positional encoding to inputs
        2. Build initial state dictionary
        3. Integrate velocity field via ODE
        4. Return integrated state
        
        Args:
            t_start (float): Starting time for integration
            anchor_gaussians (dict): Anchor Gaussians (optional)
            freeze_mask (Tensor): Freeze mask (optional)
        
        Returns:
            tuple: Integrated (xyz, scales, rotations, opacity, shs)
        """
        N = point.shape[0]
        device = point.device
        t_target = times_sel[0, 0].item() if times_sel.ndim > 1 else times_sel[0].item()
        
        # Two-stage training: skip velocity before activation iteration
        if self.training:
            current_iter = getattr(self.velocity_field, 'current_iteration', float('inf'))
            velocity_activation = getattr(self.velocity_field, 'velocity_activation_iter', 0)
            
            if current_iter < velocity_activation:
                # Return canonical state unchanged
                return (point, scales, rotations, opacity, shs)
        
        # Apply positional encoding
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc) if scales is not None else None
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc) if rotations is not None else None
        
        # Build initial state
        initial_state = {
            'xyz': point,
            'scale': scales,
            'rotation': rotations,
        }
        
        # Add color/opacity if velocity field integrates them
        if self.velocity_field.integrate_color_opacity:
            initial_state['opacity'] = opacity
            initial_state['shs'] = shs
        
        # Integrate velocity field
        final_state = self.integrator.integrate(initial_state, t_start, t_target)
        
        # Extract results
        means3D = final_state['xyz']
        scales_out = final_state.get('scale', scales)
        rotations_out = final_state.get('rotation', rotations)
        
        # Handle color/opacity
        if self.velocity_field.integrate_color_opacity:
            opacity_out = final_state.get('opacity', opacity)
            shs_out = final_state.get('shs', shs)
        else:
            opacity_out = opacity
            shs_out = shs
        
        return means3D, scales_out, rotations_out, opacity_out, shs_out
    
    def get_mlp_parameters(self):
        """Get MLP parameters (excluding grid)"""
        return self.velocity_field.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        """Get grid parameters"""
        return self.velocity_field.get_grid_parameters()


def poc_fre(input_data, poc_buf):
    """
    Positional encoding with frequency modulation.
    
    Encodes input data using sinusoidal functions at different frequencies:
        PE(x) = [x, sin(2^0 * x), cos(2^0 * x), ..., sin(2^k * x), cos(2^k * x)]
    
    Args:
        input_data (Tensor): [..., D] input features
        poc_buf (Tensor): [K] frequency buffer with values [2^0, 2^1, ..., 2^(K-1)]
    
    Returns:
        Tensor: [..., D * (1 + 2K)] positionally encoded features
    """
    if input_data is None:
        return None
    
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb

