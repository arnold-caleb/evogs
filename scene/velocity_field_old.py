"""
Velocity Field for Dynamic Gaussian Splatting

Key difference from 4DGS deformation field:
- 4DGS: Predicts displacement Δx = f(x, t)  → no temporal continuity
- This:  Predicts velocity dx/dt = v(x, t)  → consistent motion trajectories

The velocity field is integrated using Neural ODE to obtain positions at any time t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from utils.graphics_utils import batch_quaternion_multiply

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("WARNING: torchdiffeq not found. Install with: pip install torchdiffeq")

try:
    from simple_knn._C import distCUDA2
    HAS_SIMPLEKNN = True
except ImportError:
    HAS_SIMPLEKNN = False
    print("WARNING: simple_knn not found. Velocity coherence loss will be disabled.")


class VelocityField(nn.Module):
    """
    Velocity-based deformation network that predicts dx/dt instead of Δx.
    
    Architecture:
    1. HexPlane grid encodes spatiotemporal features
    2. Feature extraction MLP processes grid features
    3. Velocity prediction heads output dx/dt, dscale/dt, etc.
    4. Neural ODE integrates velocities to get final states
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
        
        # CRITICAL: Query velocity at current positions for true dynamical system
        # When True: v(x₀, t) - velocity depends on initial position and time (like displacement field)
        # When False: v(x(t), t) - TRUE DYNAMICAL SYSTEM where velocity adapts to current position
        self.query_at_canonical = getattr(args, 'query_at_canonical', True)
        
        # SIMPLIFIED INTEGRATION: Use single-step Euler instead of multi-step ODE
        # When True: x(t) = x_0 + v(x_0, t) * t (like displacement field)
        # When False: x(t) = x_0 + ∫₀ᵗ v dτ (true velocity integration with odeint)
        self.use_single_step = False
        # EXPERIMENTAL: Which properties to integrate (for rendering quality)
        # True: Integrate all properties (xyz, scale, rotation, opacity, shs)
        # False: Only integrate geometry (xyz, scale, rotation) - keep color/opacity canonical
        self.integrate_color_opacity = getattr(args, 'integrate_color_opacity', True)
        
        # HYBRID MODE: Control rotation integration separately
        # True: Integrate rotation via ODE (default velocity field behavior)
        # False: Use direct displacement for rotation (displacement field style)
        # This allows mixing ODE integration (xyz, scale) with direct prediction (rotation)
        self.integrate_rotation = getattr(args, 'integrate_rotation', True)
        
        # FORWARD-ONLY INTEGRATION: Simpler anchor integration strategy
        # True: Only integrate forward from most recent anchor (no blending)
        # False: Smooth blend results from all anchors (default)
        self.forward_only_anchors = getattr(args, 'forward_only_anchors', False)
        
        # Spatiotemporal feature grid (Eulerian representation)
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        
        # Optional empty voxel masking
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        
        # Static region detector
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        
        # ODE integration settings
        # CRITICAL FIX: Use same method for train & eval to avoid train/eval mismatch
        # NOTE: Adjoint method tested but performs poorly - stick with standard odeint
        self.use_adjoint = getattr(args, 'use_adjoint', False)  # Disabled by default
        
        # ODE method selection: 'euler' for L40 44GB, 'rk4' for A100 40GB+
        self.ode_method_train = getattr(args, 'ode_method_train', 'euler')  # 'euler' or 'rk4'
        self.ode_method_eval = getattr(args, 'ode_method_eval', None)  # If None, uses train method
        
        if self.ode_method_eval is None:
            self.ode_method_eval = self.ode_method_train  # Consistent train/eval
        
        # Step counts: Conservative for memory, can increase if A100 has headroom
        if self.use_adjoint:
            # With adjoint: Can use MORE steps! Memory is O(1) w.r.t. steps
            self.ode_steps_train = getattr(args, 'ode_steps_train', 10)
            self.ode_steps_eval = getattr(args, 'ode_steps_eval', 10)
            print(f"[VELOCITY FIELD] Using ADJOINT method: {self.ode_steps_train} steps (O(1) memory)")
        elif self.ode_method_train == 'rk4':
            # RK4 on A100 40GB: Start conservative, increase if memory allows
            self.ode_steps_train = getattr(args, 'ode_steps_train', 6)   # 6 steps safe, can go to 8-10 if needed
            self.ode_steps_eval = getattr(args, 'ode_steps_eval', 6)
            print(f"[VELOCITY FIELD] Using RK4 integration: {self.ode_steps_train} steps (A100)")
        else:
            # Euler on L40: Limited by GPU memory
            self.ode_steps_train = getattr(args, 'ode_steps_train', 8)    # More steps with Euler (still memory-safe)
            self.ode_steps_eval = getattr(args, 'ode_steps_eval', 8)
            print(f"[VELOCITY FIELD] Using Euler integration: {self.ode_steps_train} steps (L40)")
        
        # TWO-STAGE TRAINING: Only activate velocity field after canonical is trained
        self.velocity_activation_iter = getattr(args, 'velocity_activation_iter', 5000)
        self.current_iteration = 0  # Will be updated by training loop
        
        self.ratio = 0
        self.create_net()
    
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print("Velocity Field Set aabb", xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    def create_net(self):
        """
        Create network architecture.
        
        Key change from 4DGS: Output heads are called "velocity_*" instead of "*_deform"
        to emphasize that we're predicting rates of change, not displacements.
        """
        mlp_out_dim = 0
        
        if self.grid_pe != 0:
            grid_out_dim = self.grid.feat_dim + (self.grid.feat_dim) * 2
        else:
            grid_out_dim = self.grid.feat_dim
        
        # Feature extraction network (shared across all velocity heads)
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W)]
        
        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        
        self.feature_out = nn.Sequential(*self.feature_out)
        
        # Velocity prediction heads (renamed from *_deform to velocity_*)
        # These predict dx/dt, dscale/dt, drot/dt, etc.
        self.velocity_xyz = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.velocity_scale = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.velocity_rotation = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4))
        self.velocity_opacity = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        self.velocity_shs = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 16 * 3))
        
        # Scale factors for velocity predictions (helps learn faster)
        # Different properties need different scales because they have different units/magnitudes
        # xyz: spatial units (meters) - needs larger scale
        # scale: log-space - needs moderate scale  
        # rotation: quaternion (should be small changes) - needs smaller scale
        # opacity: logit-space - needs moderate scale
        self.velocity_scale_xyz = nn.Parameter(torch.tensor(10.0))  # Larger for spatial motion
        self.velocity_scale_scale = nn.Parameter(torch.tensor(1.0))
        self.velocity_scale_rotation = nn.Parameter(torch.tensor(0.1))  # Smaller for rotation
        self.velocity_scale_opacity = nn.Parameter(torch.tensor(1.0))
    
    def query_velocity(self, xyz, t):
        """
        Query the velocity field at positions xyz and time t.
        
        Args:
            xyz: [N, 3] positions
            t: [N, 1] time values
        
        Returns:
            velocities: dict with keys 'xyz', 'scale', 'rotation', 'opacity', 'shs'
        """
        if self.no_grid:
            h = torch.cat([xyz, t], -1)
        else:
            # Query HexPlane grid at (x, t)
            grid_feature = self.grid(xyz, t)
            
            # Optional positional encoding
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            
            h = grid_feature
        
        # Extract features
        hidden = self.feature_out(h)
        
        # Predict raw velocities
        v_xyz_raw = self.velocity_xyz(hidden)
        v_scale_raw = self.velocity_scale(hidden)
        v_rot_raw = self.velocity_rotation(hidden)
        v_opacity_raw = self.velocity_opacity(hidden)
        v_shs_raw = self.velocity_shs(hidden)
        
        # Scale velocities by learnable scale factors (helps learn faster)
        # Each property has its own scale factor since they have different units/magnitudes
        # NO MASKING - let the velocity field learn motion everywhere
        velocities = {
            'xyz': v_xyz_raw * self.velocity_scale_xyz,  # Scaled spatial velocity
            'scale': v_scale_raw * self.velocity_scale_scale,  # Scaled log-space velocity
            'rotation': v_rot_raw * self.velocity_scale_rotation,  # Scaled quaternion velocity
            'opacity': v_opacity_raw * self.velocity_scale_opacity,  # Scaled logit-space velocity
            'shs': v_shs_raw.reshape([v_shs_raw.shape[0], 16, 3]) * self.velocity_scale_scale  # Use scale_scale for SH
        }
        
        return hidden, velocities
    
    def _integrate_single_step(self, initial_state, t0, t1):
        """
        Single-step Euler integration: x(t) = x_0 + v(x_0, t) * dt
        
        This treats velocity as dx/dt and directly computes displacement.
        Much simpler than multi-step ODE and behaves like displacement field.
        """
        N = initial_state['xyz'].shape[0]
        device = initial_state['xyz'].device
        dt = t1 - t0
        
        # Query velocity at canonical position and target time
        xyz_canonical = initial_state['xyz']
        t_tensor = torch.ones(N, 1, device=device) * t1
        
        _, velocities = self.query_velocity(xyz_canonical, t_tensor)
        
        # Single-step update: x(t) = x_0 + v * dt
        final_state = {
            'xyz': initial_state['xyz'] + velocities['xyz'] * dt,
            'scale': initial_state.get('scale'),
            'rotation': initial_state.get('rotation'),
            'opacity': initial_state.get('opacity'),
            'shs': initial_state.get('shs'),
        }
        
        # Apply velocity to other properties if present
        if initial_state.get('scale') is not None:
            final_state['scale'] = initial_state['scale'] + velocities['scale'] * dt
        if initial_state.get('rotation') is not None:
            final_state['rotation'] = initial_state['rotation'] + velocities['rotation'] * dt
        if initial_state.get('opacity') is not None:
            final_state['opacity'] = initial_state['opacity'] + velocities['opacity'] * dt
        if initial_state.get('shs') is not None:
            # Only integrate DC component (first SH coefficient)
            shs_delta = velocities['shs'][:, 0:1, :] * dt
            final_state['shs'] = initial_state['shs'].clone()
            final_state['shs'][:, 0:1, :] = initial_state['shs'][:, 0:1, :] + shs_delta
        
        return final_state
    
    def integrate_velocity(self, initial_state, t0, t1, method='rk4', steps=None):
        """
        Integrate velocity field from t0 to t1.
        
        Two modes:
        1. Single-step: x(t) = x_0 + v(x_0, t) * t (simple, like displacement field)
        2. Multi-step ODE: x(t) = x_0 + ∫₀ᵗ v dτ (complex, true dynamics)
        
        Args:
            initial_state: dict with 'xyz', 'scale', 'rotation', 'opacity', 'shs'
            t0: start time (scalar)
            t1: end time (scalar)
            method: ODE solver ('rk4' for training, 'dopri5' for eval)
            steps: number of integration steps (None = adaptive)
        
        Returns:
            final_state: dict with integrated values
        """
        N = initial_state['xyz'].shape[0]
        device = initial_state['xyz'].device
        
        # Edge case: if t0 == t1, no integration needed
        if abs(t1 - t0) < 1e-6:
            return initial_state.copy()
        
        # SIMPLIFIED MODE: Single-step Euler (like displacement field)
        if self.use_single_step:
            return self._integrate_single_step(initial_state, t0, t1)
        
        # FULL ODE MODE: Multi-step integration
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq required for velocity field integration")
        
        # MEMORY-EFFICIENT CHUNKING for RK4 on L40 (44GB)
        # RK4 needs 4x memory of Euler → chunk for memory safety
        if method == 'rk4' and N > 100000:  # Chunk if >100K Gaussians with RK4
            return self._integrate_velocity_chunked_rk4(initial_state, t0, t1, method, steps)
        else:
            # NO CHUNKING: Euler can process all at once
            return self._integrate_velocity_chunk(initial_state, t0, t1, method, steps)
    
    def _integrate_velocity_chunk(self, initial_state, t0, t1, method='rk4', steps=None):
        """
        Integrate a chunk of Gaussians (internal method).
        Called by integrate_velocity with chunking for memory efficiency.
        """
        N = initial_state['xyz'].shape[0]
        device = initial_state['xyz'].device
        
        # Problem 2 fix: Integrate ALL properties (xyz, scale, rotation, opacity, shs)
        # Pack all states into a single vector for ODE integration
        state_components = []
        component_sizes = []
        
        # Always integrate xyz
        state_components.append(initial_state['xyz'])  # [N, 3]
        component_sizes.append(3)
        
        # Integrate scale if provided
        if initial_state.get('scale') is not None:
            state_components.append(initial_state['scale'])  # [N, 3]
            component_sizes.append(3)
        
        # Integrate rotation if provided
        if initial_state.get('rotation') is not None:
            state_components.append(initial_state['rotation'])  # [N, 4]
            component_sizes.append(4)
        
        # Integrate opacity if provided
        if initial_state.get('opacity') is not None:
            state_components.append(initial_state['opacity'])  # [N, 1]
            component_sizes.append(1)
        
        # Integrate SHs if provided (only DC component for now)
        if initial_state.get('shs') is not None:
            # SHs shape: [N, 16, 3] - only integrate DC (first 3)
            shs_dc = initial_state['shs'][:, 0, :]  # [N, 3]
            state_components.append(shs_dc)
            component_sizes.append(3)
        
        # Concatenate all components into single state vector
        state_init = torch.cat([s.reshape(N, -1) for s in state_components], dim=1)  # [N, total_dim]
        total_dim = sum(component_sizes)
        
        # Store canonical positions for experimental canonical query mode
        xyz_canonical = initial_state['xyz'].clone()
        
        # Define ODE function for ALL properties
        def ode_func(t, state_flat):
            # state_flat: [N * total_dim]
            state = state_flat.reshape(N, total_dim)
            t_expanded = torch.ones(N, 1, device=device) * t
            
            # Unpack state
            idx = 0
            xyz_current = state[:, idx:idx+3]
            idx += 3
            
            # ANNEALING SCHEDULE: Smoothly transition canonical → current queries
            # Early training: query_at_canonical (stable, displacement-like)
            # Late training: query_at_current (true ODE, coherent dynamics)
            if self.query_at_canonical:
                # Pure canonical (stable but not true ODE)
                xyz_query = xyz_canonical
            else:
                # ANNEALING: Gradual transition for stability
                # α = 0 (early): query at canonical (stable)
                # α = 1 (late): query at current (true dynamics)
                anneal_start = 1000  # Start transitioning after warmup
                anneal_end = 3000    # Fully dynamic by iteration 3000
                
                if self.training and self.current_iteration < anneal_start:
                    # Early training: Pure canonical (stable)
                    alpha = 0.0
                elif self.training and self.current_iteration < anneal_end:
                    # Mid training: Smooth transition
                    alpha = (self.current_iteration - anneal_start) / (anneal_end - anneal_start)
                else:
                    # Late training / evaluation: Pure current (true ODE)
                    alpha = 1.0
                
                # Interpolate: xyz_query = (1-α)*x₀ + α*x(t)
                xyz_query = (1 - alpha) * xyz_canonical + alpha * xyz_current
                
                # Debug: Print transition progress
                if self.training and self.current_iteration % 500 == 0:
                    print(f"[ANNEALING] Iter {self.current_iteration}: α={alpha:.3f} ({'canonical' if alpha < 0.5 else 'current'} dominant)")
            
            # Query velocity field
            _, velocities = self.query_velocity(xyz_query, t_expanded)
            
            # Pack all velocities
            v_components = [velocities['xyz']]  # Always have xyz velocity
            
            if initial_state.get('scale') is not None:
                v_components.append(velocities['scale'])
            
            if initial_state.get('rotation') is not None:
                v_components.append(velocities['rotation'])
            
            if initial_state.get('opacity') is not None:
                v_components.append(velocities['opacity'])
            
            if initial_state.get('shs') is not None:
                v_shs_dc = velocities['shs'][:, 0, :]  # Only DC component
                v_components.append(v_shs_dc)
            
            # Concatenate all velocities
            v_all = torch.cat([v.reshape(N, -1) for v in v_components], dim=1)
            
            return v_all.reshape(-1)
        
        # Integration time points
        if steps is None:
            # Adaptive stepping
            t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)
        else:
            # Fixed stepping
            t_span = torch.linspace(t0, t1, steps + 1, device=device, dtype=torch.float32)
        
        # Integrate using adjoint method if enabled (O(1) memory w.r.t. steps)
        if self.use_adjoint:
            # odeint_adjoint requires explicit parameter tracking
            # Collect all trainable parameters from the velocity field
            adjoint_params = tuple(self.parameters())
            state_traj = odeint_adjoint(
                ode_func,
                state_init.reshape(-1),
                t_span,
                method=method,
                adjoint_params=adjoint_params
            )
        else:
            # Standard odeint (stores all intermediate states)
            state_traj = odeint(
                ode_func,
                state_init.reshape(-1),
                t_span,
                method=method
            )
        
        # Extract final state and unpack
        state_final = state_traj[-1].reshape(N, total_dim)
        
        final_state = {}
        idx = 0
        
        # Unpack xyz
        final_state['xyz'] = state_final[:, idx:idx+3]
        idx += 3
        
        # Unpack scale
        if initial_state.get('scale') is not None:
            final_state['scale'] = state_final[:, idx:idx+3]
            idx += 3
        else:
            final_state['scale'] = None
        
        # Unpack rotation
        if initial_state.get('rotation') is not None:
            final_state['rotation'] = state_final[:, idx:idx+4]
            idx += 4
        else:
            final_state['rotation'] = None
        
        # Unpack opacity
        if initial_state.get('opacity') is not None:
            final_state['opacity'] = state_final[:, idx:idx+1]
            idx += 1
        else:
            final_state['opacity'] = None
        
        # Unpack SHs
        if initial_state.get('shs') is not None:
            shs_dc_final = state_final[:, idx:idx+3]
            # Reconstruct full SHs (keep higher-order terms unchanged for now)
            final_state['shs'] = initial_state['shs'].clone()
            final_state['shs'][:, 0, :] = shs_dc_final
        else:
            final_state['shs'] = None
        
        return final_state
    
    def _integrate_velocity_chunked_rk4(self, initial_state, t0, t1, method='rk4', steps=None):
        """
        Memory-efficient RK4 integration with chunking for L40 GPUs (44GB).
        
        RK4 computes 4 intermediate states per step → 4x memory usage.
        Solution: Process Gaussians in chunks to fit in GPU memory.
        
        Memory estimate:
        - 220K Gaussians × 20 dims × 4 bytes × 4 (RK4) × 6 steps ≈ 42GB ✅ Fits!
        - But add safety margin → chunk into 2-4 batches
        
        Args:
            initial_state: Full state dict
            t0, t1: Time range
            method: 'rk4'
            steps: Number of ODE steps
        """
        N = initial_state['xyz'].shape[0]
        device = initial_state['xyz'].device
        
        # Determine chunk size based on GPU memory
        # L40 44GB: RK4 needs 4x memory → VERY conservative chunking
        # With 6 steps, each chunk needs ~15GB → max 30K Gaussians per chunk
        chunk_size = 30000  # Ultra-conservative for RK4's 4x memory requirement
        
        if N <= chunk_size:
            # Small enough to process in one go
            return self._integrate_velocity_chunk(initial_state, t0, t1, method, steps)
        
        # Split into chunks
        n_chunks = (N + chunk_size - 1) // chunk_size
        print(f"[RK4 CHUNKING] Processing {N} Gaussians in {n_chunks} chunks of ~{chunk_size}")
        
        # Allocate output tensors
        final_state = {}
        final_xyz = torch.zeros_like(initial_state['xyz'])
        final_scale = torch.zeros_like(initial_state['scale']) if initial_state.get('scale') is not None else None
        final_rotation = torch.zeros_like(initial_state['rotation']) if initial_state.get('rotation') is not None else None
        final_opacity = torch.zeros_like(initial_state['opacity']) if initial_state.get('opacity') is not None else None
        final_shs = initial_state['shs'].clone() if initial_state.get('shs') is not None else None
        
        # Process each chunk
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N)
            
            # Extract chunk
            chunk_state = {
                'xyz': initial_state['xyz'][start_idx:end_idx],
                'scale': initial_state['scale'][start_idx:end_idx] if initial_state.get('scale') is not None else None,
                'rotation': initial_state['rotation'][start_idx:end_idx] if initial_state.get('rotation') is not None else None,
                'opacity': initial_state['opacity'][start_idx:end_idx] if initial_state.get('opacity') is not None else None,
                'shs': initial_state['shs'][start_idx:end_idx] if initial_state.get('shs') is not None else None,
            }
            
            # Integrate chunk
            chunk_result = self._integrate_velocity_chunk(chunk_state, t0, t1, method, steps)
            
            # Store results
            final_xyz[start_idx:end_idx] = chunk_result['xyz']
            if final_scale is not None:
                final_scale[start_idx:end_idx] = chunk_result['scale']
            if final_rotation is not None:
                final_rotation[start_idx:end_idx] = chunk_result['rotation']
            if final_opacity is not None:
                final_opacity[start_idx:end_idx] = chunk_result['opacity']
            if final_shs is not None:
                final_shs[start_idx:end_idx] = chunk_result['shs']
        
        # Assemble final state
        final_state = {
            'xyz': final_xyz,
            'scale': final_scale,
            'rotation': final_rotation,
            'opacity': final_opacity,
            'shs': final_shs,
        }
        
        return final_state
    
    def integrate_bidirectional(self, initial_state, t0, t1, method='euler', steps=4):
        """
        Bidirectional integration for cycle-consistency (Neural Flow Maps approach).
        
        Integrates forward (t0 → t1) then backward (t1 → t0) to enforce reversibility.
        
        Args:
            initial_state: Starting state at t0
            t0: Start time
            t1: End time
            method: Integration method
            steps: Number of ODE steps
        
        Returns:
            state_forward: Integrated state at t1
            state_backward: Integrated back to t0 (should ≈ initial_state)
            cycle_loss: |state_backward - initial_state|² (consistency error)
        """
        # Forward integration: t0 → t1
        state_forward = self.integrate_velocity(initial_state, t0, t1, method, steps)
        
        # Backward integration: t1 → t0
        state_backward = self.integrate_velocity(state_forward, t1, t0, method, steps)
        
        # Compute cycle-consistency error
        cycle_error_xyz = (state_backward['xyz'] - initial_state['xyz']).norm(dim=-1).mean()
        
        # Optional: Also check scale/rotation cycle consistency
        cycle_error_total = cycle_error_xyz
        if state_backward.get('scale') is not None and initial_state.get('scale') is not None:
            cycle_error_scale = (state_backward['scale'] - initial_state['scale']).norm(dim=-1).mean()
            cycle_error_total += 0.1 * cycle_error_scale
        
        if state_backward.get('rotation') is not None and initial_state.get('rotation') is not None:
            cycle_error_rot = (state_backward['rotation'] - initial_state['rotation']).norm(dim=-1).mean()
            cycle_error_total += 0.1 * cycle_error_rot
        
        return state_forward, state_backward, cycle_error_total
    
    def compute_divergence_loss(self, xyz, t, n_samples=500):
        """
        Compute divergence of velocity field: ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z
        
        For incompressible flow (volume-preserving): ∇·v ≈ 0
        Prevents Gaussians from expanding (gaps) or collapsing (thinning).
        
        Uses finite differences with neighbors (avoids second-order derivatives through grid_sample).
        
        Args:
            xyz: [N, 3] positions
            t: scalar or [N, 1] time
            n_samples: Number of points to sample (for efficiency)
        
        Returns:
            div_loss: Mean squared divergence
        """
        # Sample for efficiency
        if xyz.shape[0] > n_samples:
            sample_idx = torch.randperm(xyz.shape[0], device=xyz.device)[:n_samples]
            xyz_sample = xyz[sample_idx]
        else:
            xyz_sample = xyz
        
        N = xyz_sample.shape[0]
        device = xyz_sample.device
        
        # Time tensor
        if isinstance(t, float):
            t_tensor = torch.ones(N, 1, device=device) * t
        else:
            t_tensor = t
        
        # Query velocity at sample points
        with torch.no_grad():
            _, velocities_center = self.query_velocity(xyz_sample, t_tensor)
            v_center = velocities_center['xyz']  # [N, 3]
        
        # Finite difference approximation using small perturbations
        epsilon = 0.01  # Small step for finite differences
        div = torch.zeros(N, device=device)
        
        for i in range(3):  # x, y, z dimensions
            # Perturb position in dimension i
            xyz_plus = xyz_sample.clone()
            xyz_plus[:, i] += epsilon
            
            xyz_minus = xyz_sample.clone()
            xyz_minus[:, i] -= epsilon
            
            # Query velocities at perturbed positions
            with torch.no_grad():
                _, v_plus = self.query_velocity(xyz_plus, t_tensor)
                _, v_minus = self.query_velocity(xyz_minus, t_tensor)
            
            # Finite difference: ∂v_i/∂x_i ≈ (v_i(x+ε) - v_i(x-ε)) / (2ε)
            dv_dx = (v_plus['xyz'][:, i] - v_minus['xyz'][:, i]) / (2 * epsilon)
            div += dv_dx
        
        # Penalize non-zero divergence (incompressibility)
        div_loss = (div ** 2).mean()
        
        return div_loss
    
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, 
                opacity=None, shs_emb=None, time_feature=None, time_emb=None, t_start=0.0, anchor_gaussians=None):
        """
        Forward pass with velocity field integration.
        
        Args:
            rays_pts_emb: [N, 3+] Gaussian positions (canonical space)
            scales_emb: [N, 3] scales
            rotations_emb: [N, 4] rotations
            opacity: [N, 1] opacities
            shs_emb: [N, 16, 3] spherical harmonics
            time_emb: [N, 1] target time
            t_start: Starting time for integration (default 0.0)
            anchor_gaussians: Dict of anchor Gaussians for multi-anchor training
        
        Returns:
            pts, scales, rotations, opacity, shs at time t
        """
        if time_emb is None:
            # Static scene - no integration needed
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(
                rays_pts_emb, scales_emb, rotations_emb, 
                opacity, shs_emb, time_feature, time_emb, t_start=t_start, anchor_gaussians=anchor_gaussians
            )
    
    def forward_static(self, rays_pts_emb):
        """Static scene (no time evolution)"""
        grid_feature = self.grid(rays_pts_emb[:, :3])
        dx = self.static_mlp(grid_feature) if self.args.static_mlp else 0
        return rays_pts_emb[:, :3] + dx
    
    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, 
                       opacity_emb, shs_emb, time_feature, time_emb, t_start=0.0, anchor_gaussians=None):
        """
        Dynamic scene with velocity field integration.
        
        TWO-STAGE TRAINING:
        - Stage 1 (iter < velocity_activation_iter): Return canonical (train static scene)
        - Stage 2 (iter >= velocity_activation_iter): Apply velocity field (train motion)
        
        MULTI-ANCHOR MODE:
        - If anchor_gaussians provided, integrates from nearest anchor instead of canonical
        - Reduces integration distance and prevents drift
        """
        xyz_canonical = rays_pts_emb[:, :3]
        N = xyz_canonical.shape[0]
        device = xyz_canonical.device
        t_target = time_emb[0, 0].item()
        
        # Opacity/SHs always from embeddings (not integrated)
        opacity_start = opacity_emb[:, :1] if opacity_emb is not None else None
        shs_start = shs_emb if shs_emb is not None else None
        
        # STAGE 1: Before activation iteration, return canonical unchanged
        # This allows training a sharp static scene before adding motion
        # CRITICAL: During evaluation/rendering, always integrate (self.training=False)
        # Only skip during training if we haven't reached activation iteration
        if self.training:
            current_iter = getattr(self, 'current_iteration', float('inf'))
            if current_iter < self.velocity_activation_iter:
                # DEBUG: This should NEVER print if velocity_activation_iter=0
                print(f"[VELOCITY FIELD] Skipping deformation: current_iter={current_iter} < activation_iter={self.velocity_activation_iter}")
                return (xyz_canonical, 
                        scales_emb[:, :3] if scales_emb is not None else None,
                        rotations_emb[:, :4] if rotations_emb is not None else None,
                        opacity_emb[:, :1] if opacity_emb is not None else None,
                        shs_emb if shs_emb is not None else None)
        
        # FIXED INTEGRATION STEPS: Use 4 steps everywhere to avoid OOM
        # With 220K Gaussians, adaptive steps cause memory issues
        if self.training:
            method = self.ode_method_train  # 'euler'
            steps = 4  # Fixed 4 steps for memory safety
        else:
            method = self.ode_method_eval  # 'euler'
            steps = 4  # Same for eval (consistent!)
        
        # ========================================================================
        # MULTI-ANCHOR INTEGRATION
        # ========================================================================
        if anchor_gaussians is not None and len(anchor_gaussians) > 0:
            anchor_times = sorted(anchor_gaussians.keys())
            n_canonical = N

            anchor_t = 0.0
            for t in anchor_times:
                if t <= t_target:
                    anchor_t = t

            anchor_gauss = anchor_gaussians[anchor_t]
            n_anchor = anchor_gauss._xyz.shape[0]
            n_integrate = min(n_anchor, n_canonical)

            xyz_anchor = anchor_gauss._xyz[:n_integrate].to(device)
            scales_anchor = anchor_gauss._scaling[:n_integrate].to(device) if anchor_gauss._scaling is not None else None
            rotations_anchor = anchor_gauss._rotation[:n_integrate].to(device) if anchor_gauss._rotation is not None else None

            if n_canonical > n_anchor:
                xyz_anchor = torch.cat([xyz_anchor, xyz_canonical[n_anchor:]], dim=0)
                if scales_emb is not None:
                    if scales_anchor is not None:
                        scales_anchor = torch.cat([scales_anchor, scales_emb[n_anchor:, :3]], dim=0)
                    else:
                        scales_anchor = scales_emb[:, :3]
                if rotations_emb is not None:
                    if rotations_anchor is not None:
                        rotations_anchor = torch.cat([rotations_anchor, rotations_emb[n_anchor:, :4]], dim=0)
                    else:
                        rotations_anchor = rotations_emb[:, :4]

            initial_state = {
                'xyz': xyz_anchor,
                'scale': scales_anchor,
            }
            if self.integrate_rotation:
                initial_state['rotation'] = rotations_anchor
            if self.integrate_color_opacity:
                initial_state['opacity'] = opacity_start
                initial_state['shs'] = shs_start

            final_state = self.integrate_velocity(initial_state, anchor_t, t_target, method, steps)

            if self.training and self.current_iteration % 500 == 0:
                dt = t_target - anchor_t
                print(f"[FORWARD-ONLY] t={t_target:.3f}, anchor={anchor_t:.3f}, dt={dt:.3f}")
        else:
            # Standard: Single integration from canonical
            xyz_start = xyz_canonical
            scales_start = scales_emb[:, :3] if scales_emb is not None else None
            rotations_start = rotations_emb[:, :4] if rotations_emb is not None else None
            t_start = 0.0
            
            # Build initial state
            initial_state = {
                'xyz': xyz_start,
                'scale': scales_start,
            }
            if self.integrate_rotation:
                initial_state['rotation'] = rotations_start
            if self.integrate_color_opacity:
                initial_state['opacity'] = opacity_start
                initial_state['shs'] = shs_start
            
            # Single ODE integration
            final_state = self.integrate_velocity(initial_state, t_start, t_target, method, steps)
        
        # CACHE integrated positions for regularization (avoid double integration)
        # This is used by compute_trajectory_coherence to reuse already-computed positions
        if hasattr(self, 'last_integrated_xyz'):
            self.last_integrated_xyz = final_state['xyz'].detach()
        else:
            # First time: create as a regular attribute (not a parameter)
            self.register_buffer('last_integrated_xyz', final_state['xyz'].detach(), persistent=False)
        
        # DEBUG: Print velocity field activity during training
        if self.training and self.current_iteration % 500 == 0:
            displacement_mag = (final_state['xyz'] - xyz_canonical).norm(dim=-1).mean().item()
            print(f"[VELOCITY FIELD] Iter {self.current_iteration}, t={t_target:.3f}, displacement_mean={displacement_mag:.6f}")
        
        # CRITICAL: Normalize quaternions to maintain unit length constraint
        # During ODE integration, quaternions can drift from ||q|| = 1
        # This ensures valid rotations after integration
        if final_state.get('rotation') is not None:
            final_state['rotation'] = torch.nn.functional.normalize(final_state['rotation'], dim=-1)
        
        # Return integrated state
        pts = xyz_canonical if self.args.no_dx else final_state['xyz']
        scales = final_state['scale'] if final_state['scale'] is not None else scales_emb[:, :3]
        
        # HYBRID MODE: Handle rotation based on integration flag
        if self.integrate_rotation:
            # Integrated rotation (from ODE)
            rotations = final_state['rotation'] if final_state.get('rotation') is not None else rotations_emb[:, :4]
        else:
            # Direct displacement for rotation (no ODE integration)
            # Query velocity at target time and apply as displacement
            t_tensor = torch.ones(N, 1, device=device) * t_target
            _, velocities = self.query_velocity(xyz_canonical, t_tensor)
            dt = t_target - t_start
            rotation_displacement = velocities['rotation'] * dt
            
            # Apply rotation: quaternion multiplication vs simple addition
            if self.args.apply_rotation:
                # Quaternion multiplication: q_final = q_canonical ⊗ Δq
                rotations = batch_quaternion_multiply(rotations_emb[:, :4], rotation_displacement)
            else:
                # Simple addition: q_final = q_canonical + Δq
                rotations = rotations_emb[:, :4] + rotation_displacement
                # Normalize to ensure valid quaternion
                rotations = torch.nn.functional.normalize(rotations, dim=-1)
        
        # Choose opacity/SHs based on integration flag
        if self.integrate_color_opacity:
            # Use integrated values
            opacity = final_state['opacity'] if final_state['opacity'] is not None else opacity_emb[:, :1]
            shs = final_state['shs'] if final_state['shs'] is not None else shs_emb
        else:
            # Use canonical values (not integrated, so always use original)
            opacity = opacity_emb[:, :1] if opacity_emb is not None else None
            shs = shs_emb if shs_emb is not None else None
        
        return pts, scales, rotations, opacity, shs
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def get_mlp_parameters(self):
        """Get MLP parameters (excluding grid)"""
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        """Get grid parameters"""
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list


class velocity_network(nn.Module):
    """
    Wrapper around VelocityField that handles positional encoding.
    Similar to deform_network but for velocity-based dynamics.
    """
    def __init__(self, args):
        super(velocity_network, self).__init__()
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
        
        # Time encoding network (optional, can be disabled)
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width),
            nn.ReLU(),
            nn.Linear(timenet_width, timenet_output)
        )
        
        # Velocity field network (replaces Deformation)
        self.velocity_field = VelocityField(
            W=net_width,
            D=defor_depth,
            input_ch=(3) + (3 * (posbase_pe)) * 2,
            grid_pe=grid_pe,
            input_ch_time=timenet_output,
            args=args
        )
        
        # Positional encoding buffers
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        
        # Initialize weights
        self.apply(initialize_weights)
    
    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, anchor_gaussians=None, freeze_mask=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel, anchor_gaussians=anchor_gaussians, freeze_mask=freeze_mask)
    
    @property
    def deformation_net(self):
        """Compatibility property: return velocity_field as deformation_net for Scene initialization"""
        return self.velocity_field
    
    @property
    def get_aabb(self):
        return self.velocity_field.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.velocity_field.get_empty_ratio
    
    def forward_static(self, points):
        points = self.velocity_field(points)
        return points
    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None, t_start=0.0, anchor_gaussians=None, freeze_mask=None):
        """
        Forward pass with positional encoding.
        Same interface as deform_network for compatibility.
        
        Args:
            t_start: Starting time for integration (for short rollout training)
            anchor_gaussians: Dict of anchor Gaussians for multi-anchor training
        """

        # Apply positional encoding to inputs
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        
        # Velocity field integration with multi-anchor support
        means3D, scales, rotations, opacity, shs = self.velocity_field(
            point_emb,
            scales_emb,
            rotations_emb,
            opacity,
            shs,
            None,
            times_sel,
            t_start=t_start,  # Pass through t_start
            anchor_gaussians=anchor_gaussians  # Pass anchor Gaussians
        )

        return means3D, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        """Get MLP parameters (excluding grid)"""
        return self.velocity_field.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        """Get grid parameters"""
        return self.velocity_field.get_grid_parameters()


def initialize_weights(m):
    """Initialize network weights with Xavier uniform"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            nn.init.xavier_uniform_(m.weight, gain=1)


def poc_fre(input_data, poc_buf):
    """Positional encoding frequency modulation"""
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb