"""
ODE Integration for Velocity Fields

This module handles the numerical integration of velocity fields to obtain
deformed Gaussian states. It supports multiple integration methods with
memory-efficient implementations for large-scale scenes.

Integration Methods:
- Euler: Fixed-step, memory-efficient (recommended for <50GB GPU)
- RK4: 4th-order Runge-Kutta, higher accuracy (requires more memory)
- Adaptive: Variable step-size with error control

Rotation Integration:
- Quaternion mode: Integrates dq/dt in ℝ⁴, then normalizes (simple)
- Angular mode: Integrates ω via dq/dt = 0.5 * q ⊗ [0, ω] (geometrically correct)

Reference:
- torchdiffeq library for ODE solvers
- Neural ODE (Chen et al. 2018)
"""

import torch
import torch.nn as nn
from scene.velocity.quaternion_utils import quaternion_derivative, normalize_quaternion

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("WARNING: torchdiffeq not found. Install with: pip install torchdiffeq")


class ODEIntegrator:
    """
    Integrates velocity field from initial state to final state.
    
    The integrator solves the ODE:
        dx/dt = v(x, t)
    from time t0 to t1, where v(x,t) is the velocity field.
    
    Args:
        velocity_field: VelocityField instance
        method (str): Integration method ('euler', 'rk4', 'dopri5')
        steps (int): Number of integration steps (for fixed-step methods)
        use_adjoint (bool): Use adjoint method for memory efficiency
    """
    
    def __init__(self, velocity_field, method='euler', steps=4, use_adjoint=False):
        self.velocity_field = velocity_field
        self.method = method
        self.steps = steps
        self.use_adjoint = use_adjoint
        
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq required for ODE integration. pip install torchdiffeq")
    
    def integrate(self, initial_state, t0, t1):
        """
        Integrate velocity field from t0 to t1.
        
        Args:
            initial_state (dict): Dictionary with keys:
                - 'xyz': [N, 3] positions
                - 'scale': [N, 3] scales (optional)
                - 'rotation': [N, 4] quaternions (optional)
                - 'opacity': [N, 1] opacity (optional)
                - 'shs': [N, 16, 3] spherical harmonics (optional)
            t0 (float): Start time
            t1 (float): End time
        
        Returns:
            final_state (dict): Integrated state at time t1
        """
        # Edge case: no integration needed
        if abs(t1 - t0) < 1e-6:
            return {k: v.clone() if v is not None else None 
                   for k, v in initial_state.items()}
        
        N = initial_state['xyz'].shape[0]
        
        # Memory-efficient chunking for large scenes
        if N > 100000 and self.method == 'rk4':
            return self._integrate_chunked(initial_state, t0, t1)
        else:
            return self._integrate_batch(initial_state, t0, t1)
    
    def _integrate_batch(self, initial_state, t0, t1):
        """
        Integrate a batch of Gaussians (main integration logic).
        
        Args:
            initial_state (dict): Initial Gaussian state
            t0, t1 (float): Time interval
        
        Returns:
            final_state (dict): Integrated state
        """
        N = initial_state['xyz'].shape[0]
        device = initial_state['xyz'].device
        
        # Pack state into single vector for ODE solver
        state_components = []
        component_sizes = []
        component_names = []
        
        # Always integrate xyz
        state_components.append(initial_state['xyz'])
        component_sizes.append(3)
        component_names.append('xyz')
        
        # Optionally integrate other properties
        for key, size in [('scale', 3), ('rotation', 4), ('opacity', 1)]:
            if initial_state.get(key) is not None:
                state_components.append(initial_state[key])
                component_sizes.append(size)
                component_names.append(key)
        
        # Integrate SHs (only DC component for now)
        if initial_state.get('shs') is not None:
            shs_dc = initial_state['shs'][:, 0, :]  # [N, 3]
            state_components.append(shs_dc)
            component_sizes.append(3)
            component_names.append('shs')
        
        # Concatenate all components
        state_init = torch.cat([s.reshape(N, -1) for s in state_components], dim=1)
        total_dim = sum(component_sizes)
        
        # Store canonical positions for querying
        xyz_canonical = initial_state['xyz'].clone()
        
        # Check if we're using angular velocity mode
        use_angular = hasattr(self.velocity_field, 'rotation_mode') and \
                     self.velocity_field.rotation_mode == 'angular'
        
        # Define ODE function
        def ode_func(t, state_flat):
            """
            ODE right-hand side: dx/dt = v(x, t)
            
            For angular velocity mode:
                dq/dt = 0.5 * q ⊗ [0, ω]  (geometrically correct)
            
            Args:
                t (float): Current time
                state_flat (Tensor): [N * total_dim] flattened state
            
            Returns:
                Tensor: [N * total_dim] flattened velocities
            """
            state = state_flat.reshape(N, total_dim)
            t_expanded = torch.ones(N, 1, device=device) * t
            
            # Extract current xyz position
            xyz_current = state[:, :3]
            
            # Query velocity field at current position
            # This is key: v(x(t), t) not v(x_0, t) for true dynamical system
            _, velocities = self.velocity_field.query_velocity(xyz_current, t_expanded)
            
            # Pack all velocities
            v_components = []
            idx = 0
            for name, size in zip(component_names, component_sizes):
                if name == 'shs':
                    # Only DC component velocity
                    v_shs_dc = velocities['shs'][:, 0, :]
                    v_components.append(v_shs_dc.reshape(N, -1))
                elif name == 'rotation' and use_angular:
                    # Special handling for angular velocity mode
                    # Compute dq/dt = 0.5 * q ⊗ [0, ω] (quaternion derivative)
                    q_current = state[:, idx:idx+4]
                    omega = velocities['angular_velocity']
                    dq_dt = quaternion_derivative(q_current, omega)
                    v_components.append(dq_dt.reshape(N, -1))
                else:
                    # Standard velocity (xyz, scale, opacity, or quaternion velocity)
                    v_components.append(velocities[name].reshape(N, -1))
                idx += size
            
            v_all = torch.cat(v_components, dim=1)
            return v_all.reshape(-1)
        
        # Time discretization
        if self.steps is None:
            # Adaptive stepping
            t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)
        else:
            # Fixed stepping
            t_span = torch.linspace(t0, t1, self.steps + 1, device=device, dtype=torch.float32)
        
        # Solve ODE
        if self.use_adjoint:
            # Memory-efficient backpropagation (O(1) memory w.r.t. steps)
            adjoint_params = tuple(self.velocity_field.parameters())
            state_traj = odeint_adjoint(
                ode_func,
                state_init.reshape(-1),
                t_span,
                method=self.method,
                adjoint_params=adjoint_params
            )
        else:
            # Standard backpropagation (stores all intermediate states)
            state_traj = odeint(
                ode_func,
                state_init.reshape(-1),
                t_span,
                method=self.method
            )
        
        # Extract final state
        state_final = state_traj[-1].reshape(N, total_dim)
        
        # Unpack into dictionary
        final_state = {}
        idx = 0
        for name, size in zip(component_names, component_sizes):
            if name == 'shs':
                # Reconstruct full SHs (only DC changed)
                shs_dc_final = state_final[:, idx:idx+size]
                final_state['shs'] = initial_state['shs'].clone()
                final_state['shs'][:, 0, :] = shs_dc_final
            else:
                final_state[name] = state_final[:, idx:idx+size]
            idx += size
        
        # Add None for properties that weren't integrated
        for key in ['scale', 'rotation', 'opacity', 'shs']:
            if key not in final_state:
                final_state[key] = None
        
        # Normalize quaternions
        # In angular velocity mode, normalization is optional (drift is minimal)
        # In quaternion mode, normalization is required (drift off manifold)
        if final_state.get('rotation') is not None:
            if use_angular:
                # Optional normalization (for numerical stability only)
                final_state['rotation'] = normalize_quaternion(final_state['rotation'])
            else:
                # Required normalization (project back to unit sphere)
                final_state['rotation'] = torch.nn.functional.normalize(
                    final_state['rotation'], dim=-1
                )
        
        return final_state
    
    def _integrate_chunked(self, initial_state, t0, t1):
        """
        Memory-efficient integration with chunking for large scenes.
        
        For RK4 method with >100K Gaussians, processes in chunks to avoid OOM.
        
        Args:
            initial_state (dict): Initial state
            t0, t1 (float): Time interval
        
        Returns:
            final_state (dict): Integrated state
        """
        N = initial_state['xyz'].shape[0]
        device = initial_state['xyz'].device
        
        # Chunk size (conservative for RK4's 4x memory requirement)
        chunk_size = 30000
        n_chunks = (N + chunk_size - 1) // chunk_size
        
        # Allocate output tensors
        final_state = {}
        final_xyz = torch.zeros_like(initial_state['xyz'])
        
        for key in ['scale', 'rotation', 'opacity']:
            if initial_state.get(key) is not None:
                final_state[key] = torch.zeros_like(initial_state[key])
            else:
                final_state[key] = None
        
        if initial_state.get('shs') is not None:
            final_state['shs'] = initial_state['shs'].clone()
        else:
            final_state['shs'] = None
        
        # Process each chunk
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N)
            
            # Extract chunk
            chunk_state = {
                'xyz': initial_state['xyz'][start_idx:end_idx],
            }
            for key in ['scale', 'rotation', 'opacity', 'shs']:
                if initial_state.get(key) is not None:
                    chunk_state[key] = initial_state[key][start_idx:end_idx]
                else:
                    chunk_state[key] = None
            
            # Integrate chunk
            chunk_result = self._integrate_batch(chunk_state, t0, t1)
            
            # Store results
            final_xyz[start_idx:end_idx] = chunk_result['xyz']
            for key in ['scale', 'rotation', 'opacity', 'shs']:
                if final_state[key] is not None:
                    final_state[key][start_idx:end_idx] = chunk_result[key]
        
        final_state['xyz'] = final_xyz
        return final_state
    
    def integrate_bidirectional(self, initial_state, t0, t1):
        """
        Bidirectional integration for cycle-consistency regularization.
        
        Integrates forward (t0 → t1) then backward (t1 → t0) to enforce
        reversibility and temporal coherence.
        
        Args:
            initial_state (dict): Starting state at t0
            t0, t1 (float): Time interval
        
        Returns:
            tuple: (state_forward, state_backward, cycle_error)
        """
        # Forward: t0 → t1
        state_forward = self.integrate(initial_state, t0, t1)
        
        # Backward: t1 → t0
        state_backward = self.integrate(state_forward, t1, t0)
        
        # Compute cycle-consistency error
        cycle_error = (state_backward['xyz'] - initial_state['xyz']).norm(dim=-1).mean()
        
        # Optional: Add scale/rotation cycle errors
        if state_backward.get('scale') is not None:
            cycle_error += 0.1 * (state_backward['scale'] - initial_state['scale']).norm(dim=-1).mean()
        
        if state_backward.get('rotation') is not None:
            cycle_error += 0.1 * (state_backward['rotation'] - initial_state['rotation']).norm(dim=-1).mean()
        
        return state_forward, state_backward, cycle_error

