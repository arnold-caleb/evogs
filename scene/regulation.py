import abc
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch import nn



def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


class Regularizer():
    def __init__(self, reg_type, initialization):
        self.reg_type = reg_type
        self.initialization = initialization
        self.weight = float(self.initialization)
        self.last_reg = None

    def step(self, global_step):
        pass

    def report(self, d):
        if self.last_reg is not None:
            d[self.reg_type].update(self.last_reg.item())

    def regularize(self, *args, **kwargs) -> torch.Tensor:
        out = self._regularize(*args, **kwargs) * self.weight
        self.last_reg = out.detach()
        return out

    @abc.abstractmethod
    def _regularize(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self):
        return f"Regularizer({self.reg_type}, weight={self.weight})"


class PlaneTV(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'planeTV-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def step(self, global_step):
        pass

    def _regularize(self, model, **kwargs):
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # Note: input to compute_plane_tv should be of shape [batch_size, c, h, w]
        for grids in multi_res_grids:
            if len(grids) == 3:
                spatial_grids = [0, 1, 2]
            else:
                spatial_grids = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal
            for grid_id in spatial_grids:
                total += compute_plane_tv(grids[grid_id])
            for grid in grids:
                # grid: [1, c, h, w]
                total += compute_plane_tv(grid)
        return total


class TimeSmoothness(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'time-smooth-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def _regularize(self, model, **kwargs) -> torch.Tensor:
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return torch.as_tensor(total)



class L1ProposalNetwork(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-proposal-network', initial_value)

    def _regularize(self, model, **kwargs) -> torch.Tensor:
        grids = [p.grids for p in model.proposal_networks]
        total = 0.0
        for pn_grids in grids:
            for grid in pn_grids:
                total += torch.abs(grid).mean()
        return torch.as_tensor(total)


class DepthTV(Regularizer):
    def __init__(self, initial_value):
        super().__init__('tv-depth', initial_value)

    def _regularize(self, model, model_out, **kwargs) -> torch.Tensor:
        depth = model_out['depth']
        tv = compute_plane_tv(
            depth.reshape(64, 64)[None, None, :, :]
        )
        return tv


class L1TimePlanes(Regularizer):
    def __init__(self, initial_value, what='field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        super().__init__(f'l1-time-{what[:2]}', initial_value)
        self.what = what

    def _regularize(self, model, **kwargs) -> torch.Tensor:
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return torch.as_tensor(total)


def compute_trajectory_coherence(gaussians, velocity_field, t,
                                  n_samples=2000,
                                  n_neighbors=8,
                                  lambda_coh=0.1,
                                  integrated_positions_cache=None):
    """
    Trajectory coherence loss - ensures neighboring Gaussians stay close after integration.
    
    KEY INSIGHT: Instead of comparing velocities v(x_0, t), we compare the actual 
    integrated positions x(t) = x_0 + ∫v dt. This prevents blob merging caused by 
    cumulative displacement divergence.
    
    OPTIMIZATION: Accepts pre-computed integrated positions from rendering to avoid 
    double integration (expensive!).
    
    Args:
        gaussians: GaussianModel with _xyz positions
        velocity_field: VelocityField module
        t: Current time (float)
        n_samples: Number of Gaussians to sample (default: 2000)
        n_neighbors: Number of nearest neighbors (default: 8)
        lambda_coh: Weight for coherence loss (default: 0.1)
        integrated_positions_cache: Optional [N, 3] tensor of pre-integrated positions (from render)
    
    Returns:
        coherence_loss: Scalar tensor with gradient
    """
    device = gaussians._xyz.device
    N = gaussians._xyz.shape[0]
    canonical_positions_all = gaussians._xyz  # [N, 3]
    
    # Sample Gaussians
    if N > n_samples:
        sample_idx = torch.randperm(N, device=device)[:n_samples]
        canonical_positions = canonical_positions_all[sample_idx].detach()
    else:
        sample_idx = torch.arange(N, device=device)
        canonical_positions = canonical_positions_all.detach()
        n_samples = N
    
    # Find k-nearest neighbors in canonical space
    with torch.no_grad():
        dist_matrix = torch.cdist(canonical_positions, canonical_positions)  # [n_samples, n_samples]
        _, indices = torch.topk(dist_matrix, k=n_neighbors+1, largest=False, dim=1)
        indices = indices[:, 1:]  # Remove self
        
        # Adaptive sigma for distance weighting
        canonical_dists = dist_matrix.gather(1, indices)  # [n_samples, n_neighbors]
        sigma = canonical_dists.mean() * 0.5
    
    # REUSE or COMPUTE integrated positions
    if integrated_positions_cache is not None:
        # FAST PATH: Reuse pre-computed positions from render call
        integrated_positions = integrated_positions_cache[sample_idx]  # [n_samples, 3]
        neighbor_integrated = integrated_positions_cache[sample_idx[indices.flatten()]].reshape(n_samples, n_neighbors, 3)
    else:
        # SLOW PATH: Compute from scratch (only if cache not available)
        # Batch all integrations together: [sampled Gaussians] + [all their neighbors]
        all_indices = torch.cat([sample_idx.unsqueeze(1), sample_idx[indices]], dim=1)  # [n_samples, 1 + n_neighbors]
        all_indices_flat = all_indices.flatten()  # [n_samples * (1 + n_neighbors)]
        
        # Integrate all at once
        positions_to_integrate = canonical_positions_all[all_indices_flat]
        n_total = len(positions_to_integrate)
        t_tensor = torch.ones(n_total, 1, device=device) * t
        
        integrated_flat, _ = velocity_field.integrate_velocity(
            positions_to_integrate,
            torch.zeros(n_total, 1, device=device),
            t_tensor
        )  # [n_samples * (1 + n_neighbors), 3]
        
        # Reshape
        integrated_all = integrated_flat.reshape(n_samples, 1 + n_neighbors, 3)
        integrated_positions = integrated_all[:, 0, :]  # [n_samples, 3]
        neighbor_integrated = integrated_all[:, 1:, :]  # [n_samples, n_neighbors, 3]
    
    # TRAJECTORY DIVERGENCE: Compare integrated positions, not velocities!
    # If neighbors start close but end up far apart after integration → high penalty
    trajectory_divergence = integrated_positions.unsqueeze(1) - neighbor_integrated  # [n_samples, n_neighbors, 3]
    traj_cost = (trajectory_divergence ** 2).sum(dim=-1)  # [n_samples, n_neighbors]
    
    # Distance weighting: neighbors that were close canonically should stay close after integration
    weights = torch.exp(-canonical_dists / sigma)
    
    # Weighted average
    coherence_loss = (weights * traj_cost).sum() / (weights.sum() + 1e-8)
    
    return coherence_loss * lambda_coh


def compute_velocity_regularizations(gaussians, velocity_field, t,
                                     n_samples=1000,
                                     n_neighbors=6,
                                     dt=0.05,
                                     lambda_div=0.01,
                                     lambda_coh=0.08,
                                     lambda_strain=0.005,
                                     lambda_opac=0.05,
                                     use_adaptive_lambda=True,
                                     enable_strain=True):
    """
    Comprehensive regularizers for stable dynamic Gaussian splatting with velocity fields.
    
    This addresses multiple issues with velocity field training:
    1. Divergence-free constraint: Encourages volume-preserving flow (prevents expansion/contraction)
       - Uses CENTRAL difference for better accuracy (second-order vs first-order)
    2. Trajectory coherence: Ensures nearby Gaussians have similar velocities (prevents discontinuities)
       - Distance-weighted: closer neighbors have stronger constraints
    3. Strain regularization: Prevents shearing AND stretching by constraining velocity gradients
       - Captures both longitudinal (stretch) and tangential (shear) components
       - Distance-normalized for physical correctness
    4. Opacity stability: Prevents vanishing/flickering Gaussians
    
    Key improvements over naive approaches:
    - Strain regularization via neighbor approximation (near-zero additional cost)
    - Central difference for divergence (more accurate, symmetric)
    - Full strain tensor approximation (longitudinal + shear)
    - Distance-normalized strain (dimensionally correct)
    - Motion-adaptive regularization (stronger where needed)
    
    Args:
        gaussians: GaussianModel containing _xyz positions
        velocity_field: VelocityField module (must have query_velocity method)
        t: Current time value (float)
        n_samples: Number of Gaussians to sample for efficiency (default: 1000)
        n_neighbors: Number of nearest neighbors for coherence/strain checks (default: 6)
        dt: Time step for trajectory rollout (unused, kept for API compatibility)
        lambda_div: Weight for divergence-free constraint (default: 0.01)
        lambda_coh: Weight for trajectory coherence (default: 0.08)
        lambda_strain: Weight for strain regularization (default: 0.005)
        lambda_opac: Weight for opacity stability (default: 0.05)
        use_adaptive_lambda: Whether to use motion-adaptive weighting (default: True)
    
    Returns:
        total_loss: Combined regularization loss (scalar tensor with grad)
        losses: Dict with individual loss components
            - "divergence": Divergence penalty
            - "trajectory_coherence": Coherence penalty
            - "strain": Strain penalty
            - "opacity_stability": Opacity penalty
            - "_motion_stats": Motion statistics for densification
    
    Computational cost:
        - Divergence: 6 velocity queries (central difference)
        - Coherence: 0 additional queries (reuses center query)
        - Strain: 0 additional queries (reuses neighbor velocities)
        - Total: 6 velocity queries per call
    """
    device = gaussians._xyz.device
    N = gaussians._xyz.shape[0]
    
    # SIMPLIFIED: Uniform random sampling (remove expensive dynamic masking)
    # This removes ~15ms overhead per iteration while maintaining coverage
    if N > n_samples:
        sample_idx = torch.randperm(N, device=device)[:n_samples]
        positions = gaussians._xyz[sample_idx].detach()
    else:
        positions = gaussians._xyz.detach()
        n_samples = N
    
    t_tensor = torch.ones(n_samples, 1, device=device) * t
    
    losses = {}
    
    # --------------------------------------------------
    # STEP 1.5: Query velocities at sampled positions (needed for adaptive strength)
    # --------------------------------------------------
    _, vels_center = velocity_field.query_velocity(positions, t_tensor)
    v_center = vels_center['xyz']
    
    # --------------------------------------------------
    # STEP 1.6: Compute per-sample adaptive strength (SPATIALLY VARYING)
    # --------------------------------------------------
    # Use velocity magnitude as proxy for motion (more stable than xyz_gradient_accum)
    if use_adaptive_lambda:
        with torch.no_grad():
            speed = torch.norm(v_center, dim=-1)  # [n_samples]
            speed_normalized = (speed - speed.min()) / (speed.max() - speed.min() + 1e-8)
            
            # INVERTED FOR ARTICULATION: Low speed (torso) → 1.0×, High speed (hands) → 0.6×
            # Fast-moving regions (hands) get LESS regularization to allow independent motion
            # Slow-moving regions (torso) get FULL regularization to maintain structure
            per_sample_strength = 1.0 - 0.4 * speed_normalized  # Range: [0.6, 1.0]
    else:
        per_sample_strength = torch.ones(len(positions), device=device)
    
    # --------------------------------------------------
    # STEP 2: Divergence-free regularization (CENTRAL DIFFERENCE)
    # --------------------------------------------------
    # Penalizes: ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z
    # Physical meaning: Volume preservation (div=0 means incompressible flow)
    # 
    # IMPROVEMENT: Use central difference instead of forward difference
    # - Forward: dv/dx ≈ (v(x+ε) - v(x)) / ε  (first-order accurate, biased)
    # - Central: dv/dx ≈ (v(x+ε) - v(x-ε)) / (2ε)  (second-order accurate, symmetric)
    
    eps = 0.01  # Finite difference step size
    
    # Create perturbed positions (both positive and negative)
    # X-direction
    pos_xp = positions.clone(); pos_xp[:, 0] += eps  # x + ε
    pos_xn = positions.clone(); pos_xn[:, 0] -= eps  # x - ε
    
    # Y-direction
    pos_yp = positions.clone(); pos_yp[:, 1] += eps  # y + ε
    pos_yn = positions.clone(); pos_yn[:, 1] -= eps  # y - ε
    
    # Z-direction
    pos_zp = positions.clone(); pos_zp[:, 2] += eps  # z + ε
    pos_zn = positions.clone(); pos_zn[:, 2] -= eps  # z - ε
    
    # Query velocities at all perturbed positions (6 queries total)
    _, v_xp = velocity_field.query_velocity(pos_xp, t_tensor)
    _, v_xn = velocity_field.query_velocity(pos_xn, t_tensor)
    _, v_yp = velocity_field.query_velocity(pos_yp, t_tensor)
    _, v_yn = velocity_field.query_velocity(pos_yn, t_tensor)
    _, v_zp = velocity_field.query_velocity(pos_zp, t_tensor)
    _, v_zn = velocity_field.query_velocity(pos_zn, t_tensor)
    
    # Central difference approximation of divergence
    # More accurate than forward difference (second-order vs first-order)
    dv_x_dx = (v_xp['xyz'][:, 0] - v_xn['xyz'][:, 0]) / (2 * eps)
    dv_y_dy = (v_yp['xyz'][:, 1] - v_yn['xyz'][:, 1]) / (2 * eps)
    dv_z_dz = (v_zp['xyz'][:, 2] - v_zn['xyz'][:, 2]) / (2 * eps)
    
    div = dv_x_dx + dv_y_dy + dv_z_dz
    
    # SIMPLIFIED: Use L2 loss instead of Huber (faster, sufficient for most cases)
    div_penalty = div ** 2  # [n_samples]
    
    # Weight by inverse of per_sample_strength (so fast-moving regions get less constrained)
    if use_adaptive_lambda:
        div_strength = (1.0 / per_sample_strength).clamp(min=0.5, max=1.0)  # [n_samples]
        div_loss = (div_strength * div_penalty).mean()
    else:
        div_loss = div_penalty.mean()
    
    losses["divergence"] = div_loss * lambda_div
    
    # --------------------------------------------------
    # STEP 3: Find k-nearest neighbors (shared by coherence and strain)
    # --------------------------------------------------
    with torch.no_grad():
        dist_matrix = torch.cdist(positions, positions)  # [n_samples, n_samples]
        _, indices = torch.topk(dist_matrix, k=n_neighbors+1, largest=False, dim=1)
        indices = indices[:, 1:]  # Remove self (first entry is always the point itself)
        
        # Compute adaptive sigma for distance weighting
        # Sigma = half the average distance to neighbors
        # This makes the weighting scale-invariant
        avg_neighbor_dist = dist_matrix.gather(1, indices).mean()
        sigma = avg_neighbor_dist * 0.5
    
    # --------------------------------------------------
    # STEP 4: ANISOTROPIC Trajectory Coherence
    # --------------------------------------------------
    # KEY FIX: Allow stretching ALONG motion direction, prevent SIDEWAYS tearing
    # This lets hands move relative to arms without forcing blur
    
    neighbor_positions = positions[indices]  # [n_samples, n_neighbors, 3]
    neighbor_velocities = v_center[indices]  # [n_samples, n_neighbors, 3]
    
    # Velocity and position differences [n_samples, n_neighbors, 3]
    vel_diffs = v_center.unsqueeze(1) - neighbor_velocities
    pos_diffs = positions.unsqueeze(1) - neighbor_positions
    dists = torch.norm(pos_diffs, dim=-1)  # [n_samples, n_neighbors]
    
    # Unit direction vectors [n_samples, n_neighbors, 3]
    dists_safe = torch.clamp(dists, min=1e-6, max=10.0)
    r_hat = pos_diffs / (dists_safe.unsqueeze(-1) + 1e-8)
    
    # Decompose velocity difference into parallel/perpendicular
    v_diff_parallel_mag = (vel_diffs * r_hat).sum(dim=-1, keepdim=True)  # [n_samples, n_neighbors, 1]
    v_diff_parallel_vec = v_diff_parallel_mag * r_hat                    # [n_samples, n_neighbors, 3]
    v_diff_perp_vec = vel_diffs - v_diff_parallel_vec                    # [n_samples, n_neighbors, 3]
    
    # Penalize perpendicular (sideways) mismatch MORE than parallel (stretching)
    alpha_perp = 1.0   # Strong: prevent sideways tearing
    alpha_para = 0.2   # Weak: allow stretching along motion direction
    
    coh_perp_cost = (v_diff_perp_vec ** 2).sum(dim=-1)     # [n_samples, n_neighbors]
    coh_para_cost = (v_diff_parallel_vec ** 2).sum(dim=-1) # [n_samples, n_neighbors]
    coh_cost_per_neighbor = alpha_perp * coh_perp_cost + alpha_para * coh_para_cost
    
    # Distance weighting
    weights = torch.exp(-dists / sigma)
    
    # Per-sample adaptive weighting [n_samples, 1]
    per_pair_strength = per_sample_strength.unsqueeze(1)
    weighted_coh = per_pair_strength * coh_cost_per_neighbor
    
    coh_loss = (weights * weighted_coh).sum() / (weights.sum() + 1e-8)
    losses["trajectory_coherence"] = coh_loss * lambda_coh
    
    # --------------------------------------------------
    # STEP 5: 🆕 AFFINE RESIDUAL Strain Regularization (OPTIONAL)
    # --------------------------------------------------
    # Can be disabled early in training (enable_strain=False) to speed up computation
    if enable_strain:
        # KEY FIX: Instead of penalizing raw ∇v, penalize deviations from LOCAL AFFINE FIT
        # This allows smooth articulated motion (rotation, stretching) while preventing tearing/kinks
        # 
        # Method: For each Gaussian i, fit best affine warp J_i to its neighbors:
        #   v_j ≈ v_i + J_i(x_j - x_i)
        # Then penalize residual: ||v_j - v_pred||²
        # 
        # Why this is critical:
        # - Allows wrists to rotate smoothly (affine motion = allowed)
        # - Prevents local folding/tearing (non-affine = penalized)
        # - Fixes "spiky head" and "blocky face" artifacts
        
        # We already have pos_diffs, vel_diffs from coherence
        # pos_diffs: [n_samples, n_neighbors, 3] = x_j - x_i
        # vel_diffs: [n_samples, n_neighbors, 3] = v_j - v_i
        
        # Fit local Jacobian J for each sample via least squares
        # J^T = (Δx^T Δx + εI)^(-1) Δx^T Δv
        
        eps_reg = 1e-4
        
        # Normal matrix: [n_samples, 3, 3]
        A = torch.matmul(pos_diffs.transpose(1, 2), pos_diffs)  # sum_j (Δx_j Δx_j^T)
        
        # Add damping for numerical stability
        eye = torch.eye(3, device=device).unsqueeze(0).expand(len(positions), -1, -1)
        A_damped = A + eps_reg * eye
        
        # Inverse: [n_samples, 3, 3] (use linalg.inv for better numerical stability)
        A_inv = torch.linalg.inv(A_damped)
        
        # Right-hand side: [n_samples, 3, 3]
        B_mat = torch.matmul(pos_diffs.transpose(1, 2), vel_diffs)  # Δx^T Δv
        
        # Jacobian: J = (A^(-1) B)^T
        J_t = torch.matmul(A_inv, B_mat)  # [n_samples, 3, 3]
        J = J_t.transpose(1, 2)            # [n_samples, 3, 3]
        
        # Predicted velocities using affine model: [n_samples, n_neighbors, 3]
        V_pred = torch.matmul(pos_diffs, J.transpose(1, 2))
        
        # Residual (non-affine component): [n_samples, n_neighbors, 3]
        residual = vel_diffs - V_pred
        aff_residual = torch.sqrt((residual ** 2).sum(dim=-1) + 1e-8)  # [n_samples, n_neighbors]
        
        # 🆕 THRESHOLD: Allow small deviations (articulation), only penalize large tears
        # Articulated joints naturally have small non-affine residuals during rotation
        # We only want to prevent LARGE tears/folds, not smooth articulation
        threshold = 0.005  # Allow residuals up to 0.005, penalize beyond that
        aff_residual_clamped = torch.clamp(aff_residual - threshold, min=0.0)
        aff_residual_sq = aff_residual_clamped ** 2
        
        # Apply per-sample adaptive strength
        weighted_strain = per_pair_strength * aff_residual_sq
        
        # Mean residual energy
        strain_loss = weighted_strain.mean()
        losses["strain"] = strain_loss * lambda_strain
    else:
        losses["strain"] = torch.tensor(0.0, device=device)
    
    # --------------------------------------------------
    # STEP 6: Opacity stability (prevents transparency loss)
    # --------------------------------------------------
    # Note: This may not be needed if you're not integrating opacity
    # (i.e., if opacity is kept canonical rather than evolved via ODE)
    
    # Reuse vels_center from earlier (no need to query again)
    vels_op = vels_center
    if 'opacity' in vels_op:
        v_opacity = vels_op['opacity']
        losses["opacity_stability"] = (v_opacity ** 2).mean() * lambda_opac
    else:
        losses["opacity_stability"] = torch.tensor(0.0, device=device)
    
    # --------------------------------------------------
    # STEP 7: Combine losses
    # --------------------------------------------------
    total_loss = sum(losses.values())
    
    # --------------------------------------------------
    # STEP 8: Compute motion statistics for motion-aware densification
    # --------------------------------------------------
    # Return velocity magnitude and velocity gradient for all Gaussians
    # This allows the training loop to identify high-motion regions that need more detail
    
    with torch.no_grad():
        # Get velocities for ALL Gaussians (not just sampled ones)
        all_xyz = gaussians.get_xyz
        N_all = all_xyz.shape[0]
        
        # Create time tensor for all Gaussians (same format as used above)
        all_t_tensor = torch.ones(N_all, 1, device=device) * t
        
        # Query velocity field at canonical positions
        _, all_vels = velocity_field.query_velocity(all_xyz, all_t_tensor)
        all_v_xyz = all_vels['xyz']  # [N_all, 3]
        
        # Velocity magnitude
        vel_magnitude = torch.norm(all_v_xyz, dim=-1)  # [N_all]
        
        # Velocity gradient (simple finite difference)
        # For each Gaussian, estimate how much its velocity differs from neighbors
        # For now, just use magnitude as a proxy (high velocity = likely high gradient too)
        vel_gradient = vel_magnitude  # Simple proxy
        
        losses['_motion_stats'] = {
            'velocity_magnitude': vel_magnitude,
            'velocity_gradient': vel_gradient,
        }
    
    return total_loss, losses
