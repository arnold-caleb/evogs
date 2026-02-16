"""
HexPlane: Multi-resolution spatio-temporal feature grid for 4D Gaussian Splatting.

HexPlane decomposes 4D space-time into six 2D planes:
- 3 spatial planes: XY, XZ, YZ (for spatial features)
- 3 space-time planes: XT, YT, ZT (for temporal features)

Multi-resolution grids are used for coarse-to-fine feature learning.
"""

import itertools
from typing import Optional, Sequence, Iterable, Collection

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """
    Normalize directions to [0, 1] range for spherical harmonics encoding.
    
    Args:
        directions: Batch of direction vectors
    
    Returns:
        Normalized directions in [0, 1]
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    """
    Normalize points to [-1, 1] based on axis-aligned bounding box.
    
    Args:
        pts: Points to normalize
        aabb: [min_bound, max_bound] bounding box
    
    Returns:
        Normalized points in [-1, 1]
    """
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """
    Wrapper for PyTorch grid_sample with automatic dimension handling.
    
    Args:
        grid: Feature grid [B, feature_dim, H, W] or [B, feature_dim, D, H, W]
        coords: Sample coordinates
        align_corners: Whether to align corners in interpolation
    
    Returns:
        Interpolated features at given coordinates
    """
    grid_dim = coords.shape[-1]

    # Add batch dimension if needed
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    # Select appropriate sampler
    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(
            f"Grid-sample called with {grid_dim}D data but only "
            f"implemented for 2D and 3D data."
        )

    # Reshape coordinates for grid_sample
    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    
    # Interpolate features
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear',
        padding_mode='border'
    )
    
    # Reshape output
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(grid_nd: int, in_dim: int, out_dim: int, reso: Sequence[int], 
                    a: float = 0.1, b: float = 0.5):
    """
    Initialize HexPlane grid parameters.
    
    For 4D input (x, y, z, t), creates 6 planes:
    - Spatial: XY, XZ, YZ
    - Temporal: XT, YT, ZT
    
    Args:
        grid_nd: Number of dimensions per plane (2 for HexPlane)
        in_dim: Input dimension (4 for space-time)
        out_dim: Output feature dimension
        reso: Resolution for each dimension [res_x, res_y, res_z, res_t]
        a, b: Uniform initialization range [a, b]
    
    Returns:
        ParameterList of grid coefficients
    """
    assert in_dim == len(reso), "Resolution must match input dimensions"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim

    # Generate all plane combinations (e.g., XY, XZ, YZ, XT, YT, ZT)
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()

    for ci, coo_comb in enumerate(coo_combs):
        # Create grid parameter [1, out_dim, *resolution]
        new_grid_coef = nn.Parameter(
            torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]])
        )
        
        # Initialize time planes to 1 (temporal coherence), spatial planes uniformly
        if has_time_planes and 3 in coo_comb:
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def interpolate_ms_features(pts: torch.Tensor, ms_grids: Collection[Iterable[nn.Module]], 
                            grid_dimensions: int, concat_features: bool, 
                            num_levels: Optional[int]) -> torch.Tensor:
    """
    Interpolate features from multi-scale HexPlane grids.
    
    For each resolution level:
    1. Sample features from each plane
    2. Multiply features across planes (outer product)
    3. Combine across scales (concatenate or sum)
    
    Args:
        pts: Query points [N, 4] (x, y, z, t)
        ms_grids: Multi-scale grids (one per resolution)
        grid_dimensions: Plane dimensions (2 for HexPlane)
        concat_features: If True, concatenate scales; else sum them
        num_levels: Number of resolution levels to use (None = all)
    
    Returns:
        Interpolated multi-scale features
    """
    # Get plane combinations (e.g., XY, XZ, YZ, XT, YT, ZT)
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    
    if num_levels is None:
        num_levels = len(ms_grids)
    
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList

    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        # Start with identity for multiplication
        interp_space = 1.
        
        for ci, coo_comb in enumerate(coo_combs):
            # Interpolate features from this plane
            feature_dim = grid[ci].shape[1]  # [1, out_dim, *reso]
            interp_out_plane = grid_sample_wrapper(
                grid[ci], pts[..., coo_comb]
            ).view(-1, feature_dim)
            
            # Multiply features across planes (outer product)
            interp_space = interp_space * interp_out_plane

        # Combine features across scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    
    return multi_scale_interp


class HexPlaneField(nn.Module):
    """
    HexPlane: Multi-resolution 4D feature grid.
    
    Decomposes 4D space-time (x, y, z, t) into six 2D planes:
    - Spatial planes: XY, XZ, YZ
    - Temporal planes: XT, YT, ZT
    
    Features are interpolated from multiple resolution levels and combined
    via outer product across planes.
    """
    
    def __init__(self, bounds, planeconfig, multires) -> None:
        super().__init__()
        
        # Initialize axis-aligned bounding box
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        # Configuration
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True  # Concatenate multi-scale features

        # Initialize multi-resolution grids
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            
            # Apply multi-resolution only to spatial dimensions (x, y, z)
            # Keep temporal resolution fixed
            config["resolution"] = (
                [r * res for r in config["resolution"][:3]] +  # Spatial: scale by res
                config["resolution"][3:]  # Temporal: keep fixed
            )
            
            # Initialize grid parameters for this resolution
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"]
            )
            
            # Update feature dimension
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]  # Concatenate: sum dimensions
            else:
                self.feat_dim = gp[-1].shape[1]  # Sum: keep same dimension
            
            self.grids.append(gp)
        
        print(f"HexPlane feature dimension: {self.feat_dim}")
    
    @property
    def get_aabb(self):
        """Get axis-aligned bounding box [min, max]"""
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        """Set spatial bounds for the grid"""
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb.cuda(), requires_grad=False)
        print(f"HexPlane: Set AABB = {self.aabb}")

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """
        Query spatio-temporal features from HexPlane grid.
        
        Args:
            pts: 3D points [N, 3]
            timestamps: Time values [N, 1]
        
        Returns:
            Interpolated features [N, feat_dim]
        """
        # Normalize points to [-1, 1] based on AABB
        pts = normalize_aabb(pts, self.aabb)
        
        # Concatenate spatial and temporal coordinates
        pts = torch.cat((pts, timestamps), dim=-1)  # [N, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        
        # Interpolate features from multi-scale grids
        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features,
            num_levels=None
        )

        # Handle empty features
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        return features

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """
        Forward pass: query HexPlane features.
        
        Args:
            pts: 3D points [N, 3]
            timestamps: Time values [N, 1]
        
        Returns:
            Spatio-temporal features [N, feat_dim]
        """
        features = self.get_density(pts, timestamps)
        return features

