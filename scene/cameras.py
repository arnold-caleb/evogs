import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    """
    Camera class for DyNeRF 4D Gaussian Splatting.
    Stores camera parameters, transformations, and temporal information.
    """
    
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 data_device="cuda", time=0, mask=None, depth=None):
        super(Camera, self).__init__()
        
        # Camera identifiers
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_name = image_name
        
        # Camera extrinsics
        self.R = R  # Rotation matrix
        self.T = T  # Translation vector
        
        # Camera intrinsics
        self.FoVx = FoVx  # Field of view (horizontal)
        self.FoVy = FoVy  # Field of view (vertical)
        
        # Temporal information (for 4D)
        self.time = time
        
        # Device setup
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(f"[Warning] Custom device {data_device} failed: {e}")
            print("Fallback to default cuda device")
            self.data_device = torch.device("cuda")
        
        # Image setup
        self.original_image = image.clamp(0.0, 1.0)[:3, :, :]
        self.image_height = self.original_image.shape[1]
        self.image_width = self.original_image.shape[2]
        
        # Apply alpha mask if provided
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
        
        # Optional depth and mask
        self.depth = depth
        self.mask = mask
        
        # Depth bounds
        self.znear = 0.01
        self.zfar = 100.0
        
        # Scene transformation
        self.trans = trans
        self.scale = scale
        
        # Compute camera transformations
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    """
    Lightweight camera class for rendering (used in GUI/interactive mode).
    """
    
    def __init__(self, width, height, fovy, fovx, znear, zfar, 
                 world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        
        self.time = time
