from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
                image_name = f"{index}"
                uid = index
                colmap_id = index
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = getattr(caminfo, "mask", None)
                image_name = getattr(caminfo, "image_name", f"{index}")
                uid = getattr(caminfo, "uid", index)
                colmap_id = getattr(caminfo, "colmap_id", index)
            return Camera(colmap_id=colmap_id, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                            image_name=str(image_name), uid=uid, data_device=torch.device("cuda"), time=time,
                            mask=mask)
        else:
            return self.dataset[index]
            
    def __len__(self):
        return len(self.dataset)

# import os
# import torch
# from torch.utils.data import Dataset
# from PIL import Image as PILImage

# from scene.cameras import Camera
# from utils.general_utils import PILtoTorch
# from utils.graphics_utils import focal2fov


# class FourDGSdataset(Dataset):
#     """Dataset wrapper for DyNeRF 4D Gaussian Splatting"""
    
#     def __init__(self, dataset, args, dataset_type):
#         self.dataset = dataset
#         self.args = args
#         self.dataset_type = dataset_type
    
#     def __getitem__(self, index):
#         try:
#             # Try to unpack as (image, w2c, time) tuple
#             image, w2c, time = self.dataset[index]
#             R, T = w2c
#             FovX = focal2fov(self.dataset.focal[0], image.shape[2])
#             FovY = focal2fov(self.dataset.focal[0], image.shape[1])
#             mask = None
#         except:
#             # Otherwise, it's a CameraInfo object
#             caminfo = self.dataset[index]
            
#             # Load image from path or use pre-loaded image
#             if hasattr(caminfo, 'image') and caminfo.image is not None:
#                 image = caminfo.image
#             elif caminfo.image_path:
#                 image_path = os.path.join(self.args.source_path, caminfo.image_path)
#                 image = PILImage.open(image_path)
#                 # Resolution scaling (0.35 = 473×354 from 1352×1014)
#                 scale_factor = 1
#                 resolution = (int(caminfo.width * scale_factor), int(caminfo.height * scale_factor))
#                 image = PILtoTorch(image, resolution)
#             else:
#                 # No ground truth (extrapolation frames)
#                 scale_factor = 1
#                 resolution = (int(caminfo.width * scale_factor), int(caminfo.height * scale_factor))
#                 image = torch.zeros((3, resolution[1], resolution[0]), dtype=torch.float32)
            
#             R = caminfo.R
#             T = caminfo.T
#             FovX = caminfo.FovX
#             FovY = caminfo.FovY
#             time = caminfo.time
#             mask = getattr(caminfo, 'mask', None)
        
#         return Camera(
#             colmap_id=index,
#             R=R,
#             T=T,
#             FoVx=FovX,
#             FoVy=FovY,
#             image=image,
#             gt_alpha_mask=None,
#             image_name=f"{index}",
#             uid=index,
#             data_device=torch.device("cuda"),
#             time=time,
#             mask=mask
#         )
    
#     def __len__(self):
#         return len(self.dataset)