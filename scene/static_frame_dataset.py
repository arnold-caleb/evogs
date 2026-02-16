"""
Dataset wrapper to load only a single frame (all viewpoints) for static training.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T


class StaticFrameDataset:
    """
    Loads a single timestep from all camera viewpoints in a DyNeRF dataset.
    Used for training static 3D Gaussian Splatting on frame 0.
    """
    def __init__(
        self,
        datadir,
        frame_idx=0,
        downsample=1.0,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
    ):
        self.root_dir = datadir
        self.frame_idx = frame_idx
        self.img_wh = (int(1352 / downsample), int(1014 / downsample))
        self.downsample = 2704 / self.img_wh[0]
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.transform = T.ToTensor()
        
        # NDC settings (same as Neural3D)
        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]
        self.white_bg = False
        self.ndc_ray = True
        
        # Load camera poses
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        
        # Find all camera directories
        cam_dirs = sorted(glob.glob(os.path.join(self.root_dir, "cam*")))
        cam_dirs = [d for d in cam_dirs if os.path.isdir(d)]
        
        print(f"Found {len(cam_dirs)} cameras")
        print(f"Loading frame {frame_idx} from all cameras...")
        
        # Extract camera info
        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = [focal, focal]
        
        # Convert poses
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        
        # Load only the specified frame from each camera
        self.image_paths = []
        self.image_poses = []
        self.cameras = []
        
        for cam_idx, cam_dir in enumerate(cam_dirs):
            frame_path = os.path.join(cam_dir, "images", f"{frame_idx:04d}.png")
            if os.path.exists(frame_path):
                self.image_paths.append(frame_path)
                self.image_poses.append(poses[cam_idx])
                self.cameras.append(cam_idx)
        
        self.poses = np.stack(self.image_poses)
        
        print(f"Loaded {len(self.image_paths)} viewpoints for frame {frame_idx}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        pose = self.image_poses[idx]
        
        # Load image
        img = Image.open(image_path)
        if self.img_wh != img.size:
            img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # (3, H, W)
        
        # Extract R, T from pose (same as Neural3D_NDC_Dataset)
        R = pose[:3, :3]
        R = -R
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)
        
        # Return tuple format: (image, (R, T), time)
        return img, (R, T), 0.0  # time = 0.0 for static frame


def read_static_frame_info(datadir, frame_idx=0):
    """
    Adapter function for the existing Scene loading infrastructure.
    Returns scene info compatible with the current Scene class.
    """
    from scene.dataset_readers import CameraInfo, SceneInfo, getNerfppNorm, BasicPointCloud
    
    dataset = StaticFrameDataset(datadir, frame_idx=frame_idx)
    
    # Convert to CameraInfo format
    cam_infos = []
    for idx in range(len(dataset)):
        image, (R, T), time = dataset[idx]
        
        # R and T are already numpy arrays from __getitem__
        
        # Calculate FoV
        focal_x, focal_y = dataset.focal
        W, H = dataset.img_wh
        FovX = 2 * np.arctan(W / (2 * focal_x))
        FovY = 2 * np.arctan(H / (2 * focal_y))
        
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,  # Note: CameraInfo uses FovY, renderer uses FoVy  
            FovX=FovX,  # Note: CameraInfo uses FovX, renderer uses FoVx
            image=image,
            image_path=dataset.image_paths[idx],
            image_name=f"frame{frame_idx:04d}_cam{dataset.cameras[idx]:02d}",
            width=W,
            height=H,
            time=time,  # Will be 0.0
            mask=None
        )
        cam_infos.append(cam_info)
    
    # Load point cloud (use existing COLMAP or generate)
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    if not os.path.exists(ply_path):
        print(f"Warning: Point cloud not found at {ply_path}")
        print("Using smart initialization instead...")
        pcd = None
    else:
        from scene.dataset_readers import fetchPly
        pcd = fetchPly(ply_path)
    
    nerf_normalization = getNerfppNorm(cam_infos)
    
    # For static training, use all cameras for both train and test (overfitting is OK - we want quality!)
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=cam_infos,
        test_cameras=cam_infos,  # Use same cameras for test (we want to overfit frame 0!)
        video_cameras=cam_infos[:1],  # Just one view for video
        maxtime=0,  # Static
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    
    return scene_info

