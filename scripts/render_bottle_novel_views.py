"""
Render extracted bottle from multiple novel viewpoints.
Validates clean segmentation and shows object from all angles.
"""

import torch
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
import cv2
from utils.graphics_utils import focal2fov
from scene.cameras import Camera


def render_bottle_multiview(model_path, mask_path, iteration=30000, n_views=36):
    """
    Render segmented bottle from circular camera path.
    
    Args:
        model_path: Path to trained static model
        mask_path: Path to bottle mask (.pt file)
        iteration: Model iteration to load
        n_views: Number of views in circular path
    """
    print("=" * 70)
    print("NOVEL VIEW SYNTHESIS: Bottle from Multiple Angles")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading model and mask...")
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipe_params = PipelineParams(parser)
    
    from arguments import ModelHiddenParams
    hyper_params = ModelHiddenParams(parser)
    
    args = parser.parse_args([
        '--model_path', model_path,
        '--source_path', 'data/dynerf/cut_roasted_beef',
        '--sh_degree', '3',
        '--eval',
    ])
    
    # Set dataset type explicitly (DyNeRF STATIC - single frame)
    args.dataset_type = 'dynerf_static'
    args.frame_idx = 0  # Frame 0 (static)
    
    # Extract parameters
    dataset = model_params.extract(args)
    hyper = hyper_params.extract(args)
    pipe = pipe_params.extract(args)
    
    # Load Gaussians directly from PLY (skip Scene to avoid deformation network issues)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Model not found: {ply_path}")
    gaussians.load_ply(ply_path)
    
    # Load cameras manually
    from scene.dataset_readers import sceneLoadTypeCallbacks
    if args.dataset_type == 'dynerf_static':
        scene_info = sceneLoadTypeCallbacks[args.dataset_type](
            dataset.source_path, 
            dataset.white_background, 
            dataset.eval,
            frame_idx=args.frame_idx
        )
    else:
        scene_info = sceneLoadTypeCallbacks[args.dataset_type](
            dataset.source_path, 
            dataset.white_background, 
            dataset.eval,
            extension=dataset.data_device
        )
    from utils.camera_utils import Camera
    train_cameras = []
    for cam_info in scene_info.train_cameras:
        train_cameras.append(Camera(
            colmap_id=cam_info.uid,
            R=cam_info.R, T=cam_info.T,
            FoVx=cam_info.FovX, FoVy=cam_info.FovY,
            image=cam_info.image, gt_alpha_mask=None,
            image_name=cam_info.image_name, uid=cam_info.uid,
            data_device=dataset.data_device
        ))
    
    # Load bottle mask
    bottle_mask = torch.load(mask_path).cuda()
    print(f"   Bottle Gaussians: {bottle_mask.sum()} / {gaussians._xyz.shape[0]}")
    
    # Extract bottle Gaussians
    bottle_gaussians = GaussianModel(gaussians.active_sh_degree, hyper)
    bottle_gaussians._xyz = gaussians._xyz[bottle_mask]
    bottle_gaussians._features_dc = gaussians._features_dc[bottle_mask]
    bottle_gaussians._features_rest = gaussians._features_rest[bottle_mask]
    bottle_gaussians._opacity = gaussians._opacity[bottle_mask]
    bottle_gaussians._scaling = gaussians._scaling[bottle_mask]
    bottle_gaussians._rotation = gaussians._rotation[bottle_mask]
    
    # Compute bottle center and radius
    bottle_center = bottle_gaussians._xyz.mean(dim=0)
    bottle_radius = (bottle_gaussians._xyz - bottle_center).norm(dim=1).max().item()
    
    print(f"   Bottle center: {bottle_center.cpu().numpy()}")
    print(f"   Bottle radius: {bottle_radius:.3f}")
    
    # Create circular camera path
    print(f"\n[2/4] Creating {n_views}-view circular camera path...")
    cameras = []
    camera_radius = bottle_radius * 2.5  # View from 2.5× bottle radius
    camera_height = bottle_center[1].item() + 0.1  # Slightly above center
    
    ref_camera = train_cameras[0]
    
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        
        # Camera position on circle
        cam_x = bottle_center[0].item() + camera_radius * np.cos(angle)
        cam_z = bottle_center[2].item() + camera_radius * np.sin(angle)
        cam_pos = np.array([cam_x, camera_height, cam_z])
        
        # Look at bottle center
        look_at = bottle_center.cpu().numpy()
        up = np.array([0, 1, 0])
        
        # Compute rotation matrix
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        
        R = np.stack([right, up_corrected, -forward], axis=0)  # Row-major
        T = -R @ cam_pos
        
        # Create camera
        cam = Camera(
            colmap_id=i,
            R=R,
            T=T,
            FoVx=ref_camera.FoVx,
            FoVy=ref_camera.FoVy,
            image=torch.zeros(3, ref_camera.image_height, ref_camera.image_width),
            gt_alpha_mask=None,
            image_name=f"novel_{i:03d}",
            uid=i,
            data_device="cuda"
        )
        cameras.append(cam)
    
    # Render from each view
    print(f"\n[3/4] Rendering {n_views} views...")
    frames = []
    
    bg_color = [1, 1, 1]  # White background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            render_pkg = render(cam, bottle_gaussians, pipe, background, stage="coarse")
            rendered = render_pkg["render"]
        
        frame = (rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        
        if (i + 1) % 6 == 0:
            print(f"   Rendered {i+1}/{n_views} views")
    
    # Save video
    print(f"\n[4/4] Saving video...")
    import imageio
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave("bottle_novel_views.mp4", frames_rgb, fps=10)
    print(f"   ✓ Saved: bottle_novel_views.mp4")
    
    # Also save 6-view grid
    grid_indices = [0, 6, 12, 18, 24, 30]  # 60° apart
    grid_size = 2  # 3x2 grid
    h, w = frames[0].shape[:2]
    grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    grid.fill(255)  # White background
    
    for idx, frame_idx in enumerate(grid_indices):
        row = idx // 3
        col = idx % 3
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = frames[frame_idx]
    
    cv2.imwrite("bottle_novel_views_grid.png", grid)
    print(f"   ✓ Saved: bottle_novel_views_grid.png (6-view grid)")
    
    print("\n" + "=" * 70)
    print("NOVEL VIEW SYNTHESIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated:")
    print("  - bottle_novel_views.mp4 (36 views, circular path)")
    print("  - bottle_novel_views_grid.png (6-view grid)")
    print("\nThis validates clean bottle extraction!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--n_views", type=int, default=36)
    args = parser.parse_args()
    
    render_bottle_multiview(args.model_path, args.mask, args.iteration, args.n_views)

