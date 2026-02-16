#!/usr/bin/env python3
"""
Render temporal evolution video using trained velocity field.

This script:
1. Loads trained Neural ODE velocity field model
2. Creates smooth spiral camera path (interpolated viewpoints)
3. Renders scene at all 50 temporal frames from each camera viewpoint
4. Saves temporal video showing scene dynamics with smooth camera motion
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, ModelHiddenParams
import torchvision
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import imageio
from PIL import Image


def interpolate_cameras(cam_list, num_interp=10):
    """
    Interpolate between consecutive cameras for smooth spiral motion.
    
    Args:
        cam_list: List of Camera objects
        num_interp: Number of interpolated frames between each pair
    
    Returns:
        List of interpolated Camera objects
    """
    interpolated_cams = []
    
    for i in range(len(cam_list)):
        cam1 = cam_list[i]
        cam2 = cam_list[(i + 1) % len(cam_list)]
        
        interpolated_cams.append(cam1)
        
        R1 = cam1.R
        T1 = cam1.T
        R2 = cam2.R
        T2 = cam2.T
        
        key_rots = R.from_matrix(np.stack([R1, R2]))
        slerp = Slerp([0, 1], key_rots)
        
        for j in range(1, num_interp):
            t = j / num_interp
            R_interp = slerp(t).as_matrix()
            T_interp = (1 - t) * T1 + t * T2
            
            interp_cam = Camera(
                colmap_id=cam1.colmap_id,
                R=R_interp,
                T=T_interp,
                FoVx=cam1.FoVx,
                FoVy=cam1.FoVy,
                image=cam1.original_image,
                gt_alpha_mask=None,
                image_name=f"{cam1.image_name}_interp_{j}",
                uid=cam1.uid,
                data_device="cuda"
            )
            interp_cam.time = cam1.time  # Keep same time
            interpolated_cams.append(interp_cam)
    
    return interpolated_cams


def render_temporal_video(args, model_args, hidden_args, pipe_args):
    """
    Render temporal evolution video with spiral camera motion.
    
    For each of N temporal frames (0-49):
        For each spiral camera viewpoint:
            Render scene at time t from that viewpoint
    
    Creates smooth video showing both:
    - Temporal dynamics (scene evolution over time)
    - Spatial dynamics (camera moving around scene)
    """
    print("=" * 80)
    print("RENDERING VELOCITY FIELD TEMPORAL VIDEO")
    print("=" * 80)
    
    # Load model
    print(f"\n1️⃣ Loading trained model from: {model_args.model_path}")
    
    # Initialize model - use config's deformation type, don't force velocity field
    # hidden_args.use_velocity_field is already set by the config
    print(f"   Config: use_velocity_field = {getattr(hidden_args, 'use_velocity_field', False)}")
    gaussians = GaussianModel(sh_degree=3, args=hidden_args)
    scene = Scene(model_args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    print(f"   ✅ Model loaded (iteration {args.iteration})")
    print(f"   Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"   Deformation type: {type(gaussians._deformation).__name__}")
    
    # Load anchor Gaussians if multi-anchor training
    if getattr(hidden_args, 'use_multi_anchor', False):
        print(f"\n   🔗 Loading anchor Gaussians for multi-anchor rendering...")
        anchor_gaussians = {}
        
        for anchor_t, ckpt_path in hidden_args.anchor_checkpoints.items():
            print(f"      Loading anchor at t={anchor_t}: {os.path.basename(ckpt_path)}")
            
            # Load checkpoint manually
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_params_tuple = checkpoint[0]
            
            # Extract only Gaussian parameters (NOT deformation network!)
            # We only need positions/attributes, not the deformation network
            active_sh_degree = model_params_tuple[0]
            _xyz = model_params_tuple[1]
            # Skip deformation_state at index 2 (incompatible with velocity field)
            _deformation_table = model_params_tuple[3]
            _features_dc = model_params_tuple[4]
            _features_rest = model_params_tuple[5]
            _scaling = model_params_tuple[6]
            _rotation = model_params_tuple[7]
            _opacity = model_params_tuple[8]
            max_radii2D = model_params_tuple[9]
            xyz_gradient_accum = model_params_tuple[10]
            denom = model_params_tuple[11]
            spatial_lr_scale = model_params_tuple[13]
            
            # Create lightweight anchor (no deformation network needed!)
            class AnchorGaussians:
                """Lightweight container for frozen anchor Gaussians (no deformation)"""
                def __init__(self):
                    pass
            
            anchor_gauss = AnchorGaussians()
            anchor_gauss.active_sh_degree = active_sh_degree
            anchor_gauss._xyz = _xyz
            anchor_gauss._features_dc = _features_dc
            anchor_gauss._features_rest = _features_rest
            anchor_gauss._scaling = _scaling
            anchor_gauss._rotation = _rotation
            anchor_gauss._opacity = _opacity
            anchor_gauss.max_radii2D = max_radii2D
            anchor_gauss.spatial_lr_scale = spatial_lr_scale
            
            # Move to CUDA (detach to remove from computation graph)
            anchor_gauss._xyz = anchor_gauss._xyz.detach().cuda()
            anchor_gauss._scaling = anchor_gauss._scaling.detach().cuda()
            anchor_gauss._rotation = anchor_gauss._rotation.detach().cuda()
            anchor_gauss._opacity = anchor_gauss._opacity.detach().cuda()
            anchor_gauss._features_dc = anchor_gauss._features_dc.detach().cuda()
            anchor_gauss._features_rest = anchor_gauss._features_rest.detach().cuda()
            
            anchor_gaussians[anchor_t] = anchor_gauss
            print(f"         ✓ Loaded {anchor_gauss._xyz.shape[0]} Gaussians")
        
        # Attach to main gaussians object
        gaussians.anchor_gaussians = anchor_gaussians
        print(f"   ✅ Multi-anchor rendering enabled (nearest-anchor integration)")
        print(f"      Anchors at: {list(anchor_gaussians.keys())}")
    else:
        print(f"   ℹ️  Standard rendering (no multi-anchor)")
    
    # Get camera viewpoints (avoid lazy dataset access by converting to list first)
    print(f"\n2️⃣ Selecting camera list...")
    camera_source = args.camera_source.lower()
    if camera_source == "test":
        camera_iterable = scene.getTestCameras()
    elif camera_source == "video":
        camera_iterable = scene.getVideoCameras()
    else:
        camera_iterable = scene.getTrainCameras()
        camera_source = "train"
    
    camera_list = list(camera_iterable)
    if len(camera_list) == 0:
        raise RuntimeError(f"No cameras available for source '{camera_source}'")
    
    num_frames = min(args.num_temporal_frames, len(camera_list))
    selected_cameras = camera_list[:num_frames]
    print(f"   Source: {camera_source} ({len(camera_list)} total views, rendering {num_frames})")
    
    # Background
    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cam_type = scene.dataset_type
    
    # Render frames
    all_frames = []
    
    # Use faster ODE solver for rendering
    original_training = gaussians._deformation.training
    gaussians._deformation.eval()
    
    # Override ODE method for speed (only for velocity field)
    # Displacement field doesn't use ODE integration, so skip this
    if hasattr(gaussians._deformation, 'velocity_field'):
        if hasattr(gaussians._deformation.velocity_field, 'ode_method_eval'):
            original_ode_method = gaussians._deformation.velocity_field.ode_method_eval
            gaussians._deformation.velocity_field.ode_method_eval = 'rk4'
            gaussians._deformation.velocity_field.ode_steps_eval = 4
    
    # Prepare output folder to store rendered frames for metric computation
    video_output_root = os.path.join(model_args.model_path, "videos", f"ours_{args.iteration}")
    render_dir = os.path.join(video_output_root, "renders")
    os.makedirs(render_dir, exist_ok=True)

    with torch.no_grad():
        # Render each view
        for idx, view in enumerate(tqdm(selected_cameras, desc="Rendering frames")):
            render_pkg = render(view, gaussians, pipe_args, background, stage="fine", cam_type=cam_type)
            image = render_pkg["render"]
            
            # Convert to numpy [H, W, 3]
            frame = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            frame_uint8 = (frame * 255).astype(np.uint8)
            all_frames.append(frame_uint8)

            # Save render frame for downstream metrics
            Image.fromarray(frame_uint8).save(os.path.join(render_dir, f"{idx:05d}.png"))
    
    # Restore original settings (only for velocity field)
    if hasattr(gaussians._deformation, 'velocity_field'):
        if hasattr(gaussians._deformation.velocity_field, 'ode_method_eval'):
            gaussians._deformation.velocity_field.ode_method_eval = original_ode_method
    if original_training:
        gaussians._deformation.train()
    
    print(f"\n   ✅ Rendered {len(all_frames)} frames")
    
    # Save video
    print(f"\n4️⃣ Saving video...")
    output_dir = os.path.join(model_args.model_path, "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, f"velocity_field_temporal_{num_frames}frames_{args.num_interp}interp.mp4")
    
    # Save with imageio
    imageio.mimsave(
        video_path,
        all_frames,
        fps=args.fps,
        quality=9,
        macro_block_size=1
    )
    
    print(f"   ✅ Video saved: {video_path}")
    print(f"   FPS: {args.fps}")
    print(f"   Duration: {len(all_frames) / args.fps:.1f} seconds")
    print(f"   Resolution: {all_frames[0].shape[0]}×{all_frames[0].shape[1]}")
    
    print("\n" + "=" * 80)
    print("✅ TEMPORAL VIDEO RENDERING COMPLETE!")
    print("=" * 80)
    print(f"\nVideo location: {video_path}")
    print(f"\nThis video shows:")
    print(f"  - {num_frames} frames from FIXED camera viewpoint")
    print(f"  - Time evolves: t=0.0 → t=1.0")
    print(f"  - Scene dynamics learned by Neural ODE velocity field")
    print(f"  - Pure temporal evolution (no camera motion)")
    
    return video_path


if __name__ == "__main__":
    # Use the standard argument parsers from the codebase
    parser = ArgumentParser(description="Render temporal video with velocity field")
    model_params = ModelParams(parser)
    hidden_params = ModelHiddenParams(parser)
    pipeline_params = PipelineParams(parser)
    
    # Add config file argument (standard in this codebase)
    parser.add_argument('--configs', type=str, default=None,
                       help="Path to config file")
    
    # Add rendering-specific arguments
    parser.add_argument("--iteration", type=int, default=14000,
                       help="Iteration to load")
    parser.add_argument("--num_temporal_frames", type=int, default=300,
                       help="Number of temporal frames to render (dataset has 300 frames per viewpoint)")
    parser.add_argument("--num_interp", type=int, default=5,
                       help="Number of interpolated viewpoints between cameras (default: 5 for speed)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for video")
    parser.add_argument("--camera_source", type=str, default="test",
                       choices=["train", "test", "video"],
                       help="Camera set to render (defaults to test views)")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    # Extract standard parameters
    model_args = model_params.extract(args)
    hidden_args = hidden_params.extract(args)
    pipe_args = pipeline_params.extract(args)
    
    render_temporal_video(args, model_args, hidden_args, pipe_args)

