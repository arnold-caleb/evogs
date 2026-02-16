#!/usr/bin/env python3
"""
Render temporal evolution video for trained D-NeRF models.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from tqdm import tqdm
import imageio


def render_temporal_video(model_path, source_path, configs, iteration, num_frames=50, camera_index=None):
    """
    Render temporal video showing scene evolution over time.
    
    Args:
        model_path: Path to trained model
        source_path: Path to dataset
        configs: Config file path
        iteration: Which checkpoint to use
        num_frames: Number of temporal frames to render
    """
    # Load arguments
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--iteration', type=int, default=20000)
    parser.add_argument('--num_frames', type=int, default=50)
    parser.add_argument('--camera_index', type=int, default=-1)
    
    # Set up args
    args = parser.parse_args([])
    args.source_path = source_path
    args.model_path = model_path
    args.iteration = iteration
    args.camera_index = camera_index if camera_index is not None else args.camera_index
    
    # Load config
    if configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(configs)
        args = merge_hparams(args, config)
    
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    hidden_args = hp.extract(args)
    
    # Initialize model
    print(f"Loading model from {model_path}, iteration {iteration}")
    gaussians = GaussianModel(sh_degree=3, args=hidden_args)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    # CRITICAL: Set deformation network to eval mode for rendering
    # This ensures velocity field integration happens (not skipped during training)
    if hasattr(gaussians, '_deformation') and gaussians._deformation is not None:
        gaussians._deformation.eval()
        # Also set the inner velocity_field to eval if it exists
        if hasattr(gaussians._deformation, 'velocity_field'):
            gaussians._deformation.velocity_field.eval()
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_cams = scene.getTrainCameras()

    # Choose reference camera
    ref_cam = None
    if args.camera_index is not None and args.camera_index >= 0 and args.camera_index < len(train_cams):
        ref_cam = train_cams[args.camera_index]
        print(f"Using camera index {args.camera_index}")
    else:
        # Use camera with time=0 as reference
        for idx, cam in enumerate(train_cams):
            if hasattr(cam, 'time') and cam.time < 0.01:
                ref_cam = cam
                print(f"Using first t≈0 camera (index {idx})")
                break

    if ref_cam is None:
        mid_idx = len(train_cams) // 2
        ref_cam = train_cams[mid_idx]
        print(f"Falling back to middle camera (index {mid_idx})")

    cam_name = getattr(ref_cam, 'image_name', None)
    print(f"Camera name: {cam_name if cam_name is not None else 'unknown'}")
    
    # Log model and integration details
    num_gaussians = gaussians.get_xyz.shape[0]
    print(f"Number of Gaussians: {num_gaussians:,}")
    
    if hasattr(gaussians, '_deformation') and gaussians._deformation is not None:
        def_net = gaussians._deformation
        is_velocity_field = hasattr(def_net, 'integrate_velocity')
        
        if is_velocity_field:
            ode_method = getattr(def_net, 'ode_method_eval', getattr(def_net, 'ode_method_train', 'unknown'))
            ode_steps = getattr(def_net, 'ode_steps_eval', getattr(def_net, 'ode_steps_train', 'unknown'))
            use_single = getattr(def_net, 'use_single_step', False)
            print(f"Velocity Field Integration:")
            print(f"  Method: {ode_method} ({ode_steps} steps)")
            if use_single:
                print(f"  ⚠️  Using SIMPLIFIED integration (single-step, like displacement field)")
            elif ode_method == 'euler':
                print(f"  Using Euler integration (faster but less accurate than RK4)")
            elif ode_method == 'rk4':
                print(f"  Using RK4 integration (accurate, slower)")
        else:
            print(f"Displacement Field (direct query, no integration)")
    
    print(f"Rendering {num_frames} temporal frames...")
    
    # Render temporal evolution with FPS logging
    frames = []
    render_times = []
    log_interval = max(1, num_frames // 10)  # Log every 10% of frames
    
    # Warmup (first frame)
    print("Warming up...")
    time_val = 0.0
    ref_cam.time = time_val
    with torch.no_grad():
        render_pkg = render(ref_cam, gaussians, pipe, background, stage="fine")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("Starting benchmark...")
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
        time_val = frame_idx / (num_frames - 1)  # 0.0 to 1.0
        
        # Update camera time
        ref_cam.time = time_val
        
        # Time the render
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Render
        with torch.no_grad():
            render_pkg = render(ref_cam, gaussians, pipe, background, stage="fine")
            image = render_pkg["render"]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        render_time = end_time - start_time
        render_times.append(render_time)
        
        # Convert to numpy
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        frames.append(image_np)
        
        # Log FPS periodically
        if (frame_idx + 1) % log_interval == 0 or frame_idx == num_frames - 1:
            recent_fps = 1.0 / render_time if render_time > 0 else 0
            avg_fps = 1.0 / np.mean(render_times) if len(render_times) > 0 else 0
            print(f"  Frame {frame_idx+1}/{num_frames}: {recent_fps:.2f} FPS (avg: {avg_fps:.2f} FPS)")
    
    # Get Gaussian count
    num_gaussians = gaussians.get_xyz.shape[0]
    
    # Print final FPS statistics
    if len(render_times) > 0:
        render_times = np.array(render_times)
        avg_fps = 1.0 / render_times.mean()
        min_fps = 1.0 / render_times.max()
        max_fps = 1.0 / render_times.min()
        std_fps = np.std(1.0 / render_times)
        
        print(f"\n{'='*60}")
        print("RENDERING FPS STATISTICS")
        print(f"{'='*60}")
        print(f"Number of Gaussians: {num_gaussians:,}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"FPS Range: {min_fps:.2f} - {max_fps:.2f}")
        print(f"FPS Std Dev: {std_fps:.2f}")
        print(f"Render Time: {render_times.mean()*1000:.2f} ms (avg), {render_times.min()*1000:.2f} ms (min), {render_times.max()*1000:.2f} ms (max)")
        print(f"{'='*60}")
        
        # Check real-time capability
        if avg_fps >= 30.0:
            print("✅ REAL-TIME CAPABLE (≥30 FPS)")
        elif avg_fps >= 15.0:
            print("⚠️  NEAR REAL-TIME (15-30 FPS)")
        else:
            print("❌ NOT REAL-TIME (<15 FPS)")
        print()
    
    # Save video
    video_dir = os.path.join(model_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"temporal_evolution_{num_frames}frames.mp4")
    
    print(f"\nSaving video to {video_path}")
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    print(f"✓ Video saved! ({len(frames)} frames)")
    
    return video_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Render temporal video for D-NeRF")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=20000)
    parser.add_argument('--num_frames', type=int, default=50)
    parser.add_argument('--camera_index', type=int, default=-1)
    
    args = parser.parse_args()
    
    render_temporal_video(
        args.model_path,
        args.source_path,
        args.configs,
        args.iteration,
        args.num_frames,
        camera_index=args.camera_index
    )

