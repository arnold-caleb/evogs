#!/usr/bin/env python3
"""
Render temporal evolution video for trained HyperNeRF models.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from tqdm import tqdm
import imageio


def render_temporal_video(model_path, source_path, configs, iteration, num_frames=150, camera_index=None):
    """
    Render temporal video showing scene evolution over time.
    
    Args:
        model_path: Path to trained model
        source_path: Path to dataset
        configs: Config file path
        iteration: Which checkpoint to use
        num_frames: Number of temporal frames to render (default 150 for HyperNeRF)
    """
    # Load arguments
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--iteration', type=int, default=14000)
    parser.add_argument('--num_frames', type=int, default=150)
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
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Get a fixed camera viewpoint (use first train camera)
    train_cams = scene.getTrainCameras()
    # Use camera with time close to 0 as reference unless index specified
    ref_cam = None
    if args.camera_index is not None and args.camera_index >= 0 and args.camera_index < len(train_cams):
        ref_cam = train_cams[args.camera_index]
        print(f"Using camera index {args.camera_index}")
    else:
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
    print(f"Rendering {num_frames} temporal frames...")
    
    # Render temporal evolution
    frames = []
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
        time = frame_idx / (num_frames - 1)  # 0.0 to 1.0
        
        # Update camera time
        ref_cam.time = time
        
        # Render
        with torch.no_grad():
            render_pkg = render(ref_cam, gaussians, pipe, background, stage="fine")
            image = render_pkg["render"]
        
        # Convert to numpy
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        frames.append(image_np)
    
    # Save video
    video_dir = os.path.join(model_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"temporal_evolution_{num_frames}frames.mp4")
    
    print(f"\nSaving video to {video_path}")
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    print(f"✓ Video saved! ({len(frames)} frames)")
    
    return video_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Render temporal video for HyperNeRF")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=14000)
    parser.add_argument('--num_frames', type=int, default=150)
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

