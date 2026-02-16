#!/usr/bin/env python3
"""
Render a smooth spiral video by interpolating between camera viewpoints.
This demonstrates novel view synthesis capability of the 3D Gaussian representation.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
import torchvision
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm


def interpolate_cameras(cam_list, num_interp=10):
    """
    Interpolate between consecutive cameras to create smooth transitions.
    
    Args:
        cam_list: List of Camera objects
        num_interp: Number of interpolated frames between each pair
    
    Returns:
        List of interpolated Camera objects
    """
    interpolated_cams = []
    
    for i in range(len(cam_list)):
        cam1 = cam_list[i]
        cam2 = cam_list[(i + 1) % len(cam_list)]  # Wrap around for loop
        
        # Add the original camera
        interpolated_cams.append(cam1)
        
        # Extract rotation matrices and translation vectors
        R1 = cam1.R  # 3x3
        T1 = cam1.T  # 3
        R2 = cam2.R
        T2 = cam2.T
        
        # Convert rotation matrices to scipy Rotation objects
        rot1 = R.from_matrix(R1)
        rot2 = R.from_matrix(R2)
        
        # Create Slerp interpolator for smooth rotation
        key_times = [0, 1]
        key_rots = R.from_matrix([R1, R2])
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate
        for j in range(1, num_interp):
            t = j / num_interp  # Interpolation parameter [0, 1]
            
            # Interpolate rotation (spherical)
            R_interp = slerp(t).as_matrix()
            
            # Interpolate translation (linear)
            T_interp = (1 - t) * T1 + t * T2
            
            # Create interpolated camera
            interp_cam = Camera(
                colmap_id=cam1.colmap_id,
                R=R_interp,
                T=T_interp,
                FoVx=cam1.FoVx,
                FoVy=cam1.FoVy,
                image=cam1.original_image,  # Placeholder, won't be used
                gt_alpha_mask=None,
                image_name=f"{cam1.image_name}_interp_{j}",
                uid=cam1.uid * 1000 + j,  # Unique ID
                data_device=cam1.data_device,
                time=cam1.time
            )
            interpolated_cams.append(interp_cam)
    
    return interpolated_cams


def render_smooth_spiral(model_params,
                         hidden_params,
                         pipeline_params,
                         iteration,
                         output_dir,
                         num_interp=10,
                         fps=30,
                         num_loops=5,
                         stage="fine"):
    """Render a smooth spiral video with interpolated viewpoints."""
    
    print("=" * 70)
    print("RENDERING SMOOTH INTERPOLATED SPIRAL")
    print("=" * 70)
    print(f"Model: {model_params.model_path}")
    print(f"Iteration: {iteration}")
    print(f"Interpolation: {num_interp} frames between each camera")
    print(f"Loops: {num_loops}")
    print(f"Stage: {stage}")
    print()
    
    # Load scene and model
    print("Loading model...")
    gaussians = GaussianModel(model_params.sh_degree, hidden_params)
    scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Get training cameras
    train_cams = list(scene.getTrainCameras())
    if len(train_cams) == 0:
        raise RuntimeError("No training cameras found for spiral rendering.")
    print(f"  Loaded {len(train_cams)} training cameras")
    
    # Interpolate cameras
    print(f"Interpolating cameras ({num_interp} frames between each)...")
    interpolated_cams = interpolate_cameras(train_cams, num_interp=num_interp)
    print(f"  Total frames per loop: {len(interpolated_cams)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Render all interpolated frames
    print(f"\nRendering {len(interpolated_cams)} interpolated views...")
    
    with torch.no_grad():
        for idx, cam in enumerate(tqdm(interpolated_cams, desc="Rendering")):
            render_pkg = render(cam, gaussians, pipeline_params, background,
                                stage=stage, cam_type=scene.dataset_type)
            rendering = render_pkg["render"]
            
            output_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
            torchvision.utils.save_image(rendering, output_path)
    
    print(f"\n✓ Rendered {len(interpolated_cams)} frames")
    print(f"✓ Saved to: {output_dir}")
    
    # Create video with ffmpeg
    print(f"\nCreating video (looping {num_loops}x)...")
    video_path = os.path.join(os.path.dirname(output_dir), f"smooth_spiral_{num_loops}loops.mp4")
    
    os.system(f"ffmpeg -y -stream_loop {num_loops-1} -framerate {fps} -pattern_type glob "
              f"-i '{output_dir}/frame_*.png' "
              f"-c:v libx264 -pix_fmt yuv420p -crf 18 "
              f"'{video_path}' 2>&1 | grep -E 'frame=|video:' || true")
    
    print(f"\n✓ Video created: {video_path}")
    print(f"  Total duration: ~{len(interpolated_cams) * num_loops / fps:.1f} seconds")
    
    print("\n" + "=" * 70)
    print("✅ SMOOTH SPIRAL COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Frames: {output_dir}/")
    print(f"  Video: {video_path}")
    
    return video_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Render smooth interpolated spiral")
    model_group = ModelParams(parser, sentinel=True)
    pipeline_group = PipelineParams(parser)
    hidden_group = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--configs", type=str, default=None, help="Optional config file to merge (mmcv)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_interp", type=int, default=10, help="Frames to interpolate between cameras")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--num_loops", type=int, default=5, help="Number of complete loops")
    parser.add_argument("--stage", type=str, choices=["coarse", "fine"], default=None,
                        help="Renderer stage to use (default: fine)")
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    
    if not hasattr(args, "output_dir") or args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, f"smooth_spiral_frames_{args.iteration}")
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(args.quiet if hasattr(args, "quiet") else False)
    
    model_params = model_group.extract(args)
    pipeline_params = pipeline_group.extract(args)
    hidden_params = hidden_group.extract(args)
    
    iteration = args.iteration
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(model_params.model_path, f"smooth_spiral_frames_{iteration}")
    
    stage = args.stage or "fine"
    
    render_smooth_spiral(
        model_params,
        hidden_params,
        pipeline_params,
        iteration,
        output_dir,
        num_interp=args.num_interp,
        fps=args.fps,
        num_loops=args.num_loops,
        stage=stage
    )

