"""
Benchmark Rendering FPS for Velocity Field Models

This script measures the rendering performance (FPS) for velocity field models
by rendering multiple frames at different time steps and camera viewpoints.

Usage:
    python scripts/benchmark_rendering_fps.py \
        --model_path output/dnerf/bouncingballs_20251031_162435 \
        --iteration 20000 \
        --n_frames 100 \
        --n_warmup 10
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import copy
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.system_utils import searchForMaxIteration


def load_model_and_scene(model_path, iteration, source_path=None):
    """Load trained Gaussian model and scene."""
    print(f"Loading model from {model_path} at iteration {iteration}")
    
    # Parse arguments to get config
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    
    # Set model path
    sys.argv = [
        'benchmark_rendering_fps.py',
        '--model_path', model_path,
    ]
    if source_path:
        sys.argv.extend(['--source_path', source_path])
    
    args, _ = parser.parse_known_args()
    args = get_combined_args(parser)
    
    # Handle iteration=-1 (find latest)
    if iteration == -1:
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        if not os.path.exists(point_cloud_dir):
            raise FileNotFoundError(f"Point cloud directory not found: {point_cloud_dir}")
        iteration = searchForMaxIteration(point_cloud_dir)
        print(f"Found latest iteration: {iteration}")
    
    # Initialize model
    gaussians = GaussianModel(args.sh_degree, args)
    
    # Load scene (this will load cameras)
    try:
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
        print(f"✅ Scene loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load scene: {e}")
        print(f"   Will try to load model directly from checkpoint...")
        scene = None
        
        # Load checkpoint directly
        checkpoint_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}")
        ply_path = os.path.join(checkpoint_path, "point_cloud.ply")
        
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Point cloud not found: {ply_path}")
        
        print(f"Loading point cloud from {ply_path}")
        gaussians.load_ply(ply_path)
        
        print(f"Loading model checkpoint from {checkpoint_path}")
        gaussians.load_model(checkpoint_path)
        
        # Set AABB bounds from point cloud if needed
        positions = gaussians.get_xyz.detach()
        if positions.numel() > 0:
            xyz_max = positions.max(dim=0)[0].cpu().numpy()
            xyz_min = positions.min(dim=0)[0].cpu().numpy()
            
            if hasattr(gaussians, '_deformation') and gaussians._deformation is not None:
                if hasattr(gaussians._deformation, 'deformation_net'):
                    gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
                elif hasattr(gaussians._deformation, 'set_aabb'):
                    gaussians._deformation.set_aabb(xyz_max, xyz_min)
    
    return gaussians, scene, args


def benchmark_rendering_fps(gaussians, scene, args, n_frames=100, n_warmup=10, 
                            camera_source='test', time_range=(0.0, 1.0)):
    """
    Benchmark rendering FPS by rendering multiple frames.
    
    Args:
        gaussians: GaussianModel
        scene: Scene object (can be None if we create synthetic cameras)
        args: Model arguments
        n_frames: Number of frames to render for benchmarking
        n_warmup: Number of warmup frames (not counted in FPS)
        camera_source: Which cameras to use ('train', 'test', 'video')
        time_range: (t_start, t_end) time range to render
    
    Returns:
        fps: Average FPS
        fps_stats: Dictionary with detailed statistics
    """
    print(f"\n{'='*80}")
    print("BENCHMARKING RENDERING FPS")
    print(f"{'='*80}")
    
    # Get cameras
    if scene is not None:
        if camera_source == 'test':
            cameras = list(scene.getTestCameras())
        elif camera_source == 'video':
            cameras = list(scene.getVideoCameras())
        else:
            cameras = list(scene.getTrainCameras())
        
        if len(cameras) == 0:
            print(f"⚠️  No {camera_source} cameras found, using train cameras")
            cameras = list(scene.getTrainCameras())
    else:
        print("⚠️  No scene available, creating synthetic camera")
        # Create a simple synthetic camera
        from scene.cameras import Camera
        dummy_image = torch.zeros((3, 800, 800), dtype=torch.float32, device="cuda")
        cameras = [Camera(
            colmap_id=0,
            R=torch.eye(3, dtype=torch.float32),
            T=torch.zeros(3, dtype=torch.float32),
            FoVx=1.0,
            FoVy=1.0,
            image=dummy_image,
            gt_alpha_mask=None,
            image_name="synthetic",
            uid=0,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            data_device="cuda",
            time=0.0
        )]
    
    print(f"   Using {len(cameras)} cameras from '{camera_source}' set")
    print(f"   Rendering {n_frames} frames (with {n_warmup} warmup)")
    print(f"   Time range: {time_range[0]:.3f} to {time_range[1]:.3f}")
    
    # Setup pipeline
    from arguments import PipelineParams
    pipeline = PipelineParams(parser=argparse.ArgumentParser())
    
    # Background color
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Get dataset type
    cam_type = getattr(args, 'dataset_type', 'dnerf')
    
    # Model info
    print(f"\n   Model Info:")
    print(f"      Gaussians: {gaussians.get_xyz.shape[0]}")
    if hasattr(gaussians, '_deformation') and gaussians._deformation is not None:
        deformation_type = type(gaussians._deformation).__name__
        print(f"      Deformation: {deformation_type}")
        is_velocity_field = hasattr(gaussians._deformation, 'integrate_velocity')
        print(f"      Type: {'Velocity Field' if is_velocity_field else 'Displacement Field'}")
    else:
        print(f"      Deformation: None (static model)")
    
    # Generate time points
    times = np.linspace(time_range[0], time_range[1], n_frames + n_warmup)
    
    # Warmup
    print(f"\n   Warming up ({n_warmup} frames)...")
    gaussians.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            camera = cameras[i % len(cameras)]
            camera_copy = copy.deepcopy(camera)
            camera_copy.time = float(times[i])
            # Ensure time is float32
            if hasattr(camera_copy, 'time'):
                camera_copy.time = float(camera_copy.time)
            
            render(camera_copy, gaussians, pipeline, background, cam_type=cam_type)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"\n   Benchmarking ({n_frames} frames)...")
    render_times = []
    
    with torch.no_grad():
        for i in tqdm(range(n_warmup, n_warmup + n_frames), desc="Rendering"):
            camera = cameras[i % len(cameras)]
            camera_copy = copy.deepcopy(camera)
            camera_copy.time = float(times[i])
            if hasattr(camera_copy, 'time'):
                camera_copy.time = float(camera_copy.time)
            
            # Time the render
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            render(camera_copy, gaussians, pipeline, background, cam_type=cam_type)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            render_times.append(end_time - start_time)
    
    # Calculate statistics
    render_times = np.array(render_times)
    fps = 1.0 / render_times.mean()
    fps_min = 1.0 / render_times.max()
    fps_max = 1.0 / render_times.min()
    fps_std = np.std(1.0 / render_times)
    
    stats = {
        'fps_mean': fps,
        'fps_min': fps_min,
        'fps_max': fps_max,
        'fps_std': fps_std,
        'render_time_mean': render_times.mean(),
        'render_time_std': render_times.std(),
        'render_time_min': render_times.min(),
        'render_time_max': render_times.max(),
        'n_frames': n_frames,
        'n_warmup': n_warmup,
    }
    
    return fps, stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark rendering FPS for velocity field models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Iteration to load (-1 for latest)")
    parser.add_argument("--source_path", type=str, default=None,
                        help="Path to source dataset (if needed)")
    parser.add_argument("--n_frames", type=int, default=100,
                        help="Number of frames to render for benchmarking (default: 100)")
    parser.add_argument("--n_warmup", type=int, default=10,
                        help="Number of warmup frames (default: 10)")
    parser.add_argument("--camera_source", type=str, default='test',
                        choices=['train', 'test', 'video'],
                        help="Which camera set to use (default: test)")
    parser.add_argument("--time_start", type=float, default=0.0,
                        help="Start time for rendering (default: 0.0)")
    parser.add_argument("--time_end", type=float, default=1.0,
                        help="End time for rendering (default: 1.0)")
    
    args = parser.parse_args()
    
    # Load model and scene
    gaussians, scene, model_args = load_model_and_scene(
        args.model_path, 
        args.iteration,
        args.source_path
    )
    
    # Benchmark
    fps, stats = benchmark_rendering_fps(
        gaussians,
        scene,
        model_args,
        n_frames=args.n_frames,
        n_warmup=args.n_warmup,
        camera_source=args.camera_source,
        time_range=(args.time_start, args.time_end)
    )
    
    # Get Gaussian count
    num_gaussians = gaussians.get_xyz.shape[0]
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Number of Gaussians: {num_gaussians:,}")
    print(f"Average FPS: {fps:.2f}")
    print(f"FPS Range: {stats['fps_min']:.2f} - {stats['fps_max']:.2f}")
    print(f"FPS Std Dev: {stats['fps_std']:.2f}")
    print(f"\nRender Time:")
    print(f"  Mean: {stats['render_time_mean']*1000:.2f} ms")
    print(f"  Std:  {stats['render_time_std']*1000:.2f} ms")
    print(f"  Min:  {stats['render_time_min']*1000:.2f} ms")
    print(f"  Max:  {stats['render_time_max']*1000:.2f} ms")
    print(f"\nFrames rendered: {stats['n_frames']} (warmup: {stats['n_warmup']})")
    print(f"{'='*80}\n")
    
    # Check if real-time capable (30 FPS threshold)
    if fps >= 30.0:
        print("✅ REAL-TIME CAPABLE (≥30 FPS)")
    elif fps >= 15.0:
        print("⚠️  NEAR REAL-TIME (15-30 FPS)")
    else:
        print("❌ NOT REAL-TIME (<15 FPS)")
    print()


if __name__ == "__main__":
    main()

