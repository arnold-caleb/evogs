"""
Simple script to create a .pth checkpoint from static frame 0 training.
"""
import torch
import os
import sys

def create_checkpoint_from_static(model_path, iteration=30000):
    """Load static Gaussians and save as .pth checkpoint."""
    print(f"Loading static model from: {model_path}")
    print(f"Iteration: {iteration}")
    
    # Path to the saved Gaussian state
    point_cloud_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    
    # The Scene class loads all the necessary state
    from scene import Scene, GaussianModel
    from arguments import ModelParams, ModelHiddenParams
    from argparse import ArgumentParser
    
    # Create argument parser to extract params
    parser = ArgumentParser()
    hp = ModelHiddenParams(parser)
    
    # Hardcode source path for cut_roasted_beef
    source_path = "data/dynerf/cut_roasted_beef"
    
    # Create dummy args
    class Args:
        source_path = source_path
        model_path = model_path
        images = "images"
        sh_degree = 3
        eval = False
    
    args = Args()
    
    # Create model params
    model_params = type('obj', (object,), {})()
    model_params.source_path = source_path
    model_params.model_path = model_path
    model_params.images = "images"
    model_params.sh_degree = 3
    model_params.white_background = False
    model_params.dataset_type = "dynerf_static"
    
    # Create hidden params
    hp_params = type('obj', (object,), {})()
    hp_params.use_velocity_field = False
    hp_params.integrate_color_opacity = False
    hp_params.static_mlp = False
    hp_params.time_smoothness_weight = 0.0
    hp_params.multires = [1, 2]
    
    # Create and load Gaussians
    gaussians = GaussianModel(model_params.sh_degree, hp_params)
    
    print(f"\nLoading scene...")
    scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    
    print(f"✓ Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    print(f"  SH degree: {gaussians.active_sh_degree}")
    
    # Capture model state
    print("\nCapturing model state...")
    model_state = gaussians.capture()
    
    # Save checkpoint
    checkpoint_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    checkpoint_data = (model_state, iteration)
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"\n✓ Saved checkpoint to: {checkpoint_path}")
    print(f"  File size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB")
    
    return checkpoint_path


if __name__ == "__main__":
    model_path = "output/static_frame0_hq/cut_roasted_beef_20251027_191723"
    checkpoint_path = create_checkpoint_from_static(model_path, 30000)
    
    print("\n" + "=" * 70)
    print("✅ CHECKPOINT CREATED!")
    print("=" * 70)

