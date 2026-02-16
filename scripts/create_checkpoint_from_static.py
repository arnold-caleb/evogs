"""
Create a .pth checkpoint from static frame 0 training output.
This allows us to initialize velocity field training from the static model.
"""
import torch
import os
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from scene import ModelParams as SceneModelParams
from arguments import ModelParams, ModelHiddenParams

def create_checkpoint_from_static(model_path, iteration=30000, output_path=None):
    """Load static Gaussians and save as .pth checkpoint."""
    print("=" * 70)
    print(f"Creating checkpoint from static training")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print(f"Iteration: {iteration}")
    
    # Parse arguments to get all needed params
    from argparse import Namespace
    args = Namespace()
    args.sh_degree = 3
    args.source_path = model_path.replace("output/static_frame0_hq/", "data/dynerf/").split("_")[0] + "/cut_roasted_beef"
    
    # Try to find the actual source path
    if not os.path.exists(args.source_path):
        # Guess based on model path
        args.source_path = "data/dynerf/cut_roasted_beef"
    
    print(f"Source path: {args.source_path}")
    
    # Initialize model params
    parser = ArgumentParser()
    hp = ModelHiddenParams(parser)
    
    # Load the scene (this loads the Gaussians)
    from arguments import get_combined_args
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ['script', '--source_path', args.source_path, '--model_path', model_path]
        args_parsed = parser.parse_args()
        model_params = ModelParams(parser).extract(args_parsed)
        model_params.dataset_type = "dynerf_static"  # Use static dataset
        hp_params = ModelHiddenParams(parser).extract(args_parsed)
        
        # Create GaussianModel and load
        gaussians = GaussianModel(model_params.sh_degree, hp_params)
        
        # Create scene to load Gaussians
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
        
        print(f"\n✓ Loaded {gaussians.get_xyz.shape[0]} Gaussians from iteration {iteration}")
        
        # Capture state
        print("\nCapturing model state...")
        model_state = gaussians.capture()
        
        # Save checkpoint
        if output_path is None:
            checkpoint_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
        else:
            checkpoint_path = output_path
            
        checkpoint_data = (model_state, iteration)
        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"\n✓ Saved checkpoint to: {checkpoint_path}")
        print(f"  File size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB")
        
        return checkpoint_path
        
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to static training output")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Iteration to load")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (default: model_path/chkpnt{N}.pth)")
    
    args = parser.parse_args()
    
    checkpoint_path = create_checkpoint_from_static(
        args.model_path,
        args.iteration,
        args.output
    )
    
    print("\n" + "=" * 70)
    print("✅ CHECKPOINT CREATED!")
    print("=" * 70)
    print(f"\nYou can now use this checkpoint in velocity training:")
    print(f"  --start_checkpoint {checkpoint_path}")

