"""
Train static 3D Gaussian Splatting on Frame 0 (all viewpoints).
This is Step 1 of our operator learning pipeline.

Output: Optimized Gaussians for the static scene at t=0
"""
import os
import sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from tqdm import tqdm

# Tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, canonical_checkpoint=None):
    """
    Training function for static 3D Gaussian Splatting (no deformation).
    
    Args:
        canonical_checkpoint: If provided, load Gaussians from this checkpoint (for anchor training)
                            This ensures same Gaussian count and correspondence across frames
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # Initialize Gaussians
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    
    # MULTI-ANCHOR: Load from canonical checkpoint if provided
    if canonical_checkpoint:
        print(f"\n[MULTI-ANCHOR] Loading canonical Gaussians from: {canonical_checkpoint}")
        (canonical_params, _) = torch.load(canonical_checkpoint)
        
        # Create scene WITHOUT loading iteration (we'll manually restore)
        scene = Scene(dataset, gaussians, load_iteration=None)
        
        # Restore canonical Gaussians
        gaussians.restore(canonical_params, opt)
        gaussians.training_setup(opt)
        
        n_gaussians = gaussians.get_xyz.shape[0]
        print(f"[MULTI-ANCHOR] Loaded {n_gaussians} Gaussians from canonical")
        print(f"[MULTI-ANCHOR] Will optimize positions to fit frame {dataset.frame_idx}")
        print(f"[MULTI-ANCHOR] Densification SHOULD be disabled in config!")
    else:
        # Standard training: initialize from scene
        scene = Scene(dataset, gaussians, load_iteration=checkpoint)
        gaussians.training_setup(opt)
        
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # Get training cameras once
    train_cams = scene.getTrainCameras()
    viewpoint_stack = [i for i in train_cams]
    import copy
    temp_list = copy.deepcopy(viewpoint_stack)
    
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = temp_list.copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # Render (use "coarse" stage to skip deformation network for static scene)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage="coarse", cam_type=scene.dataset_type)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        loss.backward()
        
        # Extract view-space gradients AFTER backward (critical!)
        # This captures how each Gaussian's screen position affects the loss
        viewspace_point_tensor_grad = viewspace_point_tensor.grad if viewspace_point_tensor.grad is not None else torch.zeros_like(viewspace_point_tensor)
        
        iter_end.record()
        
        # Densification thresholds (define early to avoid UnboundLocalError)
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        densify_threshold = opt.densify_grad_threshold_coarse
        opacity_threshold = opt.opacity_threshold_coarse
        
        # CRITICAL: Accumulate gradients BEFORE entering no_grad() block!
        if iteration < opt.densify_until_iter:
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )
            # Accumulate view-space gradients (using .grad attribute after backward)
            gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                          testing_iterations, scene, render, (pipe, background))
            
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
            
            # Densification
            if iteration < opt.densify_until_iter:
                
                # Densify
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000:
                    # Debug: Check gradients before densification
                    if iteration % 500 == 0:  # Print more frequently to monitor
                        grads = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-7)  # Avoid div by zero
                        grads[grads.isnan()] = 0.0
                        num_above_thresh = (torch.norm(grads, dim=-1) >= densify_threshold).sum().item()
                        print(f"\n[DENSIFY DEBUG iter {iteration}] Gaussians: {gaussians.get_xyz.shape[0]}, Grads > thresh: {num_above_thresh}, Thresh: {densify_threshold}, Avg grad: {torch.norm(grads, dim=-1).mean().item():.6f}")
                    
                    gaussians.densify(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        5,
                        5,
                        scene.model_path,
                        iteration,
                        "static"
                    )
                    
                    # Debug: Print Gaussian count after densification
                    if iteration % 500 == 0:
                        print(f"[DENSIFY DEBUG] After densification: {gaussians.get_xyz.shape[0]} Gaussians")
                
                # Prune
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0] > 200000:
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + f"/chkpnt{iteration}.pth"
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        frame_idx = getattr(args, 'frame_idx', 0)
        unique_str = f"static_frame{frame_idx}_{args.data_device}"
        args.model_path = os.path.join("./output/", unique_str)
    
    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # Test and log at specified iterations
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # Sample cameras using list comprehension (not slicing)
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        validation_configs = [
            {'name': 'test', 'cameras': [test_cams[idx % len(test_cams)] for idx in range(min(5, len(test_cams)))]},
            {'name': 'train', 'cameras': [train_cams[idx % len(train_cams)] for idx in range(0, min(len(train_cams), 20), 5)]}
        ]
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += (20 * torch.log10(1.0 / torch.sqrt(torch.nn.functional.mse_loss(image, gt_image)))).double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script for static frame 0")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[3000, 7000, 15000, 30000])
    parser.add_argument('--save_iterations', nargs="+", type=int, default=[3000, 7000, 15000, 30000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs="+", type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--canonical_checkpoint', type=str, default=None, 
                        help='Path to canonical checkpoint for anchor training (ensures Gaussian correspondence)')
    parser.add_argument('--configs', type=str, default="")
    # Note: --frame_idx is now defined in ModelParams class
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # Load config file if provided
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print("Optimizing " + args.model_path)
    
    # Debug: Verify frame_idx is set
    frame_idx = getattr(args, 'frame_idx', 0)
    dataset_type = getattr(args, 'dataset_type', 'dynerf')
    print(f"[DEBUG] Training with dataset_type={dataset_type}, frame_idx={frame_idx}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.canonical_checkpoint
    )
    
    # All done
    frame_idx = getattr(args, 'frame_idx', 0)
    print(f"\n[COMPLETE] Static frame {frame_idx} training finished!")
    print(f"Output: {args.model_path}")
    print(f"\nNext step: Use as anchor for multi-anchor velocity field training")

