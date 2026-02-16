"""
EvoGS: 4D Gaussian Splatting as a Learned Dynamical System
Code is adapted from the original 4D Gaussian Splatting codebase: https://github.com/hustvl/4DGaussians/blob/master/train.py

Key features:
- Multi-stage training: coarse (optional warm start) → fine (dynamics learning)
- Sparse temporal supervision: train on subset of frames to test generalization
- Multi-anchor constraints: reduce integration drift with fixed waypoints
- Future reconstruction: train on first N% of frames, evaluate on rest
- Adaptive densification: refine Gaussians in high-motion regions

Training flow:
1. Load static checkpoint (optional) or initialize from scratch
2. Run coarse stage for geometry warm-start (optional, usually skipped for velocity)
3. Load anchor Gaussians at key timesteps (optional, for multi-anchor training)
4. Run fine stage with velocity field and full dynamics
5. Adaptive densification based on temporal gradients
6. Periodic evaluation and checkpointing
"""

import copy
import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from time import time
from bisect import bisect_left

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.sparse_temporal_sampler import SparseTemporalSampler
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loader_utils import FineSampler, get_stamp_list
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
try:
    from lpipsPyTorch import lpips as lpips_fn
except ImportError:
    lpips_fn = None
from utils.scene_utils import render_training_image
from utils.timer import Timer
from scene.regulation import compute_trajectory_coherence, compute_velocity_regularizations
from scene.adaptive_coherence import AdaptiveCoherencePredictor, compute_adaptive_coherence_loss

to8b = lambda x : (255*np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer, anchor_gaussians=None):
    first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    frame_times = getattr(scene, 'train_times', None)
    num_time_steps = getattr(scene, 'num_time_steps', 0)
    if num_time_steps is None or num_time_steps == 0:
        num_time_steps = len(frame_times) if frame_times else 300
    
    # Compute unique temporal frames (frame_times may have duplicates across cameras)
    # e.g., 19 cameras × 300 frames = 5700 entries, but only 300 unique timestamps
    unique_frame_times = sorted(set(frame_times)) if frame_times else []
    num_unique_frames = len(unique_frame_times) if unique_frame_times else num_time_steps

    def frame_index_from_time(time_value: float) -> int:
        """
        Map a timestamp to the nearest unique temporal frame index (0 to num_unique_frames-1).
        Uses deduplicated timestamps so the result is a temporal index, not a dataset index.
        """
        if unique_frame_times:
            insert_idx = bisect_left(unique_frame_times, time_value)
            if insert_idx >= len(unique_frame_times):
                return len(unique_frame_times) - 1
            if unique_frame_times[insert_idx] == time_value or insert_idx == 0:
                return insert_idx
            prev_idx = insert_idx - 1
            prev_time = unique_frame_times[prev_idx]
            next_time = unique_frame_times[insert_idx]
            return insert_idx if abs(next_time - time_value) < abs(time_value - prev_time) else prev_idx

        if num_unique_frames <= 1:
            return 0

        max_frame = num_unique_frames - 1

        # DyNeRF datasets often store time directly as an integer frame index.
        if time_value >= -1e-6 and time_value <= max_frame + 1 and time_value > 1.5:
            return int(round(time_value))

        # Otherwise assume normalized 0-1 range.
        return int(round(max(0.0, min(1.0, time_value)) * max_frame))
    
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    # ============================================================================
    # FILTER TRAINING CAMERAS: Only use first 75% of frames for training
    # (Rendering will still use all frames via test_cams/video_cams)
    # ============================================================================
    train_time_max = getattr(hyper, 'train_time_max', 1.0)

    # this takes longer to load. 
    if train_time_max < 1.0:
        print(f"\n[FILTERING] Training only on frames with time <= {train_time_max} (first {int(train_time_max*100)}%)")
        filtered_train_cams = []
        for cam in train_cams:
            time_value = getattr(cam, 'time', 0.0)
            if time_value <= train_time_max:
                filtered_train_cams.append(cam)
        train_cams = filtered_train_cams
        print(f"  Filtered from {len(scene.getTrainCameras())} to {len(train_cams)} training cameras")
        print(f"  (Test/video cameras remain unfiltered for full-sequence rendering)")

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")

    using_sparse_temporal_sampler = False

    if opt.dataloader:
        viewpoint_stack = train_cams  # Use filtered training cameras
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=16, collate_fn=list)
            random_loader = False
        else:
            sampler = None
            if (hasattr(hyper, 'sparse_supervision') and hyper.sparse_supervision and getattr(hyper, 'supervised_frame_stride', 1) > 1):
                stride = getattr(hyper, 'supervised_frame_stride', 1)
                offset = getattr(hyper, 'supervised_frame_offset', 0)
                base_dataset = getattr(viewpoint_stack, 'dataset', None)
                num_cameras = getattr(base_dataset, 'cam_number', None)
                num_frames = getattr(base_dataset, 'time_number', None)
                if num_cameras is not None and num_frames is not None:
                    sampler = SparseTemporalSampler(viewpoint_stack, num_cameras=num_cameras, num_frames=num_frames, temporal_stride=stride, frame_offset=offset)
                    using_sparse_temporal_sampler = True
                else:
                    print("[SPARSE WARNING] Could not infer camera/frame counts for sampler; falling back to dense sampling.") 
            if sampler is not None:
                print(f"[SPARSE] Using sparse temporal sampler with stride={stride} and offset={offset}")
                # One-time dump: show ALL supervised temporal frame indices for verification
                supervised_temporal = sorted(set(
                    idx % sampler.num_frames for idx in sampler.valid_indices
                ))
                skipped_temporal = sorted(set(range(sampler.num_frames)) - set(supervised_temporal))
                print(f"[SPARSE] Supervised temporal frames ({len(supervised_temporal)}/{sampler.num_frames}): {supervised_temporal}")
                print(f"[SPARSE] Skipped temporal frames   ({len(skipped_temporal)}/{sampler.num_frames}): {skipped_temporal}")
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=16, collate_fn=list)
                random_loader = False
            else:
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    # Skip coarse stage block (not needed for velocity field training)
        load_in_memory = False 

    count = 0

    # ============================================================================
    # PER-ITERATION TRAINING LOOP
    # ============================================================================
    for iteration in range(first_iter, final_iter+1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 iterations we increase the levels of SH up to a maximum degree
        # Enable for displacement field (trains from scratch), disable for velocity (starts from checkpoint)
        if iteration % 1000 == 0:
            # Only increase if not using multi-anchor (which starts from checkpoint with full SH)
            if not (hasattr(hyper, 'use_multi_anchor') and hyper.use_multi_anchor):
                gaussians.oneupSHdegree()


        # ============================================================================
        # STEP 1: Sample viewpoints from data loader
        # ============================================================================
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                if using_sparse_temporal_sampler:
                    loader = iter(viewpoint_stack_loader)
                    continue
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size, shuffle=True, num_workers=32, collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
            
        # ============================================================================
        # STEP 2: Render batch of views and accumulate outputs
        # ============================================================================
        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        # Cache integrated positions for efficient regularization
        integrated_xyz_cache = None

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))

            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
            # Cache integrated positions from the FIRST rendering in this batch
            # ASSUMPTION: All viewpoints in batch have same timestep (sampled together)
            # This avoids double integration in regularization
            if (integrated_xyz_cache is None and stage == "fine" and 
                hasattr(gaussians, '_deformation') and 
                hasattr(gaussians._deformation, 'velocity_field') and
                hasattr(gaussians._deformation.velocity_field, 'last_integrated_xyz')):
                integrated_xyz_cache = gaussians._deformation.velocity_field.last_integrated_xyz
        
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
        # ============================================================================
        # FUTURE RECONSTRUCTION: No supervision beyond train_time_max
        # ============================================================================
        supervised = True
        time_value = viewpoint_cams[0].time if len(viewpoint_cams) > 0 else 0.0
        
        if hasattr(hyper, 'future_reconstruction') and hyper.future_reconstruction and stage in ("fine", "coarse"):
            train_time_max = getattr(hyper, 'train_time_max', 1.0)
            if time_value > train_time_max:
                supervised = False
        
        # ============================================================================
        # SPARSE SUPERVISION: Only compute photometric loss for supervised frames
        # ============================================================================
        per_camera_frames = []
        per_camera_supervised = []

        if hasattr(hyper, 'sparse_supervision') and hyper.sparse_supervision and stage in ("fine", "coarse"):
            stride = getattr(hyper, 'supervised_frame_stride', 1)
            offset = getattr(hyper, 'supervised_frame_offset', 0)
            if stride <= 0:
                stride = 1
                offset = 0
            anchor_times = getattr(hyper, 'anchor_checkpoints', {}).keys()

            for cam in viewpoint_cams:
                time_value_cam = getattr(cam, 'time', 0.0)
                frame_idx_cam = frame_index_from_time(time_value_cam)
                is_anchor = any(abs(time_value_cam - anchor_t) < 0.01 for anchor_t in anchor_times)
                is_supervised_frame = (frame_idx_cam % stride) == offset
                per_camera_frames.append(frame_idx_cam)
                per_camera_supervised.append(is_anchor or is_supervised_frame)

            if supervised:
                supervised = any(per_camera_supervised)
        else:
            for cam in viewpoint_cams:
                time_value_cam = getattr(cam, 'time', 0.0)
                frame_idx_cam = frame_index_from_time(time_value_cam)
                per_camera_frames.append(frame_idx_cam)
                per_camera_supervised.append(True)

        # Compute photometric loss only for supervised frames
        supervision_mask = torch.tensor(per_camera_supervised, device=image_tensor.device, dtype=torch.bool) if per_camera_supervised else None
        has_supervised = supervision_mask is not None and supervision_mask.any()

        if supervised and has_supervised:
            supervised_images = image_tensor[supervision_mask]
            supervised_gt_images = gt_image_tensor[supervision_mask]
            Ll1 = l1_loss(supervised_images, supervised_gt_images[:,:3,:,:])
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            loss = Ll1
        else:
            with torch.no_grad():
                Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
                psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            loss = torch.tensor(0.0, device=image_tensor.device, requires_grad=True)

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss = loss + tv_loss

        # ============================================================================
        # ANCHOR LOSS: Compare evolved Gaussians at anchor times with frozen anchors
        # ============================================================================
        # The velocity field integrates trainable Gaussians from t=0 to t_target.
        # Anchor loss ensures that when integrated to an anchor time (e.g., t=0.25),
        # the evolved 3D positions match the pre-optimized static Gaussians at that time.
        # This provides direct 3D supervision along the trajectory, complementing
        # the indirect photometric loss from rendering.
        anchor_loss_value = torch.tensor(0.0, device=loss.device)
        lambda_anchor = getattr(hyper, 'lambda_anchor', 0.0)
        anchor_loss_interval = getattr(hyper, 'anchor_loss_interval', 10)

        if (stage == "fine" and anchor_gaussians is not None and len(anchor_gaussians) > 0
                and lambda_anchor > 0 and anchor_loss_interval > 0
                and iteration % anchor_loss_interval == 0):

            anchor_times_list = sorted(anchor_gaussians.keys())
            # Cycle through anchors so each gets equal attention
            anchor_idx = (iteration // anchor_loss_interval) % len(anchor_times_list)
            anchor_t = anchor_times_list[anchor_idx]
            anchor_gauss = anchor_gaussians[anchor_t]

            N_train = gaussians._xyz.shape[0]
            N_anchor = anchor_gauss._xyz.shape[0]

            if N_train == N_anchor:
                # Integrate trainable Gaussians from t=0 to anchor time
                t_anchor_tensor = torch.ones(N_train, 1, device=gaussians._xyz.device) * anchor_t

                evolved_xyz, _, _, _, _ = gaussians._deformation(
                    gaussians._xyz, gaussians._scaling, gaussians._rotation,
                    gaussians._opacity, gaussians.get_features, t_anchor_tensor
                )

                # L2 loss between evolved positions and frozen anchor positions
                anchor_pos_loss = torch.mean((evolved_xyz - anchor_gauss._xyz) ** 2)
                anchor_loss_value = lambda_anchor * anchor_pos_loss
                loss = loss + anchor_loss_value
            else:
                # N mismatch: skip anchor loss (happens when --start_checkpoint not used)
                if iteration % 1000 == 0:
                    print(f"[ANCHOR] Skipping anchor loss: N_train={N_train} != N_anchor={N_anchor}")
                    print(f"  Hint: Use --start_checkpoint to initialize from same static checkpoint as anchors")

        # Log loss and frame info periodically
        if iteration % 100 == 0:
            frame_idx = frame_index_from_time(time_value)
            sup_tag = "supervised" if supervised else "unsupervised"
            anchor_str = f"  anchor={anchor_loss_value.item():.6f}" if anchor_loss_value.item() > 0 else ""
            print(f"[Iter {iteration}] loss={loss.item():.6f}  L1={Ll1.item():.6f}  PSNR={psnr_:.2f}  frame={frame_idx}  ({sup_tag}){anchor_str}")

        # ============================================================================
        # STEP 5: Backward pass and gradient accumulation
        # ============================================================================
        # Update current_iteration for velocity field training
        if (stage == "fine" and hasattr(gaussians, '_deformation') and 
            hasattr(gaussians._deformation, 'velocity_field')):
            gaussians._deformation.velocity_field.current_iteration = iteration
        
        loss.backward()

        if torch.isnan(loss).any():
            print(f"[ERROR] Loss is NaN at iteration {iteration}. Stopping training.")
            print(f"  Last frame={frame_index_from_time(time_value)}, t={time_value:.4f}")
            print(f"  Saving emergency checkpoint before exit...")
            torch.save((gaussians.capture(), iteration), 
                       scene.model_path + f"/chkpnt_nan_{stage}_{iteration}.pth")
            sys.exit(1)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)

        for idx in range(0, len(viewspace_point_tensor_list)):
            grad = viewspace_point_tensor_list[idx].grad
            if grad is not None:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + grad
        iter_end.record()

        # ============================================================================
        # STEP 6: Logging, evaluation, and checkpointing
        # ============================================================================
        with torch.no_grad():
            # Log ACTUAL L1 error (not training loss), so we see degradation on future frames!
            ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "psnr": f"{psnr_:.{2}f}", "point":f"{total_point}"})
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type, hyper)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) or (iteration < 3000 and iteration % 50 == 49) or (iteration < 60000 and iteration %  100 == 99) :
                        render_training_image(scene, gaussians, [test_cams[iteration % len(test_cams)]], render, pipe, background, stage + "test", iteration, timer.get_elapsed_time(), scene.dataset_type)
                        render_training_image(scene, gaussians, [train_cams[iteration % len(train_cams)]], render, pipe, background, stage + "train", iteration, timer.get_elapsed_time(), scene.dataset_type)

            timer.start()

            # ============================================================================
            # STEP 7: Adaptive densification and pruning
            # ============================================================================
            # Densification: Adaptive Gaussian refinement based on temporal gradients
            # Note: We're always in "fine" stage (no coarse training, start from static checkpoint)
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning decisions
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Accumulate gradients from temporal photometric loss + regularization
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                
                # Thresholds: Higher gradient = more motion = needs more Gaussians
                # densify_threshold: Split/clone Gaussians with gradient > threshold (default: 0.0004)
                # opacity_threshold: Prune Gaussians with opacity < threshold (default: 0.005)
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (opt.densify_until_iter)
                opacity_threshold = opt.opacity_threshold_fine_init - iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (opt.densify_until_iter)  
                
                # Densify: Split/clone in high-gradient regions (hands moving!)
                # Start: ~220K Gaussians → Add in high-motion regions
                # Conservative limit with adaptive coherence: 280K (was 360K)
                max_gaussians = 280000 if getattr(opt, 'use_adaptive_coherence', False) else 360000
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < max_gaussians:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                    print(f"[Iter {iteration}] Densified: {gaussians.get_xyz.shape[0]} Gaussians (threshold={densify_threshold:.6f})")

                # Prune: Remove low-opacity/poorly-optimized Gaussians
                # Condition: Only prune if we have MORE than initial count (allow growth first)
                # Start: 220K → Allow to grow → Then prune if > 250K
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0] > 250000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    print(f"[Iter {iteration}] Pruned: {gaussians.get_xyz.shape[0]} Gaussians")
                    
                # Optional: Add random points (disabled by config)
                max_gaussians = 280000 if getattr(opt, 'use_adaptive_coherence', False) else 360000
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < max_gaussians and opt.add_point:
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)

                # Periodically reset opacity to remove accumulated noise
                if iteration % opt.opacity_reset_interval == 0:
                    print(f"[Iter {iteration}] Reset opacity")
                    gaussians.reset_opacity()
            
            # STAGED TRAINING: DISABLED - Focus on velocity field learning throughout
            # (Refinement can be done in a separate run after dynamics are learned)
            # refinement_stage_start = opt.iterations // 2  # Halfway point
            # if iteration == refinement_stage_start:
            #     print(f"\n[STAGE TRANSITION] Switching to refinement...")
            #     # Adjust LRs here
            # pass  # No staged training for now

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, load_iteration=None):
    """
    Main training orchestrator: Sets up scene, loads checkpoints, and runs training stages.
    
    Training pipeline:
        1. Load checkpoint (if provided) - typically a static frame 0 checkpoint
        2. Run coarse stage (optional) - geometry warm-start
        3. Load anchor Gaussians (optional) - for multi-anchor training
        4. Run fine stage - velocity field training with full dynamics
        5. Final evaluation - compute metrics on seen vs unseen frames
    """
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, load_coarse=None)
    timer.start()

    # Load checkpoint BEFORE scene_reconstruction
    if checkpoint and os.path.exists(checkpoint):
        print(f"\nLoading checkpoint BEFORE fine stage: {checkpoint}")
        gaussians.training_setup(opt)  # Initialize optimizer first
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Loaded checkpoint from iteration {first_iter}")
        
        # CRITICAL: For velocity field with multi-anchor, reset first_iter to 0
        # We're loading Gaussians from a static checkpoint but starting fresh velocity training
        if hasattr(hyper, 'use_velocity_field') and hyper.use_velocity_field and \
           hasattr(hyper, 'use_multi_anchor') and hyper.use_multi_anchor:
            print(f"   Resetting iteration counter to 0 for fresh velocity field training")
            first_iter = 0
    else:
        # Fresh training: Initialize optimizer
        print("\nInitializing optimizer for fresh training...")
        gaussians.training_setup(opt)

    # ============================================================================
    # COARSE STAGE: warm start geometry before fine-stage dynamics/regularization
    # ============================================================================
    coarse_iterations = getattr(opt, 'coarse_iterations', 0)
    if coarse_iterations and coarse_iterations > 0:
        print(f"\n=== Starting coarse stage ({coarse_iterations} iterations) ===")
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, None, debug_from,
                             gaussians, scene, "coarse", tb_writer, coarse_iterations, timer,
                             anchor_gaussians=None)
        torch.cuda.empty_cache()
        timer.start()

    # ============================================================================
    # MULTI-ANCHOR TRAINING: Load fixed waypoint Gaussians at key timesteps
    # ============================================================================
    anchor_gaussians = {}
    use_multi_anchor = getattr(hyper, 'use_multi_anchor', False)
    use_anchor_constraints = getattr(hyper, 'use_anchor_constraints', False)
    if use_multi_anchor or use_anchor_constraints:
        print("\n=== LOADING ANCHOR GAUSSIANS ===")
        anchor_checkpoints = getattr(hyper, 'anchor_checkpoints', {})
        
        for anchor_time, checkpoint_path in anchor_checkpoints.items():
            if os.path.exists(checkpoint_path):
                print(f"Loading anchor at t={anchor_time}: {checkpoint_path}")
                
                # Extract Gaussian parameters directly from checkpoint tuple.
                # Anchors only need frozen positions/attributes — no deformation
                # or velocity network required. This avoids state_dict mismatch
                # warnings when anchor checkpoints use a different network type.
                (model_params_tuple, _) = torch.load(checkpoint_path, map_location="cpu")
                
                class AnchorGaussians:
                    """Lightweight container for frozen anchor Gaussians."""
                    pass
                
                anchor_gauss = AnchorGaussians()
                # Checkpoint tuple layout: (active_sh_degree, _xyz, deform_state,
                #   _deformation_table, _features_dc, _features_rest, _scaling,
                #   _rotation, _opacity, max_radii2D, xyz_gradient_accum, denom,
                #   opt_dict, spatial_lr_scale, [clone_birth_iter])
                anchor_gauss._xyz = model_params_tuple[1].detach().cuda()
                anchor_gauss._features_dc = model_params_tuple[4].detach().cuda()
                anchor_gauss._features_rest = model_params_tuple[5].detach().cuda()
                anchor_gauss._scaling = model_params_tuple[6].detach().cuda()
                anchor_gauss._rotation = model_params_tuple[7].detach().cuda()
                anchor_gauss._opacity = model_params_tuple[8].detach().cuda()
                anchor_gauss.idx_map = None
                
                anchor_gaussians[anchor_time] = anchor_gauss
                print(f"  ✓ Loaded {anchor_gauss._xyz.shape[0]} anchor Gaussians at t={anchor_time}")
            else:
                print(f"  ✗ Warning: Anchor checkpoint not found: {checkpoint_path}")
        
        print(f"Total anchors loaded: {len(anchor_gaussians)}")
        _lambda_anchor = getattr(hyper, 'lambda_anchor', 0.0)
        _anchor_interval = getattr(hyper, 'anchor_loss_interval', 10)
        if _lambda_anchor > 0 and _anchor_interval > 0:
            print(f"Anchor loss ENABLED: lambda={_lambda_anchor}, interval={_anchor_interval}")
            print(f"  Every {_anchor_interval} iters, integrates to a random anchor time")
            print(f"  and penalizes L2 distance to frozen anchor positions")
        else:
            print(f"Anchor loss DISABLED (lambda={_lambda_anchor}, interval={_anchor_interval})")
        print("")

    if use_multi_anchor:
        gaussians.anchor_gaussians = anchor_gaussians
    else:
        gaussians.anchor_gaussians = None

    # Optional: derive dynamic mask from anchor displacement to limit velocity field scope
    
    # Start from fine stage that will be trained with velocity field
    # Pass checkpoint=None to skip loading again in scene_reconstruction
    # Pass anchor_gaussians to scene_reconstruction
    anchor_dict_for_loss = anchor_gaussians if (use_multi_anchor or use_anchor_constraints) else None
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, None, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations, timer,
                         anchor_dict_for_loss)
    
    # Final evaluation: Compute seen vs unseen metrics at end of training
    train_time_max = getattr(hyper, 'train_time_max', 1.0)
    if train_time_max < 1.0:
        print("\n" + "="*80)
        print("[FINAL EVALUATION] Computing final metrics for seen vs unseen frames")
        print("="*80)
        from gaussian_renderer import render
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        evaluate_seen_unseen_metrics(scene, render, [pipe, background], "fine", scene.dataset_type, train_time_max, hyper)

def prepare_output_and_logger(expname):    
    """Setup output directory and tensorboard logger"""
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def evaluate_seen_unseen_metrics(scene, renderFunc, renderArgs, stage, dataset_type, train_time_max, hyper):
    """
    Evaluate PSNR, SSIM, and LPIPS separately for:
    - Seen frames (time <= train_time_max, i.e., first 75%)
    - Unseen frames (time > train_time_max, i.e., last 25%)
    
    NOTE: Evaluates on TRAINING cameras only (matching HexPlane evaluation protocol)
    """
    print("\n" + "="*80)
    print(f"[EVALUATION] Computing metrics for seen (≤{train_time_max:.2f}) vs unseen (>{train_time_max:.2f}) frames")
    print("[EVALUATION] Using TRAINING cameras only (matching HexPlane protocol)")
    print("="*80)
    
    # Get training cameras for evaluation (matching HexPlane's approach)
    train_cams = scene.getTrainCameras()
    
    # Evaluate on ALL training cameras, split by time
    # This matches HexPlane's "TRAINING FRAMES [0-36]" evaluation
    eval_cams = []
    for cam in train_cams:
        # Only add if it has original_image (ground truth)
        if hasattr(cam, 'original_image') and cam.original_image is not None:
            eval_cams.append(cam)
    
    # Debug: Check time distribution
    eval_times = [getattr(cam, 'time', 0.0) for cam in eval_cams]
    
    print(f"[DEBUG] Total train cameras: {len(train_cams)}")
    print(f"[DEBUG] Total eval cameras (train only): {len(eval_cams)}")
    if len(eval_times) > 0:
        print(f"[DEBUG] Eval time range: [{min(eval_times):.3f}, {max(eval_times):.3f}]")
        print(f"[DEBUG] Eval times <= {train_time_max:.2f}: {sum(1 for t in eval_times if t <= train_time_max)}")
        print(f"[DEBUG] Eval times > {train_time_max:.2f}: {sum(1 for t in eval_times if t > train_time_max)}")
    
    if len(eval_cams) == 0:
        print("[WARNING] No cameras available for evaluation")
        return
    
    print(f"[INFO] Evaluating on {len(eval_cams)} training cameras (split by time)")
    
    # LPIPS is a function from lpipsPyTorch, no initialization needed
    use_lpips = lpips_fn is not None
    if not use_lpips:
        print("[WARNING] LPIPS not available, skipping LPIPS metrics")
    
    seen_psnrs = []
    seen_ssims = []
    seen_lpipss = []
    unseen_psnrs = []
    unseen_ssims = []
    unseen_lpipss = []
    
    torch.cuda.empty_cache()
    
    # Debug: Count frames by category before rendering
    seen_count_debug = 0
    unseen_count_debug = 0
    time_values_debug = []
    for cam in eval_cams:
        time_val = getattr(cam, 'time', 0.0)
        time_values_debug.append(time_val)
        if time_val <= train_time_max:
            seen_count_debug += 1
        else:
            unseen_count_debug += 1
    print(f"[DEBUG] Expected split: {seen_count_debug} seen, {unseen_count_debug} unseen")
    print(f"[DEBUG] Time range in eval cameras: [{min(time_values_debug):.3f}, {max(time_values_debug):.3f}]")
    
    for idx, viewpoint in enumerate(eval_cams):
        time_value = getattr(viewpoint, 'time', 0.0)
        is_seen = time_value <= train_time_max
        
        # Render
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
        # For video cameras, original_image might not be available - skip if missing
        if not hasattr(viewpoint, 'original_image') or viewpoint.original_image is None:
            if idx < 5:  # Only warn for first few
                print(f"[WARNING] Camera {idx} has no original_image, skipping")
            continue
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        
        # Ensure images are in [B, C, H, W] format for metrics
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)
        
        # Compute metrics
        psnr_val = psnr(image, gt_image).mean().double().item()
        ssim_val = ssim(image, gt_image).item()
        
        if use_lpips:
            try:
                # lpipsPyTorch.lpips is a function: lpips(img1, img2, net_type='vgg')
                lpips_val = lpips_fn(image, gt_image, net_type='vgg').item()
            except Exception as e:
                print(f"[WARNING] LPIPS computation failed: {e}")
                lpips_val = None
        else:
            lpips_val = None
        
        # Categorize by seen/unseen
        if is_seen:
            seen_psnrs.append(psnr_val)
            seen_ssims.append(ssim_val)
            if lpips_val is not None:
                seen_lpipss.append(lpips_val)
        else:
            unseen_psnrs.append(psnr_val)
            unseen_ssims.append(ssim_val)
            if lpips_val is not None:
                unseen_lpipss.append(lpips_val)
    
    # Print results
    print("\n" + "-"*80)
    print("SEEN FRAMES (time <= {:.2f}): {} frames".format(train_time_max, len(seen_psnrs)))
    print("-"*80)
    if len(seen_psnrs) > 0:
        print("  PSNR : {:>12.7f}".format(torch.tensor(seen_psnrs).mean().item()))
        print("  SSIM : {:>12.7f}".format(torch.tensor(seen_ssims).mean().item()))
        if len(seen_lpipss) > 0:
            print("  LPIPS: {:>12.7f}".format(torch.tensor(seen_lpipss).mean().item()))
    else:
        print("  No seen frames found!")
    
    print("\n" + "-"*80)
    print("UNSEEN FRAMES (time > {:.2f}): {} frames".format(train_time_max, len(unseen_psnrs)))
    print("-"*80)
    if len(unseen_psnrs) > 0:
        print("  PSNR : {:>12.7f}".format(torch.tensor(unseen_psnrs).mean().item()))
        print("  SSIM : {:>12.7f}".format(torch.tensor(unseen_ssims).mean().item()))
        if len(unseen_lpipss) > 0:
            print("  LPIPS: {:>12.7f}".format(torch.tensor(unseen_lpipss).mean().item()))
    else:
        print("  No unseen frames found!")
    
    print("="*80 + "\n")
    
    torch.cuda.empty_cache()
    
    return {
        'seen': {
            'psnr': torch.tensor(seen_psnrs).mean().item() if seen_psnrs else 0.0,
            'ssim': torch.tensor(seen_ssims).mean().item() if seen_ssims else 0.0,
            'lpips': torch.tensor(seen_lpipss).mean().item() if seen_lpipss else 0.0,
            'count': len(seen_psnrs)
        },
        'unseen': {
            'psnr': torch.tensor(unseen_psnrs).mean().item() if unseen_psnrs else 0.0,
            'ssim': torch.tensor(unseen_ssims).mean().item() if unseen_ssims else 0.0,
            'lpips': torch.tensor(unseen_lpipss).mean().item() if unseen_lpipss else 0.0,
            'count': len(unseen_psnrs)
        }
    }

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type, hyper=None):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        # Evaluate seen vs unseen frames at final iteration
        if iteration == testing_iterations[-1] and hyper is not None:
            train_time_max = getattr(hyper, 'train_time_max', 1.0)
            if train_time_max < 1.0:
                metrics = evaluate_seen_unseen_metrics(scene, renderFunc, renderArgs, stage, dataset_type, train_time_max, hyper)
                if tb_writer and metrics:
                    # Log to tensorboard
                    tb_writer.add_scalar(f'{stage}/metrics_seen/psnr', metrics['seen']['psnr'], iteration)
                    tb_writer.add_scalar(f'{stage}/metrics_seen/ssim', metrics['seen']['ssim'], iteration)
                    tb_writer.add_scalar(f'{stage}/metrics_seen/lpips', metrics['seen']['lpips'], iteration)
                    tb_writer.add_scalar(f'{stage}/metrics_unseen/psnr', metrics['unseen']['psnr'], iteration)
                    tb_writer.add_scalar(f'{stage}/metrics_unseen/ssim', metrics['unseen']['ssim'], iteration)
                    tb_writer.add_scalar(f'{stage}/metrics_unseen/lpips', metrics['unseen']['lpips'], iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()

def setup_seed(seed):
    """Setup random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # Disabled network GUI for headless training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, load_iteration=getattr(args, 'load_iteration', None))

    # All done
    print("\nTraining complete.")
