#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
from typing import Optional

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    time1 = time()  # Initialize time1 before the loop

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2 = time()

    print("FPS:",(len(views)-1)/(time2-time1))
    multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    
    # Check if we have fixed-viewpoint temporal cameras (look for cam pattern in image_name)
    if name == "video" and len(views) > 0 and hasattr(views[0], 'image_name'):
        # Group frames by camera viewpoint
        camera_groups = {}
        for idx, view in enumerate(views):
            # Extract camera ID from image_name (e.g., "cam00_t000" -> "cam00")
            cam_id = view.image_name.split('_')[0] if '_' in view.image_name else 'cam00'
            if cam_id not in camera_groups:
                camera_groups[cam_id] = []
            camera_groups[cam_id].append(idx)
        
        # Save separate videos for each camera viewpoint
        if len(camera_groups) > 1:  # Multiple camera viewpoints detected
            print(f"\nSaving separate videos for {len(camera_groups)} camera viewpoints...")
            for cam_id, frame_indices in camera_groups.items():
                cam_images = [render_images[i] for i in frame_indices]
                video_path = os.path.join(model_path, name, "ours_{}".format(iteration), f'video_{cam_id}_temporal.mp4')
                imageio.mimwrite(video_path, cam_images, fps=30)
                print(f"  Saved {cam_id}: {len(cam_images)} frames -> {video_path}")
        else:
            # Single viewpoint or no grouping - save as before
            imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
    else:
        # Not video or no camera grouping - save as before
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams,
                skip_train : bool, skip_test : bool, skip_video: bool,
                video_camera_index: Optional[int] = None,
                video_camera_group: Optional[int] = None,
                video_camera_prefix: Optional[str] = None,
                video_camera_source: str = "video"):
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, cam_type)
        
        if not skip_video:
            chosen_source = video_camera_source or "video"
            if chosen_source == "train":
                all_video_cams = scene.getTrainCameras()
            elif chosen_source == "test":
                all_video_cams = scene.getTestCameras()
            else:
                chosen_source = "video"
                all_video_cams = scene.getVideoCameras()

            print(f"[VIDEO] camera_source='{chosen_source}' -> {len(all_video_cams)} cameras")
            video_cams = all_video_cams

            def camera_group_key(cam, fallback_idx: int):
                center = getattr(cam, "camera_center", None)
                if center is not None:
                    if isinstance(center, torch.Tensor):
                        center_np = center.detach().cpu().numpy()
                    else:
                        center_np = np.array(center)
                    return ("center", tuple(np.round(center_np, 4).tolist()))
                image_name = getattr(cam, "image_name", None)
                if image_name is not None and "_" in image_name:
                    return ("name", image_name.split('_')[0])
                camera_id = getattr(cam, "camera_id", None)
                if camera_id is not None:
                    return ("camera_id", camera_id)
                return ("fallback", fallback_idx)

            camera_groups = {}
            for idx, cam in enumerate(all_video_cams):
                key = camera_group_key(cam, idx)
                camera_groups.setdefault(key, []).append((idx, cam))

            if camera_groups:
                group_items = sorted(camera_groups.items(), key=lambda item: item[1][0][0])
                summary_lines = []
                frames_per_group = []
                for group_idx, (key, entries) in enumerate(group_items):
                    repr_idx, repr_cam = entries[0]
                    frames_per_group.append(len(entries))
                    summary_lines.append(
                        f"group{group_idx:02d}: repr index {repr_idx}, frames {len(entries)}, name={getattr(repr_cam, 'image_name', 'unknown')}"
                    )
                print("[VIDEO] Available camera groups:\n  " + "\n  ".join(summary_lines))
                if group_items:
                    representative = frames_per_group[0]
                    uniform = all(count == representative for count in frames_per_group)
                    if uniform:
                        print(f"[VIDEO] frames_per_view ≈ {representative}. To pick camera N by index use --video_camera_index {0} + N*{representative}.")
                    else:
                        print(f"[VIDEO] frames_per_view varies across groups: {frames_per_group[:8]}...")

            selection_done = False

            if video_camera_prefix and camera_groups:
                group_items = sorted(camera_groups.items(), key=lambda item: item[1][0][0])
                for group_idx, (key, entries) in enumerate(group_items):
                    repr_cam = entries[0][1]
                    name = getattr(repr_cam, "image_name", "")
                    if name and name.startswith(video_camera_prefix):
                        video_cams = [cam for _, cam in entries]
                        print(f"[VIDEO] Rendering group #{group_idx} '{key}' via prefix '{video_camera_prefix}' ({len(video_cams)} frames, repr name={name})")
                        selection_done = True
                        break
                if not selection_done:
                    print(f"[VIDEO] Warning: no camera group matches prefix '{video_camera_prefix}'.")

            if not selection_done and video_camera_group is not None:
                if not camera_groups:
                    raise ValueError("No camera groups available to select.")
                group_items = sorted(camera_groups.items(), key=lambda item: item[1][0][0])
                if video_camera_group < 0 or video_camera_group >= len(group_items):
                    raise ValueError(f"Requested video_camera_group {video_camera_group} but only {len(group_items)} groups available.")
                group_key, group_entries = group_items[video_camera_group]
                video_cams = [cam for _, cam in group_entries]
                print(f"[VIDEO] Rendering group #{video_camera_group} '{group_key}' ({len(video_cams)} frames)")
                selection_done = True

            elif not selection_done and video_camera_index is not None:
                if video_camera_index < 0 or video_camera_index >= len(all_video_cams):
                    raise ValueError(f"Requested video_camera_index {video_camera_index} but only {len(all_video_cams)-1} valid indices (0..{len(all_video_cams)-1}). "
                                     "If you want a different camera, multiply the camera id by frames_per_view (see log).")

                selected_group = None
                for key, entries in camera_groups.items():
                    if any(idx == video_camera_index for idx, _ in entries):
                        selected_group = (key, entries)
                        break

                if selected_group is None:
                    raise ValueError(f"Could not find camera group containing index {video_camera_index}.")

                group_key, group_entries = selected_group
                video_cams = [cam for _, cam in group_entries]
                print(f"[VIDEO] Rendering camera group '{group_key}' ({len(video_cams)} frames) selected by index {video_camera_index}")
                selection_done = True

            elif not selection_done and chosen_source in ("train", "test") and camera_groups:
                group_items = sorted(camera_groups.items(), key=lambda item: item[1][0][0])
                group_key, group_entries = group_items[0]
                video_cams = [cam for _, cam in group_entries]
                print(f"[VIDEO] Defaulting to first {chosen_source} group '{group_key}' ({len(video_cams)} frames). Use --video_camera_group to choose another.")

            render_set(dataset.model_path, "video", scene.loaded_iter, video_cams, gaussians, pipeline, background, cam_type)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--video_camera_index", type=int, default=None,
                        help="Optional index to render a single camera viewpoint (default: render all)")
    parser.add_argument("--video_camera_group", type=int, default=None,
                        help="Optional group id to render a block of temporal frames from train/test cameras")
    parser.add_argument("--video_camera_prefix", type=str, default=None,
                        help="Optional camera image_name prefix (e.g. cam00) to select a viewpoint")
    parser.add_argument("--video_camera_source", type=str, default="video",
                        choices=["video", "train", "test"],
                        help="Which camera set to use for temporal rendering")
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    video_camera_index = getattr(args, "video_camera_index", None)
    video_camera_group = getattr(args, "video_camera_group", None)
    video_camera_prefix = getattr(args, "video_camera_prefix", None)
    video_camera_source = getattr(args, "video_camera_source", "video")
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration,
                pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video,
                video_camera_index, video_camera_group, video_camera_prefix, video_camera_source)