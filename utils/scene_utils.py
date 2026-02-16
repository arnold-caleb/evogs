import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np
import torch
import os
from datetime import datetime
import imageio

import copy
@torch.no_grad()
def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now, dataset_type):
    def render(gaussians, viewpoint, path, scaling, cam_type):
        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=cam_type)
        
        # Add timestamp to labels
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label1 = f"stage:{stage},iter:{iteration},time:{timestamp}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "elapsed:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        if dataset_type == "PanopticSports":
            gt_np = viewpoint['image'].permute(1,2,0).cpu().numpy()
        else:
            gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((gt_np, image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  
        draw1 = ImageDraw.Draw(image_with_labels)
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40) 
        text_color = (255, 0, 0)  
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10) 
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(path)
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    for idx in range(len(viewpoints)):
        # Add timestamp to filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}_{timestamp}.jpg")
        render(gaussians,viewpoints[idx],image_save_path,scaling = 1,cam_type=dataset_type)
    pc_mask = gaussians.get_opacity
    pc_mask = pc_mask > 0.1

def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    plt.savefig(filename)

@torch.no_grad()
def render_fixed_viewpoint_temporal_video(scene, gaussians, render_func, pipe, background, stage, iteration, dataset_type, 
                                        fixed_camera_idx=0, total_frames=300, fps=30, save_path=None):
    """
    Render a temporal video from a fixed viewpoint to show extrapolation failure.
    
    Args:
        scene: Scene object containing cameras
        gaussians: Trained Gaussian model
        render_func: Rendering function
        pipe: Pipeline parameters
        background: Background color
        stage: Training stage ("coarse" or "fine")
        iteration: Current training iteration
        dataset_type: Dataset type
        fixed_camera_idx: Index of camera to use for fixed viewpoint
        total_frames: Total number of frames to render (300 for full sequence)
        fps: Frames per second for video
        save_path: Path to save the video (optional)
    """
    print(f"🎬 Rendering fixed viewpoint temporal video...")
    print(f"   Camera: {fixed_camera_idx}, Frames: {total_frames}, FPS: {fps}")
    
    # Get all cameras from the scene
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    
    if fixed_camera_idx >= len(all_cameras):
        print(f"⚠️  Camera index {fixed_camera_idx} out of range. Using camera 0.")
        fixed_camera_idx = 0
    
    # Use the fixed camera as base viewpoint
    base_camera = all_cameras[fixed_camera_idx]
    print(f"   Using camera: {base_camera.image_name}")
    print(f"   Original camera time: {base_camera.time}")
    print(f"   Camera image size: {base_camera.image_width}x{base_camera.image_height}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_path is None:
        save_path = os.path.join(scene.model_path, f"{stage}_temporal_video_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Render frames
    rendered_frames = []
    frame_times = []
    
    print(f"   Rendering {total_frames} frames...")
    for frame_idx in range(total_frames):
        # Normalize time to [0, 1] for the full sequence
        time_normalized = float(frame_idx) / (total_frames - 1)
        
        # Create a copy of the base camera with modified time
        camera_copy = copy.deepcopy(base_camera)
        camera_copy.time = time_normalized
        
        # Render the frame
        render_pkg = render_func(camera_copy, gaussians, pipe, background, stage=stage, cam_type=dataset_type)
        image = render_pkg["render"]
        
        # Debug: Check if image is all zeros or all ones
        if frame_idx == 0:  # Only print for first frame to avoid spam
            print(f"     Debug - Image shape: {image.shape}")
            print(f"     Debug - Image min/max: {image.min():.4f}/{image.max():.4f}")
            print(f"     Debug - Image mean: {image.mean():.4f}")
        
        # Convert to numpy and save
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Add frame info overlay
        from PIL import Image, ImageDraw, ImageFont
        pil_image = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add frame information
        frame_info = f"Frame: {frame_idx:03d}/{total_frames-1:03d}"
        time_info = f"Time: {time_normalized:.3f}"
        train_info = "TRAINED" if frame_idx < 150 else "EXTRAPOLATED"
        
        # Color coding: green for trained frames, red for extrapolated
        color = (0, 255, 0) if frame_idx < 150 else (255, 0, 0)
        
        draw.text((10, 10), frame_info, fill=color, font=font)
        draw.text((10, 35), time_info, fill=color, font=font)
        draw.text((10, 60), train_info, fill=color, font=font)
        
        # Convert back to numpy
        image_with_overlay = np.array(pil_image)
        rendered_frames.append(image_with_overlay)
        frame_times.append(time_normalized)
        
        # Save individual frame
        frame_path = os.path.join(save_path, f"frame_{frame_idx:03d}.png")
        imageio.imwrite(frame_path, image_with_overlay)
        
        if frame_idx % 50 == 0:
            print(f"     Rendered frame {frame_idx}/{total_frames-1}")
    
    # Create video from rendered frames
    video_path = os.path.join(save_path, f"temporal_video_iter{iteration}_{timestamp}.mp4")
    print(f"   Creating video: {video_path}")
    
    try:
        imageio.mimwrite(video_path, rendered_frames, fps=fps, format="FFMPEG", quality=8)
        print(f"✅ Temporal video saved: {video_path}")
    except Exception as e:
        print(f"❌ Failed to create video: {e}")
        # Fallback: save as GIF
        gif_path = os.path.join(save_path, f"temporal_video_iter{iteration}_{timestamp}.gif")
        try:
            imageio.mimwrite(gif_path, rendered_frames, fps=fps)
            print(f"✅ Temporal GIF saved: {gif_path}")
        except Exception as e2:
            print(f"❌ Failed to create GIF: {e2}")
    
    # Create summary info
    summary_path = os.path.join(save_path, "temporal_rendering_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Fixed Viewpoint Temporal Rendering Summary\n")
        f.write(f"==========================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Stage: {stage}\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Fixed Camera: {base_camera.image_name} (index {fixed_camera_idx})\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Trained Frames: 0-149 (150 frames)\n")
        f.write(f"Extrapolated Frames: 150-299 (150 frames)\n")
        f.write(f"Video Path: {video_path}\n")
        f.write(f"Individual Frames: {save_path}/frame_*.png\n")
        f.write(f"\nExpected Result:\n")
        f.write(f"- Frames 0-149: Good reconstruction (trained on these)\n")
        f.write(f"- Frames 150-299: Poor reconstruction (extrapolation failure)\n")
        f.write(f"- This demonstrates the limitation of deterministic methods\n")
    
    print(f"📋 Summary saved: {summary_path}")
    return video_path, save_path

