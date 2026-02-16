#!/usr/bin/env python3
"""
Generate smart initial point cloud from camera poses and first frame.
Much better than random initialization!
"""

import numpy as np
import sys
import os
from pathlib import Path

def load_poses_bounds(dataset_path):
    """Load camera poses from poses_bounds.npy"""
    poses_bounds = np.load(os.path.join(dataset_path, 'poses_bounds.npy'))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]
    
    # Extract camera centers
    poses_converted = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    camera_centers = -np.matmul(poses_converted[:, :3, :3].transpose(0, 2, 1), 
                                 poses_converted[:, :3, 3:4]).squeeze(-1)
    
    return camera_centers, bounds, poses_converted

def generate_viewing_frustum_points(camera_center, camera_rot, bounds, num_points=500):
    """Generate points in the viewing frustum of a camera"""
    near, far = bounds
    
    # Generate points along viewing rays
    # Sample in frustum shape
    u = np.random.uniform(-0.5, 0.5, num_points)
    v = np.random.uniform(-0.5, 0.5, num_points)
    depth = np.random.uniform(near, far, num_points)
    
    # Convert to 3D points in camera space
    points_cam = np.stack([u * depth, v * depth, depth], axis=1)
    
    # Transform to world space
    points_world = camera_center + (camera_rot @ points_cam.T).T
    
    return points_world

def generate_smart_point_cloud(dataset_path, output_path, 
                               points_per_cam=2000, 
                               background_points=10000):
    """
    Generate smart initial point cloud from camera frustums.
    
    This creates points where cameras are actually looking,
    which is MUCH better than random initialization.
    """
    print(f"Generating smart point cloud for {dataset_path}")
    
    # Load camera poses
    camera_centers, bounds, poses = load_poses_bounds(dataset_path)
    num_cams = len(camera_centers)
    
    print(f"Found {num_cams} cameras")
    print(f"Depth range: {bounds.min():.2f} to {bounds.max():.2f}")
    
    # Generate points in each camera's frustum
    all_points = []
    for i, (center, pose, bound) in enumerate(zip(camera_centers, poses, bounds)):
        rot_matrix = pose[:3, :3]
        points = generate_viewing_frustum_points(center, rot_matrix, bound, points_per_cam)
        all_points.append(points)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{num_cams} cameras...")
    
    # Combine all frustum points
    frustum_points = np.vstack(all_points)
    print(f"Generated {len(frustum_points)} frustum points")
    
    # Compute scene bounds from camera centers
    scene_min = camera_centers.min(axis=0) - bounds.mean() * 0.5
    scene_max = camera_centers.max(axis=0) + bounds.mean() * 0.5
    
    print(f"Scene bounds: {scene_min} to {scene_max}")
    
    # Add some background points
    bg_points = np.random.uniform(scene_min, scene_max, (background_points, 3))
    
    # Combine
    xyz = np.vstack([frustum_points, bg_points])
    
    # Initialize colors (neutral gray with some variation)
    rgb = np.random.uniform(0.4, 0.6, (len(xyz), 3))
    
    # Save PLY file
    save_ply(output_path, xyz, rgb)
    
    print(f"✅ Saved {len(xyz)} points to {output_path}")
    print(f"  - Frustum points: {len(frustum_points)}")
    print(f"  - Background points: {len(bg_points)}")
    
    return xyz, rgb

def save_ply(path, xyz, rgb):
    """Save point cloud as PLY file"""
    # Write PLY header
    with open(path, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {len(xyz)}\n'.encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property float nx\n')
        f.write(b'property float ny\n')
        f.write(b'property float nz\n')
        f.write(b'property float red\n')
        f.write(b'property float green\n')
        f.write(b'property float blue\n')
        f.write(b'end_header\n')
        
        # Write data
        normals = np.zeros_like(xyz)
        data = np.hstack([xyz, normals, rgb])
        data.astype(np.float32).tofile(f)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_smart_init.py <dataset_path> [output_path]")
        print("Example: python scripts/generate_smart_init.py data/dynerf/cut_roasted_beef")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(dataset_path, 'points3D_downsample2.ply')
    
    generate_smart_point_cloud(dataset_path, output_path, 
                               points_per_cam=2000,  # 20 cams × 2000 = 40K points
                               background_points=5000)

