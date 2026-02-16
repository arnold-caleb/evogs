#!/usr/bin/env python
"""
Convert HyperNeRF points.npy to points3D_downsample2.ply format.
"""
import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement

def convert_npy_to_ply(npy_path, ply_path):
    """Convert points.npy to points3D_downsample2.ply"""
    print(f"  Loading {os.path.basename(npy_path)}...")
    points = np.load(npy_path)
    
    # Create PLY with random colors (will be overridden during training)
    xyz = points[:, :3] if points.shape[1] >= 3 else points
    rgb = np.random.randint(0, 255, size=(len(xyz), 3), dtype=np.uint8)
    
    # Create structured array for PLY
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    vertex_data = np.zeros(len(xyz), dtype=dtype)
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['red'] = rgb[:, 0]
    vertex_data['green'] = rgb[:, 1]
    vertex_data['blue'] = rgb[:, 2]
    
    # Create PLY element and save
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    
    ply_data.write(ply_path)
    print(f"  ✓ Converted {len(xyz)} points → {os.path.basename(ply_path)}")

if __name__ == "__main__":
    # Convert all HyperNeRF datasets
    hypernerf_dir = "/n/fs/visualai-scr/Data/hypernerf"
    scenes = sorted([d for d in os.listdir(hypernerf_dir) if os.path.isdir(os.path.join(hypernerf_dir, d))])
    
    print(f"Converting points.npy → points3D_downsample2.ply for {len(scenes)} HyperNeRF datasets...")
    print("")
    
    for scene in scenes:
        scene_path = os.path.join(hypernerf_dir, scene)
        npy_path = os.path.join(scene_path, "points.npy")
        ply_path = os.path.join(scene_path, "points3D_downsample2.ply")
        
        if os.path.exists(ply_path):
            print(f"[{scene}] Already has PLY, skipping...")
        elif os.path.exists(npy_path):
            print(f"[{scene}]")
            convert_npy_to_ply(npy_path, ply_path)
        else:
            print(f"[{scene}] ✗ No points.npy found")
    
    print("")
    print("✓ All conversions complete!")

