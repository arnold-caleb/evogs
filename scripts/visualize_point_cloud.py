#!/usr/bin/env python3
"""
Visualize point cloud and save as images viewable in the editor
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from plyfile import PlyData

def read_ply(path):
    """Read PLY file"""
    plydata = PlyData.read(path)
    
    xyz = np.stack([
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ], axis=1)
    
    # Try to read colors
    try:
        rgb = np.stack([
            np.asarray(plydata.elements[0]["red"]),
            np.asarray(plydata.elements[0]["green"]),
            np.asarray(plydata.elements[0]["blue"])
        ], axis=1)
    except:
        rgb = None
    
    return xyz, rgb

def visualize_point_cloud(ply_path, output_prefix="pointcloud_viz", num_points=10000):
    """
    Create multiple viewpoint visualizations of the point cloud
    """
    print(f"Loading point cloud from {ply_path}")
    xyz, rgb = read_ply(ply_path)
    
    print(f"Total points: {len(xyz)}")
    print(f"Bounds: {xyz.min(axis=0)} to {xyz.max(axis=0)}")
    
    # Subsample for visualization if too many points
    if len(xyz) > num_points:
        indices = np.random.choice(len(xyz), num_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices] if rgb is not None else None
        print(f"Subsampled to {num_points} points for visualization")
    
    # Normalize colors if needed
    if rgb is not None:
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        colors = rgb
    else:
        colors = 'blue'
    
    # Create figure with multiple viewpoints
    fig = plt.figure(figsize=(20, 15))
    
    viewpoints = [
        (30, 45, "Front-Right"),
        (30, 135, "Front-Left"),
        (30, -45, "Back-Right"),
        (0, 0, "Front"),
        (90, 0, "Top"),
        (0, 90, "Side")
    ]
    
    for idx, (elev, azim, name) in enumerate(viewpoints, 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # Plot points
        if rgb is not None:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                      c=colors, s=1, alpha=0.6)
        else:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                      c='blue', s=1, alpha=0.6)
        
        # Set viewpoint
        ax.view_init(elev=elev, azim=azim)
        
        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name} View (elev={elev}°, azim={azim}°)')
        
        # Equal aspect ratio
        max_range = np.array([
            xyz[:, 0].max()-xyz[:, 0].min(),
            xyz[:, 1].max()-xyz[:, 1].min(),
            xyz[:, 2].max()-xyz[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (xyz[:, 0].max()+xyz[:, 0].min()) * 0.5
        mid_y = (xyz[:, 1].max()+xyz[:, 1].min()) * 0.5
        mid_z = (xyz[:, 2].max()+xyz[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(f'Point Cloud Visualization\n{ply_path}\n{len(xyz)} points', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_prefix}_multiview.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_file}")
    plt.close()
    
    # Also create a simple density heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY projection
    axes[0].hist2d(xyz[:, 0], xyz[:, 1], bins=50, cmap='viridis')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Projection (Top View)')
    axes[0].set_aspect('equal')
    
    # XZ projection
    axes[1].hist2d(xyz[:, 0], xyz[:, 2], bins=50, cmap='viridis')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Projection (Front View)')
    axes[1].set_aspect('equal')
    
    # YZ projection
    axes[2].hist2d(xyz[:, 1], xyz[:, 2], bins=50, cmap='viridis')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Projection (Side View)')
    axes[2].set_aspect('equal')
    
    plt.suptitle('Point Cloud Density Heatmaps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    density_file = f"{output_prefix}_density.png"
    plt.savefig(density_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved density heatmap to {density_file}")
    plt.close()
    
    return output_file, density_file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_point_cloud.py <ply_file> [output_prefix]")
        print("Example: python scripts/visualize_point_cloud.py data/dynerf/cut_roasted_beef/points3D_downsample2.ply")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "pointcloud_viz"
    
    visualize_point_cloud(ply_path, output_prefix)

