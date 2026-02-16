#!/usr/bin/env python3
"""
Quick script to count Gaussians from a PLY file header.
"""
import sys
import os

def count_gaussians_from_ply(ply_path):
    """Count Gaussians by reading PLY header."""
    if not os.path.exists(ply_path):
        print(f"Error: File not found: {ply_path}")
        return None
    
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                break
            header_lines.append(line)
            if b'end_header' in line:
                break
            if b'element vertex' in line:
                # Extract count
                parts = line.decode('utf-8', errors='ignore').strip().split()
                if len(parts) >= 3:
                    try:
                        count = int(parts[2])
                        return count
                    except ValueError:
                        pass
    
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_gaussians.py <ply_path>")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    count = count_gaussians_from_ply(ply_path)
    
    if count is not None:
        print(f"Number of Gaussians: {count:,}")
    else:
        print("Could not determine Gaussian count from PLY file")

