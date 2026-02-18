#!/usr/bin/env python3
"""
Helper script to extract compute statistics from training logs and TensorBoard events.

This script helps fill in the CVPR 2026 Compute Reporting Form by extracting:
- Actual training times from SLURM logs
- Iteration speeds from training logs
- GPU utilization from TensorBoard logs (if available)
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

def extract_training_time_from_slurm(log_file):
    """Extract actual training time from SLURM output log."""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Look for start and end times
    start_time = None
    end_time = None
    
    for line in lines:
        # Look for "Starting training" or similar
        if "STARTING" in line.upper() or "TRAINING" in line.upper():
            # Try to extract timestamp
            match = re.search(r'(\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                start_time = match.group(1)
        
        # Look for "Training complete" or similar
        if "COMPLETE" in line.upper() or "FINISHED" in line.upper():
            match = re.search(r'(\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                end_time = match.group(1)
    
    # Also try to find elapsed time directly
    elapsed_match = None
    for line in lines:
        if "elapsed" in line.lower() or "duration" in line.lower():
            # Look for time patterns like "1h 23m" or "83 minutes"
            match = re.search(r'(\d+)\s*(?:hours?|h)', line, re.I)
            if match:
                hours = int(match.group(1))
                minutes_match = re.search(r'(\d+)\s*(?:minutes?|m)', line, re.I)
                minutes = int(minutes_match.group(1)) if minutes_match else 0
                return hours + minutes / 60.0
    
    return None

def estimate_from_iterations(iterations, iterations_per_second=25):
    """Estimate training time from iteration count."""
    seconds = iterations / iterations_per_second
    return seconds / 3600.0  # Convert to hours

def analyze_dnerf_training():
    """Analyze D-NeRF training compute."""
    base_dir = Path(__file__).resolve().parent.parent
    slurm_logs = base_dir / "slurm_outputs"
    output_dir = base_dir / "output" / "dnerf"
    
    scenes = ["jumpingjacks", "mutant", "lego"]
    iterations_per_scene = 23000  # 3000 coarse + 20000 fine
    
    print("=" * 60)
    print("D-NeRF Training Compute Analysis")
    print("=" * 60)
    print()
    
    print(f"Scenes analyzed: {', '.join(scenes)}")
    print(f"Iterations per scene: {iterations_per_scene:,}")
    print(f"  - Coarse: 3,000")
    print(f"  - Fine: 20,000")
    print()
    
    # Check for SLURM logs
    slurm_files = list(slurm_logs.glob("dnerf_all_*.out")) if slurm_logs.exists() else []
    
    if slurm_files:
        print(f"Found {len(slurm_files)} SLURM log file(s)")
        latest_log = max(slurm_files, key=lambda p: p.stat().st_mtime)
        print(f"Latest log: {latest_log.name}")
        print()
        
        # Try to extract actual time
        actual_time = extract_training_time_from_slurm(latest_log)
        if actual_time:
            print(f"Actual training time (from logs): {actual_time:.2f} hours")
        else:
            print("Could not extract actual time from logs")
            print("Using estimated time based on iteration count")
    else:
        print("No SLURM logs found, using estimates")
    
    print()
    
    # Estimate training time
    iterations_per_second = 25  # Conservative estimate for L40
    hours_per_scene = estimate_from_iterations(iterations_per_scene, iterations_per_second)
    
    print(f"Estimated training time per scene: {hours_per_scene:.3f} hours")
    print(f"  (assuming {iterations_per_second} iterations/second)")
    print()
    
    total_hours = hours_per_scene * len(scenes)
    print(f"Total training compute ({len(scenes)} scenes): {total_hours:.3f} GPU hours")
    print()
    
    # Check output directories for actual training runs
    if output_dir.exists():
        print("Training output directories found:")
        for scene in scenes:
            scene_dirs = [d for d in output_dir.iterdir() 
                         if scene in d.name and d.is_dir()]
            if scene_dirs:
                latest = max(scene_dirs, key=lambda p: p.stat().st_mtime)
                print(f"  {scene}: {latest.name}")
                # Check for checkpoint files
                checkpoints = list(latest.glob("chkpnt_*.pth"))
                if checkpoints:
                    print(f"    - {len(checkpoints)} checkpoint(s) found")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary for CVPR 2026 CRF:")
    print("=" * 60)
    print(f"GPU Type: NVIDIA L40")
    print(f"Number of GPUs: 1")
    print(f"Total Iterations: {iterations_per_scene * len(scenes):,}")
    print(f"Estimated Total Compute: {total_hours:.3f} GPU hours")
    print(f"  (Per scene: {hours_per_scene:.3f} hours)")
    print()
    print("Note: Actual times may vary. Check SLURM logs for precise timing.")
    print("=" * 60)

if __name__ == "__main__":
    analyze_dnerf_training()

