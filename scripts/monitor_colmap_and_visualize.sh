#!/bin/bash
# Monitor COLMAP job and auto-visualize when complete

JOB_ID=$1
WORKDIR="data/dynerf/cut_roasted_beef"

if [ -z "$JOB_ID" ]; then
    echo "Usage: bash scripts/monitor_colmap_and_visualize.sh <job_id>"
    echo "Example: bash scripts/monitor_colmap_and_visualize.sh 25219680"
    exit 1
fi

cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

echo "=== Monitoring COLMAP Job $JOB_ID ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    # Check if job is still running
    job_status=$(squeue -j $JOB_ID 2>/dev/null | tail -n+2)
    
    if [ -z "$job_status" ]; then
        echo "Job $JOB_ID completed!"
        break
    fi
    
    echo "[$(date +%H:%M:%S)] Job still running..."
    sleep 30
done

echo ""
echo "=== Checking Results ==="

# Check for point cloud
if [ -f "$WORKDIR/points3D_downsample2.ply" ]; then
    echo "✅ Point cloud created!"
    
    num_points=$(grep "element vertex" $WORKDIR/points3D_downsample2.ply | awk '{print $3}')
    echo "Number of points: $num_points"
    
    echo ""
    echo "=== Visualizing Point Cloud ==="
    source /u/aa0008/miniconda/etc/profile.d/conda.sh
    conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12
    
    python scripts/visualize_point_cloud.py \
        $WORKDIR/points3D_downsample2.ply \
        visualizations/colmap_pointcloud
    
    echo ""
    echo "✅ Visualizations saved:"
    echo "  - visualizations/colmap_pointcloud_multiview.png"
    echo "  - visualizations/colmap_pointcloud_density.png"
    echo ""
    echo "Next: Train with this point cloud:"
    echo "  sbatch slurm_scripts/run_hq_training.slurm"
else
    echo "❌ Point cloud not found - check logs:"
    echo "  slurm_outputs/colmap_recon_${JOB_ID}.out"
    echo "  slurm_outputs/colmap_recon_${JOB_ID}.err"
fi

