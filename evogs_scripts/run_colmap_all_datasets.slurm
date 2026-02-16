#!/bin/bash
#SBATCH --job-name=colmap_all
#SBATCH --output=slurm_outputs/colmap_all_%j.out
#SBATCH --error=slurm_outputs/colmap_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

echo "=== COLMAP DENSE RECONSTRUCTION FOR DYNERF DATASETS ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

which colmap
colmap --version

cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

datasets=("coffee_martini" "cook_spinach" "sear_steak" "flame_salmon_1" "flame_steak")

for dataset in "${datasets[@]}"; do
    WORKDIR="data/dynerf/$dataset"
    
    echo ""
    echo "=== Processing: $dataset ==="
    
    # Check preprocessing
    if [ ! -d "$WORKDIR/sparse_" ] || [ ! -d "$WORKDIR/image_colmap" ]; then
        echo "Preprocessing incomplete. Skipping."
        continue
    fi
    
    # Check if already done
    if [ -f "$WORKDIR/points3D_downsample2.ply" ]; then
        echo "Point cloud exists. Skipping."
        continue
    fi
    
    # Run COLMAP
    bash colmap_dynerf.sh $WORKDIR
    
    # Downsample if successful
    if [ -f "$WORKDIR/colmap/dense/workspace/fused.ply" ]; then
        python scripts/downsample_point.py \
            $WORKDIR/colmap/dense/workspace/fused.ply \
            $WORKDIR/points3D_downsample2.ply
        echo "Complete: $WORKDIR/points3D_downsample2.ply"
    else
        echo "Dense reconstruction failed"
    fi
done

echo ""
echo "=== SUMMARY ==="
for dataset in "${datasets[@]}"; do
    WORKDIR="data/dynerf/$dataset"
    if [ -f "$WORKDIR/points3D_downsample2.ply" ]; then
        echo "✅ $dataset"
    else
        echo "❌ $dataset"
    fi
done