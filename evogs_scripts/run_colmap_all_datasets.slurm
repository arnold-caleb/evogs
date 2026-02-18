#!/bin/bash
#SBATCH --job-name=colmap_all
#SBATCH --output=slurm_outputs/colmap_all_%j.out
#SBATCH --error=slurm_outputs/colmap_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== COLMAP DENSE RECONSTRUCTION FOR DYNERF DATASETS ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

which colmap
colmap --version

cd "$EVOGS_ROOT"

datasets=("coffee_martini" "cook_spinach" "sear_steak" "flame_salmon_1" "flame_steak")

for dataset in "${datasets[@]}"; do
    WORKDIR="${DYNERF_DATA}/$dataset"
    
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
    WORKDIR="${DYNERF_DATA}/$dataset"
    if [ -f "$WORKDIR/points3D_downsample2.ply" ]; then
        echo "✅ $dataset"
    else
        echo "❌ $dataset"
    fi
done
