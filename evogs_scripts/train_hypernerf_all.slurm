#!/bin/bash
#SBATCH --job-name=train_hypernerf_all
#SBATCH --output=slurm_outputs/hypernerf_all_%j.out
#SBATCH --error=slurm_outputs/hypernerf_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

echo "=== TRAINING HYPERNERF SCENES ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

SCENES=(
    "americano"
)

echo ""
echo "Training ${#SCENES[@]} scene(s)"
echo ""

for scene in "${SCENES[@]}"; do
    echo "=== Training: $scene ==="
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="output/hypernerf/${scene}_${TIMESTAMP}_0.75"
    
    python train.py \
        --source_path "data/nerfies/${scene}" \
        --model_path "$OUTPUT_DIR" \
        --configs "arguments/hypernerf/${scene}.py" \
        --expname "hypernerf_${scene}" \
        --eval \
        --iterations 14000 \
        --test_iterations 7000 14000 \
        --save_iterations 7000 14000 \
        --checkpoint_iterations 7000 14000 \
        --sparse_supervision \
        --supervised_frame_stride 0 \
        --supervised_frame_offset 0
    
    echo "Output: $OUTPUT_DIR"
    echo ""
done

echo "=== COMPLETE ==="