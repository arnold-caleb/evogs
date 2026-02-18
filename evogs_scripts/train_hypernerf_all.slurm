#!/bin/bash
#SBATCH --job-name=train_hypernerf_all
#SBATCH --output=slurm_outputs/hypernerf_all_%j.out
#SBATCH --error=slurm_outputs/hypernerf_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== TRAINING HYPERNERF SCENES ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$EVOGS_ROOT"

SCENES=(
    "americano"
)

echo ""
echo "Training ${#SCENES[@]} scene(s)"
echo ""

for scene in "${SCENES[@]}"; do
    echo "=== Training: $scene ==="
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="output/hypernerf/${scene}_${TIMESTAMP}"
    
    python train.py \
        --source_path "${HYPERNERF_DATA}/${scene}" \
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
