#!/bin/bash
#SBATCH --job-name=train_dnerf_all
#SBATCH --output=slurm_outputs/dnerf_all_%j.out
#SBATCH --error=slurm_outputs/dnerf_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== TRAINING D-NERF SCENES ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$EVOGS_ROOT"

SCENES=("jumpingjacks")

echo ""
echo "Training ${#SCENES[@]} scene(s)"
echo ""

for scene in "${SCENES[@]}"; do
    echo "=== Training: $scene ==="
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="output/dnerf/${scene}_${TIMESTAMP}"
    
    python train.py \
        --source_path "${DNERF_DATA}/${scene}" \
        --model_path "$OUTPUT_DIR" \
        --configs "arguments/dnerf/${scene}.py" \
        --expname "dnerf_${scene}" \
        --eval \
        --iterations 20000 \
        --test_iterations 10000 20000 \
        --save_iterations 10000 20000 \
        --checkpoint_iterations 10000 20000
    
    echo "Output: $OUTPUT_DIR"
    echo ""
done

echo "=== COMPLETE ==="
