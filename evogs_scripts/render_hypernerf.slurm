#!/bin/bash
#SBATCH --job-name=render_hypernerf
#SBATCH --output=slurm_outputs/render_hypernerf_%j.out
#SBATCH --error=slurm_outputs/render_hypernerf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== RENDERING HYPERNERF TEMPORAL VIDEO ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source "$CONDA_INIT"
conda activate "$CONDA_ENV"
cd "$EVOGS_ROOT"

# ============================================
# CONFIGURATION — edit for your model
# ============================================
SCENE="americano"
MODEL_DIR="output/hypernerf/${SCENE}"

echo ""
echo "Model: $MODEL_DIR"
echo "Rendering 150 temporal frames from camera 8"

python scripts/render_hypernerf_temporal.py \
    --model_path "$MODEL_DIR" \
    --source_path "${HYPERNERF_DATA}/${SCENE}" \
    --configs "arguments/hypernerf/${SCENE}.py" \
    --iteration 14000 \
    --num_frames 150 \
    --camera_index 8

echo ""
echo "=== COMPLETE ==="
echo "Output: $MODEL_DIR/videos/"
