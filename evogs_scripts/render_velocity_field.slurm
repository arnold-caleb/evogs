#!/bin/bash
#SBATCH --job-name=render_velocity
#SBATCH --output=slurm_outputs/render_velocity_%j.out
#SBATCH --error=slurm_outputs/render_velocity_%j.err
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

echo "=== RENDERING VELOCITY FIELD TEMPORAL VIDEO ==="
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
SCENE="cut_roasted_beef"
MODEL_PATH="output/velocity_field/${SCENE}"
ITERATION=14000
CONFIG="arguments/dynerf/${SCENE}_velocity.py"
SOURCE_PATH="${DYNERF_DATA}/${SCENE}"

echo ""
echo "Model: $MODEL_PATH"
echo "Rendering 300 temporal frames from fixed viewpoint"

python scripts/render_velocity_field_temporal.py \
    --model_path "$MODEL_PATH" \
    --source_path "$SOURCE_PATH" \
    --configs "$CONFIG" \
    --iteration $ITERATION \
    --num_temporal_frames 300 \
    --num_interp 5 \
    --fps 30

echo ""
echo "=== COMPLETE ==="
echo "Output: $MODEL_PATH/videos/"
