#!/bin/bash
#SBATCH --job-name=velocity_field
#SBATCH --output=slurm_outputs/velocity_field_%j.out
#SBATCH --error=slurm_outputs/velocity_field_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# === Load user config (paths, conda, SLURM account, etc.) ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "Copy the example and fill in your paths:"
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== VELOCITY FIELD (NEURAL ODE) TRAINING ==="
echo "Learns dx/dt = v(x,t) instead of Δx = f(x,t)"
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

cd "$EVOGS_ROOT"

# ============================================
# CONFIGURATION — edit these for your scene
# ============================================
SCENE="cut_roasted_beef"
DATA_DIR="${DYNERF_DATA}/${SCENE}"
CONFIG="arguments/dynerf/${SCENE}_future_velocity.py"
STATIC_CKPT="output/static_4anchors/frame0/chkpnt30000.pth"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export exp_name="velocity_field"
export scene_name="${SCENE}_${TIMESTAMP}"

echo ""
echo "=== RUNNING VELOCITY FIELD TRAINING ==="
echo "Dataset: ${SCENE}"
echo "Config:  ${CONFIG}"
echo "Output:  output/${exp_name}/${scene_name}"

python train.py \
    -s "$DATA_DIR" \
    --port 6019 \
    --expname "$exp_name/$scene_name" \
    --configs "$CONFIG" \
    --start_checkpoint "$STATIC_CKPT" \
    --iterations 14000 \
    --test_iterations 3000 7000 14000 \
    --save_iterations 3000 7000 14000 \
    --checkpoint_iterations 3000 7000 14000

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Output: output/$exp_name/$scene_name"
