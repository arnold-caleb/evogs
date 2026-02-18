#!/bin/bash
# Interactive training script using srun for debugging with breakpoints
# Usage: ./evogs_scripts/train_interactive.sh [scene_name] [config_file]
#
# Example:
#   ./evogs_scripts/train_interactive.sh cut_roasted_beef arguments/dynerf/cut_roasted_beef_velocity.py

SCENE_NAME=${1:-"cut_roasted_beef"}
CONFIG_FILE=${2:-"arguments/dynerf/cut_roasted_beef_velocity.py"}

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=========================================="
echo "INTERACTIVE TRAINING SESSION"
echo "=========================================="
echo "Scene: $SCENE_NAME"
echo "Config: $CONFIG_FILE"
echo ""
echo "Requesting interactive GPU node..."
echo ""

srun --job-name=evogs_debug \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --mem=64GB \
     --time=4:00:00 \
     --gres=gpu:1 \
     --pty bash -c "
echo '=========================================='
echo 'INTERACTIVE SESSION STARTED'
echo '=========================================='
echo 'Node:' \$(hostname)
echo ''
nvidia-smi
echo ''
echo 'Activating conda environment...'
source $CONDA_INIT
conda activate $CONDA_ENV
echo 'Environment activated!'
echo ''
echo '=========================================='
echo 'READY FOR DEBUGGING'
echo '=========================================='
echo ''
echo 'You can now run training with breakpoints:'
echo ''
echo '  python -m pdb train.py \\\\'
echo '    -s data/dynerf/$SCENE_NAME \\\\'
echo '    --port 6019 \\\\'
echo '    --expname debug/$SCENE_NAME \\\\'
echo '    --configs $CONFIG_FILE \\\\'
echo '    --iterations 1000'
echo ''
echo 'Current directory:' \$(pwd)
echo '=========================================='
echo ''

cd $EVOGS_ROOT
exec bash
"
