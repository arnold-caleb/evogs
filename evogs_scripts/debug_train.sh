#!/bin/bash
# Quick debug script - runs training with Python debugger
# Usage: ./evogs_scripts/debug_train.sh [--pdb|--ipdb]

DEBUGGER=${1:-"--pdb"}

if [ "$DEBUGGER" = "--ipdb" ]; then
    DEBUG_CMD="python -m ipdb"
elif [ "$DEBUGGER" = "--pdb" ]; then
    DEBUG_CMD="python -m pdb"
else
    DEBUG_CMD="python"
fi

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=========================================="
echo "QUICK DEBUG SESSION"
echo "=========================================="
echo "Using debugger: $DEBUG_CMD"
echo ""

SCENE="cut_roasted_beef"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

srun --job-name=evogs_debug \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --mem=64GB \
     --time=2:00:00 \
     --gres=gpu:1 \
     --pty bash -c "
source $CONDA_INIT
conda activate $CONDA_ENV

cd $EVOGS_ROOT

echo 'Starting training with debugger...'
echo ''

$DEBUG_CMD train.py \
    -s '${DYNERF_DATA}/${SCENE}' \
    --port 6019 \
    --expname 'debug/${SCENE}_$TIMESTAMP' \
    --configs arguments/dynerf/${SCENE}_velocity.py \
    --iterations 1000 \
    --test_iterations 500 1000 \
    --save_iterations 1000
"
