#!/bin/bash
# Quick debug script - runs training with Python debugger
# Usage: ./debug_train.sh [--pdb|--ipdb]

DEBUGGER=${1:-"--pdb"}

if [ "$DEBUGGER" = "--ipdb" ]; then
    DEBUG_CMD="python -m ipdb"
elif [ "$DEBUGGER" = "--pdb" ]; then
    DEBUG_CMD="python -m pdb"
else
    DEBUG_CMD="python"
fi

echo "=========================================="
echo "QUICK DEBUG SESSION"
echo "=========================================="
echo "Using debugger: $DEBUG_CMD"
echo ""

DATA_DIR="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/data/dynerf/cut_roasted_beef"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

srun --job-name=evogs_debug \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --mem=64GB \
     --time=2:00:00 \
     --account=visualai \
     --partition=visualai \
     --gres=gpu:l40:1 \
     --pty bash -c "
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

echo 'Starting training with debugger...'
echo ''

$DEBUG_CMD train.py \
    -s '$DATA_DIR' \
    --port 6019 \
    --expname 'debug/cut_roasted_beef_$TIMESTAMP' \
    --configs arguments/dynerf/cut_roasted_beef_velocity.py \
    --iterations 1000 \
    --test_iterations 500 1000 \
    --save_iterations 1000
"

