#!/bin/bash
# Interactive training script using srun for debugging with breakpoints
# Usage: ./train_interactive.sh [scene_name] [config_file]
#
# Example:
#   ./train_interactive.sh cut_roasted_beef arguments/dynerf/cut_roasted_beef_velocity.py
#
# This will allocate resources and drop you into an interactive session
# where you can run Python with pdb/ipdb breakpoints.

SCENE_NAME=${1:-"cut_roasted_beef"}
CONFIG_FILE=${2:-"arguments/dynerf/cut_roasted_beef_velocity.py"}

echo "=========================================="
echo "INTERACTIVE TRAINING SESSION"
echo "=========================================="
echo "Scene: $SCENE_NAME"
echo "Config: $CONFIG_FILE"
echo ""
echo "Requesting interactive GPU node..."
echo "This may take a few minutes if the queue is busy."
echo ""

# Launch interactive session with srun
srun --job-name=evogs_debug \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --mem=64GB \
     --time=4:00:00 \
     --account=visualai \
     --partition=visualai \
     --gres=gpu:l40:1 \
     --pty bash -c "
echo '=========================================='
echo 'INTERACTIVE SESSION STARTED'
echo '=========================================='
echo 'Node:' \$(hostname)
echo ''
nvidia-smi
echo ''
echo 'Activating conda environment...'
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12
echo 'Environment activated!'
echo ''
echo '=========================================='
echo 'READY FOR DEBUGGING'
echo '=========================================='
echo ''
echo 'You can now run training with breakpoints:'
echo ''
echo '  python -m pdb train.py \\'
echo '    -s data/dynerf/$SCENE_NAME \\'
echo '    --port 6019 \\'
echo '    --expname debug/$SCENE_NAME \\'
echo '    --configs $CONFIG_FILE \\'
echo '    --iterations 1000'
echo ''
echo 'Or with ipdb (if installed):'
echo ''
echo '  python -m ipdb train.py ...'
echo ''
echo 'Or add breakpoints in your code:'
echo ''
echo '  import pdb; pdb.set_trace()  # Add this line where you want to break'
echo ''
echo 'Current directory:' \$(pwd)
echo '=========================================='
echo ''

# Change to project directory
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

# Drop into interactive bash shell
exec bash
"

