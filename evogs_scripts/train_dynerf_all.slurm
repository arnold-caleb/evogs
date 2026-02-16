#!/bin/bash
#SBATCH --job-name=velocity_field
#SBATCH --output=/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/slurm_outputs/velocity_field_e_%j.out
#SBATCH --error=/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/slurm_outputs/velocity_field_e_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

echo "=== VELOCITY FIELD (NEURAL ODE) TRAINING ==="
echo "Learns dx/dt = v(x,t) instead of Δx = f(x,t)"
echo "=== SYSTEM INFORMATION ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

DATA_DIR="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/data/dynerf/cut_roasted_beef"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export exp_name="velocity_field"
export scene_name="cut_roasted_beef_${TIMESTAMP}"

echo ""
echo "=== RUNNING VELOCITY FIELD TRAINING ==="
echo "Dataset: cut_roasted_beef"
echo "Output: $scene_name"

STATIC_CKPT="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/output/static_4anchors/frame0_20251104_103403/chkpnt30000.pth"

python /n/fs/aa-rldiff/view_synthesis/gaussian-splatting/train.py \
    -s "$DATA_DIR" \
    --port 6019 \
    --expname "$exp_name/$scene_name" \
    --configs /n/fs/aa-rldiff/view_synthesis/gaussian-splatting/arguments/dynerf/cut_roasted_beef_velocity.py \
    --start_checkpoint "$STATIC_CKPT" \
    --iterations 14000 \
    --test_iterations 3000 7000 14000 \
    --save_iterations 3000 7000 14000 \
    --checkpoint_iterations 3000 7000 14000

echo ""
echo "=== TRAINING COMPLETE ==="
OUTPUT_DIR="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/output/$exp_name/$scene_name"
echo "Output: $OUTPUT_DIR"