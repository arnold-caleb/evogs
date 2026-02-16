#!/bin/bash
#SBATCH --job-name=render_velocity
#SBATCH --output=/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/slurm_outputs/render_velocity_%j.out
#SBATCH --error=/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/slurm_outputs/render_velocity_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

echo "=== RENDERING VELOCITY FIELD TEMPORAL VIDEO ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

MODEL_PATH="output/velocity_field_optimized_long/flame_steak_20251024_221036"
ITERATION=20000
CONFIG="arguments/dynerf/flame_steak_velocity_xyz_only.py"
SOURCE_PATH="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/data/dynerf/flame_steak"

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
echo "Output: $MODEL_PATH/videos/velocity_field_temporal_300frames_5interp.mp4"