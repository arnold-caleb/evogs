#!/bin/bash
#SBATCH --job-name=render_dnerf_hook
#SBATCH --output=slurm_outputs/render_dnerf_%j.out
#SBATCH --error=slurm_outputs/render_dnerf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:1

echo "=== RENDERING MUTANT TEMPORAL VIDEO ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

MODEL_DIR="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/output/dnerf/mutant_20251119_200816"

echo ""
echo "Model: $MODEL_DIR"
echo "Rendering 100 temporal frames from camera 10"

python scripts/render_dnerf_temporal.py \
    --model_path "$MODEL_DIR" \
    --source_path "data/dnerf/mutant" \
    --configs "arguments/dnerf/mutant.py" \
    --iteration 20000 \
    --num_frames 100 \
    --camera_index 10

echo ""
echo "=== COMPLETE ==="
echo "Output: $MODEL_DIR/videos/mutant_temporal_evolution_100frames.mp4"