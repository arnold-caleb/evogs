#!/bin/bash
#SBATCH --job-name=train_anchors
#SBATCH --output=slurm_outputs/train_anchors_%j.out
#SBATCH --error=slurm_outputs/train_anchors_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=visualai
#SBATCH --partition=visualai
#SBATCH --gres=gpu:l40:1

echo "=== TRAINING 4-ANCHOR STATIC GAUSSIANS ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

DATA_DIR="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/data/dynerf/cut_roasted_beef"
CONFIG="arguments/static/cut_roasted_beef_frame0_hq.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="output/static_4anchors"

echo ""
echo "Training frames: 0, 75, 150, 225 (30K iterations each)"
echo ""

# Frame 0 (canonical)
echo "=== [1/4] Training Frame 0 ==="
OUTPUT_DIR_F0="${BASE_OUTPUT_DIR}/frame0_${TIMESTAMP}"

python train_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F0" \
    --configs "$CONFIG" \
    --dataset_type "dynerf_static" \
    --frame_idx 0 \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F0" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet

echo "Output: $OUTPUT_DIR_F0"
echo ""

# Frame 75
echo "=== [2/4] Training Frame 75 ==="
OUTPUT_DIR_F75="${BASE_OUTPUT_DIR}/frame75_${TIMESTAMP}"
CANONICAL_CHECKPOINT="${OUTPUT_DIR_F0}/chkpnt30000.pth"

python train_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F75" \
    --configs "arguments/static/cut_roasted_beef_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 75 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F75" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet

echo "Output: $OUTPUT_DIR_F75"
echo ""

# Frame 150
echo "=== [3/4] Training Frame 150 ==="
OUTPUT_DIR_F150="${BASE_OUTPUT_DIR}/frame150_${TIMESTAMP}"

python train_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F150" \
    --configs "arguments/static/cut_roasted_beef_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 150 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F150" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet

echo "Output: $OUTPUT_DIR_F150"
echo ""

# Frame 225
echo "=== [4/4] Training Frame 225 ==="
OUTPUT_DIR_F225="${BASE_OUTPUT_DIR}/frame225_${TIMESTAMP}"

python train_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F225" \
    --configs "arguments/static/cut_roasted_beef_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 225 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F225" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet

echo "Output: $OUTPUT_DIR_F225"
echo ""

echo "=== COMPLETE ==="
echo "Anchor checkpoints:"
echo "  Frame 0:   $OUTPUT_DIR_F0/chkpnt30000.pth"
echo "  Frame 75:  $OUTPUT_DIR_F75/chkpnt30000.pth"
echo "  Frame 150: $OUTPUT_DIR_F150/chkpnt30000.pth"
echo "  Frame 225: $OUTPUT_DIR_F225/chkpnt30000.pth"