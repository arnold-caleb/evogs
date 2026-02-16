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


echo "=== TRAINING 4-ANCHOR STATIC GAUSSIANS (FRAMES 0, 75, 150, 225) ==="
echo "=== SYSTEM INFORMATION ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12
echo "Environment activated"

# Set memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Go to project directory
cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/data/dynerf/cut_roasted_beef"
CONFIG="arguments/static/cut_roasted_beef_frame0_hq.py"  # Using HQ config
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="output/static_4anchors"

echo ""
echo "=== 4-ANCHOR TRAINING CONFIGURATION ==="
echo "Data: $DATA_DIR"
echo "Config: $CONFIG"
echo "Base Output: $BASE_OUTPUT_DIR"
echo "Training 4 anchor frames:"
echo "  • Frame 0   (t=0.00) - CANONICAL"
echo "  • Frame 75  (t=0.25)"
echo "  • Frame 150 (t=0.50)"
echo "  • Frame 225 (t=0.75)"
echo "Each model: 30K iterations"
echo "Forward-only integration strategy"
echo ""

# ============================================
# TRAINING FRAME 0 (t=0.0, CANONICAL START)
# ============================================
echo "=== [1/4] TRAINING ANCHOR FRAME 0 (t=0.0, CANONICAL) ==="
OUTPUT_DIR_F0="${BASE_OUTPUT_DIR}/frame0_${TIMESTAMP}"

python train_anchor.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F0" \
    --configs "$CONFIG" \
    --dataset_type "dynerf_static" \
    --frame_idx 0 \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

echo ""
echo "✓ Frame 0 training complete: $OUTPUT_DIR_F0"
echo ""
echo "=== RENDERING FRAME 0 ==="
python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F0" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet
echo "✓ Frame 0 renders saved at: $OUTPUT_DIR_F0/renders_iter30000/"
echo ""

# ============================================
# TRAINING FRAME 75 (t=0.25) - FROM CANONICAL
# ============================================
echo "=== [2/4] TRAINING ANCHOR FRAME 75 (t=0.25) FROM CANONICAL ==="
OUTPUT_DIR_F75="${BASE_OUTPUT_DIR}/frame75_${TIMESTAMP}"
CANONICAL_CHECKPOINT="${OUTPUT_DIR_F0}/chkpnt30000.pth"

echo "Loading canonical Gaussians from: $CANONICAL_CHECKPOINT"
echo "This ensures same Gaussian count and correspondence!"

python train_anchor.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F75" \
    --configs "arguments/static/cut_roasted_beef_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 75 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

echo ""
echo "✓ Frame 75 training complete: $OUTPUT_DIR_F75"
echo ""
echo "=== RENDERING FRAME 75 ==="
python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F75" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet
echo "✓ Frame 75 renders saved at: $OUTPUT_DIR_F75/renders_iter30000/"
echo ""

# ============================================
# TRAINING FRAME 150 (t=0.5, MIDDLE) - FROM CANONICAL
# ============================================
echo "=== [3/4] TRAINING ANCHOR FRAME 150 (t=0.5) FROM CANONICAL ==="
OUTPUT_DIR_F150="${BASE_OUTPUT_DIR}/frame150_${TIMESTAMP}"
CANONICAL_CHECKPOINT="${OUTPUT_DIR_F0}/chkpnt30000.pth"

echo "Loading canonical Gaussians from: $CANONICAL_CHECKPOINT"
echo "This ensures same Gaussian count and correspondence!"

python train_anchor.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F150" \
    --configs "arguments/static/cut_roasted_beef_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 150 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

echo ""
echo "✓ Frame 150 training complete: $OUTPUT_DIR_F150"
echo ""
echo "=== RENDERING FRAME 150 ==="
python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F150" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet
echo "✓ Frame 150 renders saved at: $OUTPUT_DIR_F150/renders_iter30000/"
echo ""

# ============================================
# TRAINING FRAME 225 (t=0.75) - FROM CANONICAL
# ============================================
echo "=== [4/4] TRAINING ANCHOR FRAME 225 (t=0.75) FROM CANONICAL ==="
OUTPUT_DIR_F225="${BASE_OUTPUT_DIR}/frame225_${TIMESTAMP}"

echo "Loading canonical Gaussians from: $CANONICAL_CHECKPOINT"
echo "This ensures same Gaussian count and correspondence!"

python train_anchor.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F225" \
    --configs "arguments/static/cut_roasted_beef_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 225 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

echo ""
echo "✓ Frame 225 training complete: $OUTPUT_DIR_F225"
echo ""
echo "=== RENDERING FRAME 225 ==="
python render_static_frame0.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F225" \
    --iteration 30000 \
    --configs "$CONFIG" \
    --quiet
echo "✓ Frame 225 renders saved at: $OUTPUT_DIR_F225/renders_iter30000/"
echo ""

# ============================================
# SUMMARY
# ============================================
echo "=== ALL 4 ANCHOR TRAINING & RENDERING COMPLETE ==="
echo ""
echo "Anchor checkpoints (WITH GAUSSIAN CORRESPONDENCE):"
echo "  Frame 0   (t=0.00): $OUTPUT_DIR_F0/chkpnt30000.pth"
echo "  Frame 75  (t=0.25): $OUTPUT_DIR_F75/chkpnt30000.pth"
echo "  Frame 150 (t=0.50): $OUTPUT_DIR_F150/chkpnt30000.pth"
echo "  Frame 225 (t=0.75): $OUTPUT_DIR_F225/chkpnt30000.pth"
echo ""
echo "Renders:"
echo "  Frame 0:   $OUTPUT_DIR_F0/renders_iter30000/"
echo "  Frame 75:  $OUTPUT_DIR_F75/renders_iter30000/"
echo "  Frame 150: $OUTPUT_DIR_F150/renders_iter30000/"
echo "  Frame 225: $OUTPUT_DIR_F225/renders_iter30000/"
echo ""
echo "✓ Gaussian correspondence: All anchors have SAME Gaussian count"
echo "✓ Gaussian #i at any frame refers to the same physical point"
echo ""
echo "Next step: Train velocity field with FORWARD-ONLY integration"
echo "  Integration strategy:"
echo "    t ∈ [0.00, 0.25) → integrate forward from t=0"
echo "    t ∈ [0.25, 0.50) → integrate forward from t=0.25"
echo "    t ∈ [0.50, 0.75) → integrate forward from t=0.5"
echo "    t ∈ [0.75, 1.00] → integrate forward from t=0.75"
echo "  Benefits:"
echo "    • No backward integration confusion"
echo "    • Maximum integration path = 0.25 time units"
echo "    • Cleaner transitions at anchor boundaries"
echo ""
echo "=== ALL DONE ==="

