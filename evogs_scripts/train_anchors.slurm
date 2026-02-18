#!/bin/bash
#SBATCH --job-name=train_anchors
#SBATCH --output=slurm_outputs/train_anchors_%j.out
#SBATCH --error=slurm_outputs/train_anchors_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== TRAINING 4-ANCHOR STATIC GAUSSIANS (FRAMES 0, 75, 150, 225) ==="
hostname
nvidia-smi

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$EVOGS_ROOT"

# ============================================
# CONFIGURATION — edit for your scene
# ============================================
SCENE="cut_roasted_beef"
DATA_DIR="${DYNERF_DATA}/${SCENE}"
CONFIG="arguments/static/${SCENE}_frame0_hq.py"
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

echo "✓ Frame 0 training complete: $OUTPUT_DIR_F0"
echo ""

# ============================================
# TRAINING FRAME 75 (t=0.25) - FROM CANONICAL
# ============================================
echo "=== [2/4] TRAINING ANCHOR FRAME 75 (t=0.25) FROM CANONICAL ==="
OUTPUT_DIR_F75="${BASE_OUTPUT_DIR}/frame75_${TIMESTAMP}"
CANONICAL_CHECKPOINT="${OUTPUT_DIR_F0}/chkpnt30000.pth"

echo "Loading canonical Gaussians from: $CANONICAL_CHECKPOINT"

python train_anchor.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F75" \
    --configs "arguments/static/${SCENE}_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 75 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

echo "✓ Frame 75 training complete: $OUTPUT_DIR_F75"
echo ""

# ============================================
# TRAINING FRAME 150 (t=0.5) - FROM CANONICAL
# ============================================
echo "=== [3/4] TRAINING ANCHOR FRAME 150 (t=0.5) FROM CANONICAL ==="
OUTPUT_DIR_F150="${BASE_OUTPUT_DIR}/frame150_${TIMESTAMP}"

python train_anchor.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F150" \
    --configs "arguments/static/${SCENE}_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 150 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

echo "✓ Frame 150 training complete: $OUTPUT_DIR_F150"
echo ""

# ============================================
# TRAINING FRAME 225 (t=0.75) - FROM CANONICAL
# ============================================
echo "=== [4/4] TRAINING ANCHOR FRAME 225 (t=0.75) FROM CANONICAL ==="
OUTPUT_DIR_F225="${BASE_OUTPUT_DIR}/frame225_${TIMESTAMP}"

python train_anchor.py \
    --source_path "$DATA_DIR" \
    --model_path "$OUTPUT_DIR_F225" \
    --configs "arguments/static/${SCENE}_anchor.py" \
    --dataset_type "dynerf_static" \
    --frame_idx 225 \
    --canonical_checkpoint "$CANONICAL_CHECKPOINT" \
    --eval \
    --iterations 30000 \
    --checkpoint_iterations 30000

echo "✓ Frame 225 training complete: $OUTPUT_DIR_F225"
echo ""

# ============================================
# SUMMARY
# ============================================
echo "=== ALL 4 ANCHOR TRAINING COMPLETE ==="
echo ""
echo "Anchor checkpoints (WITH GAUSSIAN CORRESPONDENCE):"
echo "  Frame 0   (t=0.00): $OUTPUT_DIR_F0/chkpnt30000.pth"
echo "  Frame 75  (t=0.25): $OUTPUT_DIR_F75/chkpnt30000.pth"
echo "  Frame 150 (t=0.50): $OUTPUT_DIR_F150/chkpnt30000.pth"
echo "  Frame 225 (t=0.75): $OUTPUT_DIR_F225/chkpnt30000.pth"
echo ""
echo "Next: Update anchor_checkpoints in your scene config, then train the velocity field."
echo "=== ALL DONE ==="
