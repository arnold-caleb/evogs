#!/bin/bash
# EvoGS Training Script: Sparse Temporal Supervision
#
# This script trains with sparse temporal supervision, where the velocity field
# is trained on only a subset of frames (e.g., every 2nd frame) and must interpolate
# the remaining frames. This tests the model's ability to learn continuous dynamics.
#
# Usage:
#   ./scripts/train_with_sparse_supervision.sh <dataset_name> <config_file> <stride>
#
# Example:
#   ./scripts/train_with_sparse_supervision.sh cut_roasted_beef arguments/dynerf/cut_roasted_beef_velocity.py 2

set -e

# === ARGUMENT PARSING ===
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dataset_name> <config_file> <temporal_stride>"
    echo ""
    echo "Arguments:"
    echo "  dataset_name      Name of dataset"
    echo "  config_file       Path to configuration file"
    echo "  temporal_stride   Train on every Nth frame (e.g., 2 = every other frame)"
    echo ""
    echo "Example:"
    echo "  $0 cut_roasted_beef arguments/dynerf/cut_roasted_beef_velocity.py 2"
    echo ""
    exit 1
fi

DATASET_NAME="$1"
CONFIG_FILE="$2"
TEMPORAL_STRIDE="$3"

# === LOAD CONFIGURATION ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$SCRIPT_DIR/config_paths.sh" ]; then
    source "$SCRIPT_DIR/config_paths.sh"
else
    echo "ERROR: config_paths.sh not found"
    exit 1
fi

# === SETUP ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}"
OUTPUT_NAME="velocity_sparse${TEMPORAL_STRIDE}/${DATASET_NAME}_${TIMESTAMP}"

# === ENVIRONMENT SETUP ===
if [ -n "$CONDA_ENV_PATH" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_PATH"
fi

# === TRAINING INFO ===
echo "=========================================="
echo "EvoGS Sparse Temporal Supervision Training"
echo "=========================================="
echo "Dataset:         $DATASET_NAME"
echo "Config:          $CONFIG_FILE"
echo "Temporal stride: $TEMPORAL_STRIDE (train on every ${TEMPORAL_STRIDE}th frame)"
echo "Output:          $OUTPUT_NAME"
echo "=========================================="
echo ""
echo "Training strategy:"
echo "  - Supervised frames:   0, $TEMPORAL_STRIDE, $((TEMPORAL_STRIDE*2)), ..."
echo "  - Interpolated frames: Intermediate frames must be predicted by velocity field"
echo "  - This tests continuous dynamics learning"
echo ""

# === RUN TRAINING ===
cd "$PROJECT_ROOT"

python train.py \
    -s "$DATASET_PATH" \
    --port 6018 \
    --expname "$OUTPUT_NAME" \
    --configs "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Sparse supervision training complete"
echo "=========================================="
echo "Model trained on every ${TEMPORAL_STRIDE}th frame"
echo "Evaluate interpolation quality with render.py"

