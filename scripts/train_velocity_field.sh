#!/bin/bash
# EvoGS Training Script: Velocity Field (Neural ODE)
# 
# This script trains a dynamic 3D Gaussian Splatting model using velocity fields
# to represent scene dynamics as continuous ODEs: dx/dt = v(x,t)
#
# Usage:
#   ./scripts/train_velocity_field.sh <dataset_name> <config_file> [options]
#
# Example:
#   ./scripts/train_velocity_field.sh cut_roasted_beef arguments/dynerf/cut_roasted_beef_velocity.py

set -e  # Exit on error

# === ARGUMENT PARSING ===
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <config_file> [--output_dir OUTPUT_DIR] [--iterations N]"
    echo ""
    echo "Arguments:"
    echo "  dataset_name    Name of dataset (e.g., cut_roasted_beef)"
    echo "  config_file     Path to configuration file (e.g., arguments/dynerf/cut_roasted_beef_velocity.py)"
    echo ""
    echo "Options:"
    echo "  --output_dir    Custom output directory (default: output/velocity_field/DATASET_NAME)"
    echo "  --iterations    Number of training iterations (default: from config)"
    echo "  --port          Port for visualization server (default: 6017)"
    echo ""
    exit 1
fi

DATASET_NAME="$1"
CONFIG_FILE="$2"
shift 2

# === LOAD CONFIGURATION ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$SCRIPT_DIR/config_paths.sh" ]; then
    source "$SCRIPT_DIR/config_paths.sh"
else
    echo "ERROR: config_paths.sh not found. Please create it from config_paths.sh.template"
    exit 1
fi

# === DEFAULT VALUES ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_ROOT}/velocity_field/${DATASET_NAME}_${TIMESTAMP}"
ITERATIONS=""
PORT=6017

# === PARSE OPTIONS ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# === DATASET PATH RESOLUTION ===
# Try to find dataset in DATA_ROOT
DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please check your DATA_ROOT in config_paths.sh or provide absolute path"
    exit 1
fi

# === ENVIRONMENT SETUP ===
if [ -n "$CONDA_ENV_PATH" ]; then
    echo "Activating conda environment: $CONDA_ENV_PATH"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_PATH"
fi

# === PRE-TRAINING CHECKS ===
echo "=========================================="
echo "EvoGS Velocity Field Training"
echo "=========================================="
echo "Dataset:      $DATASET_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Config file:  $CONFIG_FILE"
echo "Output dir:   $OUTPUT_DIR"
echo "Port:         $PORT"
if [ -n "$ITERATIONS" ]; then
    echo "Iterations:   $ITERATIONS"
fi
echo "=========================================="
echo ""

# Verify dataset structure
if [ ! -f "$DATASET_PATH/poses_bounds.npy" ]; then
    echo "WARNING: poses_bounds.npy not found in dataset"
    echo "This may indicate an incorrect dataset path or format"
fi

# Verify config file exists
if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $PROJECT_ROOT/$CONFIG_FILE"
    exit 1
fi

# === TRAINING ===
echo "Starting training..."
echo ""

cd "$PROJECT_ROOT"

TRAIN_CMD="python train.py \
    -s $DATASET_PATH \
    --port $PORT \
    --expname velocity_field/${DATASET_NAME}_${TIMESTAMP} \
    --configs $CONFIG_FILE"

# Add optional arguments
if [ -n "$ITERATIONS" ]; then
    TRAIN_CMD="$TRAIN_CMD --iterations $ITERATIONS"
fi

# Run training
eval $TRAIN_CMD

# === POST-TRAINING SUMMARY ===
echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Evaluate model: python render.py -m $OUTPUT_DIR"
echo "  2. Compute metrics: python metrics.py -m $OUTPUT_DIR"
echo "  3. Visualize results: Check output/velocity_field/ directory"
echo "=========================================="

