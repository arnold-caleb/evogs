#!/bin/bash
# EvoGS Evaluation Script
#
# Evaluates a trained model by rendering test views and computing metrics.
#
# Usage:
#   ./scripts/evaluate_model.sh <model_path>
#
# Example:
#   ./scripts/evaluate_model.sh output/velocity_field/lego_20250209_143022

set -e

# === ARGUMENT PARSING ===
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_path> [--skip_train] [--skip_test]"
    echo ""
    echo "Arguments:"
    echo "  model_path    Path to trained model directory"
    echo ""
    echo "Options:"
    echo "  --skip_train  Skip rendering training views"
    echo "  --skip_test   Skip rendering test views"
    echo ""
    exit 1
fi

MODEL_PATH="$1"
shift

# === OPTIONS ===
SKIP_TRAIN=false
SKIP_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip_test)
            SKIP_TEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# === LOAD CONFIGURATION ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$SCRIPT_DIR/config_paths.sh" ]; then
    source "$SCRIPT_DIR/config_paths.sh"
    if [ -n "$CONDA_ENV_PATH" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_PATH"
    fi
fi

# === VALIDATION ===
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "EvoGS Model Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

# === RENDER VIEWS ===
if [ "$SKIP_TRAIN" = false ]; then
    echo "Rendering training views..."
    python render.py -m "$MODEL_PATH" --skip_test
    echo ""
fi

if [ "$SKIP_TEST" = false ]; then
    echo "Rendering test views..."
    python render.py -m "$MODEL_PATH" --skip_train
    echo ""
fi

# === COMPUTE METRICS ===
echo "Computing metrics..."
python metrics.py -m "$MODEL_PATH"
echo ""

# === SUMMARY ===
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Results saved to: $MODEL_PATH"
echo ""
echo "Check the following files:"
echo "  - $MODEL_PATH/test/renders/      : Rendered images"
echo "  - $MODEL_PATH/results.json       : Quantitative metrics (PSNR, SSIM, LPIPS)"
echo "=========================================="


