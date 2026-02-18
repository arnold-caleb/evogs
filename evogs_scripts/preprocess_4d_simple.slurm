#!/bin/bash
#SBATCH --job-name=preprocess_4d_simple
#SBATCH --output=slurm_outputs/preprocess_4d_simple_%j.out
#SBATCH --error=slurm_outputs/preprocess_4d_simple_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

# === Load user config ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config_paths.sh" 2>/dev/null || {
    echo "ERROR: evogs_scripts/config_paths.sh not found."
    echo "  cp evogs_scripts/config_paths.sh.example evogs_scripts/config_paths.sh"
    exit 1
}

echo "=== 4D DATASET PREPROCESSING ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader,nounits

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

cd "$EVOGS_ROOT"

# ============================================
# CONFIGURATION — edit for your scene
# ============================================
SCENE="cut_roasted_beef"

echo ""
echo "=== EXTRACTING FRAMES ==="
python scripts/preprocess_dynerf_simple.py --datadir "${DYNERF_DATA}/${SCENE}"

echo ""
echo "=== CHECKING RESULTS ==="
SCENE_DIR="${DYNERF_DATA}/${SCENE}"
if [ -d "${SCENE_DIR}/cam00/images" ]; then
    echo "Frames extracted successfully"
    for cam_dir in ${SCENE_DIR}/cam*/images; do
        if [ -d "$cam_dir" ]; then
            frame_count=$(ls "$cam_dir" | wc -l)
            echo "  $(basename $(dirname $cam_dir)): $frame_count frames"
        fi
    done
else
    echo "Frame extraction failed"
fi

echo ""
echo "=== COMPLETE ==="
echo "Dataset: ${SCENE_DIR}/"
