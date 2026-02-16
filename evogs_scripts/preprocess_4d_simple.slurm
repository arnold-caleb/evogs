#!/bin/bash
#SBATCH --partition=visualai
#SBATCH --account=visualai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --job-name=preprocess_4d_simple
#SBATCH -o slurm_outputs/preprocess_4d_simple_%j.out 
#SBATCH -e slurm_outputs/preprocess_4d_simple_%j.err 
#SBATCH --mail-type=all
#SBATCH --mail-user=aa0008@cs.princeton.edu

echo "=== 4D DATASET PREPROCESSING ==="
hostname
nvidia-smi --query-gpu=name --format=csv,noheader,nounits

source ~/miniconda/etc/profile.d/conda.sh

export TMPDIR=/n/fs/aa-rldiff/tmp
export CONDA_PKGS_DIRS=/n/fs/aa-rldiff/.conda_pkgs
export CONDA_ENVS_PATH=/n/fs/aa-rldiff/.conda_envs
export XDG_CACHE_HOME=/n/fs/aa-rldiff/.cache
export PIP_CACHE_DIR=/n/fs/aa-rldiff/.pip_cache

cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

echo ""
echo "=== ACTIVATING ENVIRONMENT ==="
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

echo ""
echo "=== EXTRACTING FRAMES ==="
python scripts/preprocess_dynerf_simple.py --datadir data/dynerf/cut_roasted_beef

echo ""
echo "=== CHECKING RESULTS ==="
if [ -d "data/dynerf/cut_roasted_beef/cam00/images" ]; then
    echo "Frames extracted successfully"
    for cam_dir in data/dynerf/cut_roasted_beef/cam*/images; do
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
echo "Dataset: data/dynerf/cut_roasted_beef/"