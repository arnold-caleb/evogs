#!/bin/bash
# Process all DyNeRF datasets: preprocess, extract images, run COLMAP, downsample
# Usage: bash scripts/process_all_datasets.sh [data_dir]
#   data_dir: path to DyNeRF data root (default: data/dynerf)

cd "$(dirname "$0")/.."

DATA_DIR="${1:-data/dynerf}"

datasets=("coffee_martini" "cook_spinach" "sear_steak" "flame_salmon_1" "flame_steak")

for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing: $dataset"
    echo "========================================"
    
    dataset_path="${DATA_DIR}/$dataset"
    
    # Step 1: Check if images need extraction
    if [ ! -d "$dataset_path/cam00/images" ]; then
        echo "Extracting images for $dataset..."
        python scripts/preprocess_dynerf_simple.py --datadir "$dataset_path"
    else
        echo "Images already extracted for $dataset"
    fi
    
    # Step 2: Convert LLFF poses to COLMAP format
    if [ ! -d "$dataset_path/sparse_" ]; then
        echo "Converting poses for $dataset..."
        python scripts/llff2colmap.py "$dataset_path"
    else
        echo "COLMAP format already exists for $dataset"
    fi
    
    echo "✅ Finished preprocessing $dataset"
    echo "NOTE: Run COLMAP separately: bash colmap_dynerf.sh $dataset_path"
done

echo ""
echo "========================================"
echo "All datasets preprocessed!"
echo "Next step: Run COLMAP with colmap_dynerf.sh or sbatch evogs_scripts/run_colmap_all_datasets.slurm"
echo "========================================"
