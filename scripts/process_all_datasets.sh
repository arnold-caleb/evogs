#!/bin/bash
# Process all DyNeRF datasets: preprocess, extract images, run COLMAP, downsample

cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

# Source conda environment
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

datasets=("coffee_martini" "cook_spinach" "sear_steak" "flame_salmon_1" "flame_steak")

for dataset in "${datasets[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing: $dataset"
    echo "========================================"
    
    dataset_path="/n/fs/visualai-scr/Data/dynerf/$dataset"
    
    # Step 1: Check if images need extraction
    if [ ! -d "$dataset_path/cam00/images" ]; then
        echo "Extracting images for $dataset..."
        python scripts/convert_videos_to_images.py --datasets $dataset --data_dir /n/fs/visualai-scr/Data/dynerf
    else
        echo "Images already extracted for $dataset"
    fi
    
    # Step 2: Create symlink if not exists
    if [ ! -L "data/dynerf/$dataset" ]; then
        echo "Creating symlink for $dataset..."
        ln -s /n/fs/visualai-scr/Data/dynerf/$dataset data/dynerf/$dataset
    else
        echo "Symlink already exists for $dataset"
    fi
    
    # Step 3: Run preprocessing
    echo "Running preprocessing for $dataset..."
    python scripts/preprocess_dynerf.py --datadir data/dynerf/$dataset
    
    echo "✅ Finished preprocessing $dataset"
    echo "NOTE: Run COLMAP separately using: sbatch slurm_scripts/run_colmap_reconstruction.slurm (modify WORKDIR)"
done

echo ""
echo "========================================"
echo "All datasets processed!"
echo "========================================"

