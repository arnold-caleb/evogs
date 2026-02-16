#!/bin/bash
# Monitor static training job and automatically run rasterization when complete

JOB_ID=25231144
SCRIPT_DIR="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting"

echo "==================================================================="
echo "MONITORING STATIC TRAINING JOB $JOB_ID"
echo "==================================================================="
echo ""

# Function to check if job is still running
check_job() {
    squeue -u $USER | grep -q $JOB_ID
    return $?
}

# Function to get latest model path
get_latest_model() {
    ls -dt $SCRIPT_DIR/output/static_frame0/cut_roasted_beef_* 2>/dev/null | head -1
}

# Monitor job
echo "Waiting for job to complete..."
while check_job; do
    echo -n "."
    sleep 30
done

echo ""
echo ""
echo "==================================================================="
echo "✓ JOB COMPLETE!"
echo "==================================================================="
echo ""

# Check for errors
ERROR_FILE="$SCRIPT_DIR/slurm_outputs/static_f0_${JOB_ID}.err"
if [ -f "$ERROR_FILE" ] && [ -s "$ERROR_FILE" ]; then
    echo "⚠️  Warning: Error file is not empty"
    echo "Last 20 lines of error log:"
    tail -20 "$ERROR_FILE"
    echo ""
fi

# Get output
OUTPUT_FILE="$SCRIPT_DIR/slurm_outputs/static_f0_${JOB_ID}.out"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Last 30 lines of output:"
    tail -30 "$OUTPUT_FILE"
    echo ""
fi

# Find trained model
MODEL_PATH=$(get_latest_model)
if [ -z "$MODEL_PATH" ]; then
    echo "❌ ERROR: Could not find trained model"
    echo "Expected in: $SCRIPT_DIR/output/static_frame0/"
    exit 1
fi

echo "==================================================================="
echo "FOUND TRAINED MODEL"
echo "==================================================================="
echo "Path: $MODEL_PATH"
echo ""

# Check if model has iteration 15000
if [ ! -d "$MODEL_PATH/point_cloud/iteration_15000" ]; then
    echo "❌ ERROR: iteration_15000 not found"
    echo "Available iterations:"
    ls "$MODEL_PATH/point_cloud/" 2>/dev/null
    exit 1
fi

echo "✓ Found iteration_15000"
echo ""

# Ask user if they want to proceed with rasterization
echo "==================================================================="
echo "READY TO RASTERIZE"
echo "==================================================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Iteration: 15000"
echo ""
echo "This will create HexPlane grid G₀ from trained Gaussians."
echo ""
read -p "Proceed with rasterization? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping rasterization."
    echo ""
    echo "To rasterize manually, run:"
    echo "  python rasterize_frame0_to_grid.py \\"
    echo "    --model_path $MODEL_PATH \\"
    echo "    --iteration 15000 \\"
    echo "    --visualize"
    exit 0
fi

echo ""
echo "==================================================================="
echo "RUNNING RASTERIZATION"
echo "==================================================================="
echo ""

# Activate environment
source /u/aa0008/miniconda/etc/profile.d/conda.sh
conda activate /n/fs/aa-rldiff/.conda_envs/gaussian_splatting_cuda12

# Run rasterization
cd "$SCRIPT_DIR"
python rasterize_frame0_to_grid.py \
    --model_path "$MODEL_PATH" \
    --iteration 15000 \
    --grid_resolution 64 64 64 \
    --grid_channels 32 \
    --multires 1 2 4 \
    --visualize

echo ""
echo "==================================================================="
echo "✅ ALL COMPLETE!"
echo "==================================================================="
echo ""
echo "Next step: Train evolution operator"
echo "Grid G₀ saved at: $MODEL_PATH/hexplane_grid/G0_iter15000.pth"

