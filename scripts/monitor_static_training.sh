#!/bin/bash
# Monitor static training job and show densification progress

JOB_ID=25231473
LOG_FILE="/n/fs/aa-rldiff/view_synthesis/gaussian-splatting/slurm_outputs/static_f0_${JOB_ID}.err"

echo "Monitoring Job $JOB_ID..."
echo "============================================================"

# Check job status
while squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; do
    echo -e "\n[$(date '+%H:%M:%S')] Job still running..."
    
    # Show latest densification debug messages
    if [ -f "$LOG_FILE" ]; then
        echo "Latest densification stats:"
        grep "DENSIFY DEBUG" "$LOG_FILE" | tail -5
    fi
    
    sleep 120  # Check every 2 minutes
done

echo -e "\n============================================================"
echo "Training COMPLETED at $(date)"
echo "============================================================"

# Show final results
if [ -f "$LOG_FILE" ]; then
    echo -e "\nFinal Densification Stats:"
    grep "DENSIFY DEBUG" "$LOG_FILE" | tail -10
    
    echo -e "\nFinal PSNR:"
    grep -E "ITER.*30000.*Evaluating|psnr" "$LOG_FILE" | tail -5
fi

# Count final Gaussians
MODEL_DIR=$(ls -td /n/fs/aa-rldiff/view_synthesis/gaussian-splatting/output/static_frame0_hq/cut_roasted_beef_* | head -1)
if [ -d "$MODEL_DIR/point_cloud/iteration_30000" ]; then
    PLY_FILE="$MODEL_DIR/point_cloud/iteration_30000/point_cloud.ply"
    GAUSSIAN_COUNT=$(($(wc -l < "$PLY_FILE") - 13))
    echo -e "\nFinal Gaussian Count: $GAUSSIAN_COUNT"
fi

echo -e "\nDone! Check results at: $MODEL_DIR"

