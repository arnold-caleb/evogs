#!/bin/bash
# Helper script to list all trained models and their checkpoints

echo "======================================"
echo "TRAINED MODELS SUMMARY"
echo "======================================"
echo ""

cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

# Find all output directories with checkpoints
for exp_dir in output/*/cut_roasted_beef*; do
    if [ -d "$exp_dir/point_cloud" ]; then
        echo "📁 $exp_dir"
        
        # List checkpoints
        checkpoints=$(ls "$exp_dir/point_cloud" 2>/dev/null | grep -E "iteration_[0-9]+" | sort -V)
        
        if [ -n "$checkpoints" ]; then
            echo "   Checkpoints:"
            for ckpt in $checkpoints; do
                iter=$(echo $ckpt | grep -oP 'iteration_\K[0-9]+')
                size=$(du -sh "$exp_dir/point_cloud/$ckpt" 2>/dev/null | cut -f1)
                echo "      ✓ Iteration $iter ($size)"
            done
        else
            echo "   ⚠ No checkpoints found"
        fi
        
        # Check if already rendered
        if [ -d "$exp_dir/video" ]; then
            videos=$(find "$exp_dir/video" -name "*.mp4" | wc -l)
            echo "   Videos: $videos rendered"
        fi
        
        echo ""
    fi
done

echo "======================================"
echo "LATEST MODELS:"
echo "======================================"
ls -lth output/*/cut_roasted_beef*/point_cloud/iteration_* | head -10 | while read line; do
    echo "$line" | awk '{print $9}' | sed 's|output/||' | sed 's|/point_cloud/iteration_| → iteration |'
done
echo ""

echo "======================================"
echo "TO RENDER A MODEL:"
echo "======================================"
echo "1. Edit slurm_scripts/render_hq.slurm"
echo "2. Set these variables (lines 33-36):"
echo "   EXP_NAME='dynerf_hq'"
echo "   SCENE_NAME='cut_roasted_beef_20251015_020351'"
echo "   ITERATION=40000"
echo "   CONFIG='arguments/dynerf/cut_roasted_beef_hq.py'"
echo "3. Run: sbatch slurm_scripts/render_hq.slurm"

