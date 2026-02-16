#!/bin/bash
# Setup SAGA data directory (separate from training data)
# Gathers frame 0 from all cameras for multi-view segmentation

set -e

cd /n/fs/aa-rldiff/view_synthesis/gaussian-splatting

echo "===================================="
echo "Setting up SAGA Data Directory"
echo "===================================="
echo ""

# Create SAGA data structure
echo "[1/3] Creating saga_data directory structure..."
mkdir -p saga_data/cut_roasted_beef/images
mkdir -p saga_data/cut_roasted_beef/sam_masks
mkdir -p saga_data/cut_roasted_beef/mask_scales
mkdir -p saga_data/cut_roasted_beef/features
echo "✓ Created saga_data/cut_roasted_beef/{images,sam_masks,mask_scales,features}/"
echo ""

# Gather frame 0 from all cameras
echo "[2/3] Gathering frame 0 from all cameras..."
count=0
for i in {0..20}; do
    cam=$(printf "cam%02d" $i)
    src_img="data/dynerf/cut_roasted_beef/${cam}/images/0000.png"
    
    if [ -f "$src_img" ]; then
        dest_img="saga_data/cut_roasted_beef/images/${cam}_0000.png"
        ln -sf "$(realpath $src_img)" "$dest_img"
        count=$((count + 1))
        echo "  ✓ Linked ${cam}/images/0000.png"
    fi
done
echo ""
echo "Total images gathered: $count"
echo ""

# Link COLMAP sparse data (SAGA needs this for camera info)
echo "[3/3] Linking COLMAP sparse data..."
if [ -d "data/dynerf/cut_roasted_beef/sparse_" ]; then
    ln -sf "$(realpath data/dynerf/cut_roasted_beef/sparse_)" saga_data/cut_roasted_beef/sparse
    echo "✓ Linked sparse reconstruction"
elif [ -d "data/dynerf/cut_roasted_beef/colmap/sparse/0" ]; then
    ln -sf "$(realpath data/dynerf/cut_roasted_beef/colmap/sparse/0)" saga_data/cut_roasted_beef/sparse
    echo "✓ Linked COLMAP sparse data"
else
    echo "⚠️  Warning: Could not find sparse COLMAP data"
    echo "   SAGA may need this for camera intrinsics"
fi
echo ""

echo "===================================="
echo "Setup Complete!"
echo "===================================="
echo ""
echo "Directory structure:"
echo "  saga_data/cut_roasted_beef/"
echo "    ├── images/       ($count images from frame 0)"
echo "    └── sparse/       (COLMAP data)"
echo ""
echo "Next steps:"
echo "  cd saga"
echo "  DATA=\"../saga_data/cut_roasted_beef\""
echo "  python extract_segment_everything_masks.py --image_root \$DATA ..."
echo ""

