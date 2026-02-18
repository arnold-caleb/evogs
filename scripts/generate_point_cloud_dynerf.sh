#!/bin/bash
# Generate dense point cloud for DyNeRF dataset using COLMAP

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/generate_point_cloud_dynerf.sh <dataset_path>"
    echo "Example: bash scripts/generate_point_cloud_dynerf.sh data/dynerf/cut_roasted_beef"
    exit 1
fi

echo "========================================"
echo "GENERATING DENSE POINT CLOUD FOR DYNERF"
echo "========================================"
echo "Dataset: $DATASET"
echo ""

cd "$(dirname "$0")/.."

# Check if COLMAP images exist
if [ ! -d "$DATASET/image_colmap" ]; then
    echo "❌ Error: $DATASET/image_colmap not found"
    echo "Run preprocessing first!"
    exit 1
fi

echo "✓ Found COLMAP images: $(ls $DATASET/image_colmap/*.png | wc -l) images"

# Check if sparse reconstruction exists
if [ ! -d "$DATASET/sparse_" ]; then
    echo "❌ Error: $DATASET/sparse_ not found"
    echo "Sparse reconstruction missing!"
    exit 1
fi

echo "✓ Found sparse reconstruction"
echo ""

# Step 1: Run COLMAP feature extraction and matching
echo "=== STEP 1: COLMAP Feature Extraction ==="
mkdir -p $DATASET/colmap
cp -r $DATASET/image_colmap $DATASET/colmap/images
cp -r $DATASET/sparse_ $DATASET/colmap/sparse_custom

echo "Running feature extraction..."
colmap feature_extractor \
    --database_path $DATASET/colmap/database.db \
    --image_path $DATASET/colmap/images \
    --SiftExtraction.max_image_size 4096 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

echo ""
echo "=== STEP 2: Update Database ==="
python database.py \
    --database_path $DATASET/colmap/database.db \
    --txt_path $DATASET/colmap/sparse_custom/cameras.txt

echo ""
echo "=== STEP 3: COLMAP Matching ==="
colmap exhaustive_matcher --database_path $DATASET/colmap/database.db

echo ""
echo "=== STEP 4: Point Triangulation ==="
mkdir -p $DATASET/colmap/sparse/0
colmap point_triangulator \
    --database_path $DATASET/colmap/database.db \
    --image_path $DATASET/colmap/images \
    --input_path $DATASET/colmap/sparse_custom \
    --output_path $DATASET/colmap/sparse/0 \
    --clear_points 1

echo ""
echo "=== STEP 5: Dense Reconstruction ==="
mkdir -p $DATASET/colmap/dense/workspace

echo "Undistorting images..."
colmap image_undistorter \
    --image_path $DATASET/colmap/images \
    --input_path $DATASET/colmap/sparse/0 \
    --output_path $DATASET/colmap/dense/workspace

echo "Running patch match stereo (this takes time)..."
colmap patch_match_stereo --workspace_path $DATASET/colmap/dense/workspace

echo "Fusing stereo results into dense point cloud..."
colmap stereo_fusion \
    --workspace_path $DATASET/colmap/dense/workspace \
    --output_path $DATASET/colmap/dense/workspace/fused.ply

# Check if fused.ply was created
if [ -f "$DATASET/colmap/dense/workspace/fused.ply" ]; then
    echo ""
    echo "✅ Dense point cloud created!"
    echo "Location: $DATASET/colmap/dense/workspace/fused.ply"
    
    # Count points
    num_points=$(grep "element vertex" $DATASET/colmap/dense/workspace/fused.ply | awk '{print $3}')
    echo "Number of points: $num_points"
    
    # Step 6: Downsample to ~40K points
    echo ""
    echo "=== STEP 6: Downsampling Point Cloud ==="
    python scripts/downsample_point.py \
        $DATASET/colmap/dense/workspace/fused.ply \
        $DATASET/points3D_downsample2.ply
    
    if [ -f "$DATASET/points3D_downsample2.ply" ]; then
        echo ""
        echo "✅ ALL DONE!"
        echo "Final point cloud: $DATASET/points3D_downsample2.ply"
        num_final=$(grep "element vertex" $DATASET/points3D_downsample2.ply | awk '{print $3}')
        echo "Final point count: $num_final (downsampled from $num_points)"
    else
        echo "❌ Downsampling failed"
        exit 1
    fi
else
    echo "❌ Dense reconstruction failed - fused.ply not created"
    exit 1
fi

