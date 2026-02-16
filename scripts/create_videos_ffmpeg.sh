#!/bin/bash
# Create visualization videos using ffmpeg

RENDER_DIR="$1"
FPS="${2:-10}"

if [ -z "$RENDER_DIR" ]; then
    echo "Usage: $0 <render_dir> [fps]"
    exit 1
fi

VIDEO_DIR="$(dirname $RENDER_DIR)/videos"
mkdir -p "$VIDEO_DIR"

echo "======================================================================"
echo "CREATING VISUALIZATION VIDEOS"
echo "======================================================================"
echo "Input: $RENDER_DIR"
echo "Output: $VIDEO_DIR"
echo "FPS: $FPS"
echo ""

# 1. Spiral video - rotating through all viewpoints (5 loops)
echo "1. Creating spiral viewpoint video (5 loops)..."
# Create file list for looping
LOOP_LIST="$VIDEO_DIR/loop_list.txt"
> "$LOOP_LIST"  # Clear file

# Add all frames 5 times
for loop in {1..5}; do
    for f in "$RENDER_DIR"/render_*.png; do
        echo "file '$f'" >> "$LOOP_LIST"
    done
done

ffmpeg -y -f concat -safe 0 -r $FPS -i "$LOOP_LIST" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "$VIDEO_DIR/spiral_viewpoints.mp4" 2>&1 | grep -E "frame=|video:" || true
rm -f "$LOOP_LIST"
echo "   ✓ spiral_viewpoints.mp4 (5 complete rotations)"

# 2. Create comparison frames (GT vs Rendered side-by-side)
echo ""
echo "2. Creating GT vs Rendered comparison frames..."
COMP_DIR="$VIDEO_DIR/comparison_frames"
mkdir -p "$COMP_DIR"

for i in $(seq -f "%03g" 0 4); do
    GT="$RENDER_DIR/gt_$i.png"
    RENDER="$RENDER_DIR/render_$i.png"
    OUTPUT="$COMP_DIR/compare_$i.png"
    
    if [ -f "$GT" ] && [ -f "$RENDER" ]; then
        # Use ImageMagick or ffmpeg to create side-by-side
        ffmpeg -y -i "$GT" -i "$RENDER" \
            -filter_complex "[0:v][1:v]hstack=inputs=2" \
            "$OUTPUT" 2>/dev/null
    fi
done

# Create video from comparison frames
ffmpeg -y -framerate $FPS -pattern_type glob -i "$COMP_DIR/compare_*.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "$VIDEO_DIR/gt_vs_rendered_comparison.mp4" 2>&1 | grep -E "frame=|video:" || true
echo "   ✓ gt_vs_rendered_comparison.mp4"

# 3. Loop video (forward + backward for smooth loop)
echo ""
echo "3. Creating smooth loop video..."
# Create file list for concat
FILELIST="$VIDEO_DIR/filelist.txt"
> "$FILELIST"  # Clear file

# Forward
for f in "$RENDER_DIR"/render_*.png; do
    echo "file '$f'" >> "$FILELIST"
done
# Backward (skip first and last to avoid duplicate)
for f in $(ls -r "$RENDER_DIR"/render_*.png | tail -n +2 | head -n -1); do
    echo "file '$f'" >> "$FILELIST"
done

ffmpeg -y -f concat -safe 0 -r $FPS -i "$FILELIST" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "$VIDEO_DIR/spiral_loop.mp4" 2>&1 | grep -E "frame=|video:" || true
echo "   ✓ spiral_loop.mp4 (smooth ping-pong loop)"

rm -rf "$COMP_DIR" "$FILELIST"

echo ""
echo "======================================================================"
echo "✅ VIDEOS CREATED!"
echo "======================================================================"
echo ""
ls -lh "$VIDEO_DIR"/*.mp4
echo ""
echo "Videos ready to view in: $VIDEO_DIR"

