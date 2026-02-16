#!/bin/bash
# Create a spiral video that loops 5 times

RENDER_DIR="$1"
VIDEO_DIR="$(dirname $RENDER_DIR)/videos"
mkdir -p "$VIDEO_DIR"

FPS=10
LOOPS=5

echo "Creating 5x looped spiral video..."
echo "  Render dir: $RENDER_DIR"
echo "  FPS: $FPS"
echo "  Loops: $LOOPS"

# Create concat file listing each image 5 times
CONCAT_FILE="$VIDEO_DIR/concat_5x.txt"
> "$CONCAT_FILE"

# Loop 5 times
for loop_num in $(seq 1 $LOOPS); do
    echo "  Loop $loop_num/5..." >&2
    for img in "$RENDER_DIR"/render_*.png; do
        echo "file '$img'" >> "$CONCAT_FILE"
        echo "duration 0.1" >> "$CONCAT_FILE"  # 1/FPS duration
    done
done

# Add last frame (required by concat demuxer)
LAST_IMG=$(ls "$RENDER_DIR"/render_*.png | tail -1)
echo "file '$LAST_IMG'" >> "$CONCAT_FILE"

# Create video
ffmpeg -y -f concat -safe 0 -i "$CONCAT_FILE" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -r $FPS \
    "$VIDEO_DIR/spiral_5x.mp4" 2>&1 | grep -E "frame=|video:|Duration"

rm -f "$CONCAT_FILE"

echo ""
echo "✓ Created: $VIDEO_DIR/spiral_5x.mp4"
ls -lh "$VIDEO_DIR"/spiral_5x.mp4

