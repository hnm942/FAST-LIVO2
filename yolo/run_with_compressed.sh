#!/bin/bash

# Script to run YOLO ROS node with compressed image support
# Usage: ./run_with_compressed.sh [image_topic]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default topic - will auto-detect compressed
IMAGE_TOPIC="${1:-/camera/color/image_raw}"

echo "=========================================="
echo "YOLO ROS Node - Compressed Image Support"
echo "=========================================="
echo "Image topic: $IMAGE_TOPIC"
echo ""
echo "Checking available topics..."
rostopic list | grep -E "(image|camera)" | head -10
echo ""
echo "Publishing to:"
echo "  - /yolo/segmented_image"
echo "  - /yolo/walkability_mask"
echo "  - /yolo/double_view"
echo ""
echo "View: rosrun rqt_image_view rqt_image_view /yolo/double_view"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"

# Check if topic has /compressed suffix
if [[ "$IMAGE_TOPIC" == *"/compressed"* ]]; then
    echo "Detected COMPRESSED image topic"
    python3 yolo_ros_node_compressed.py \
        _image_topic:="$IMAGE_TOPIC" \
        _use_compressed:=true \
        _conf_threshold:=0.25 \
        _device:=""
else
    echo "Using RAW image topic"
    python3 yolo_ros_node_compressed.py \
        _image_topic:="$IMAGE_TOPIC" \
        _use_compressed:=false \
        _conf_threshold:=0.25 \
        _device:=""
fi
