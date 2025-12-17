#!/bin/bash

# Change to the waypoint_navigation directory
cd "$(dirname "$0")/.." || exit 1

# Start the alignment service
python -m utils.hole_alignment --canbus-config ./configs/canbus_config.json --model-path ./detection/best1.engine &
ALIGNMENT_PID=$!
echo "✓ Alignment service started (PID: $ALIGNMENT_PID)"

# Give it a moment to initialize
sleep 2