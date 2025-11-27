#!/bin/bash
source ~/farm-ng-amiga/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamped log filename
LOG_FILE="logs/navigation_$(date +%Y%m%d_%H%M%S).log"

# Run with output to both terminal and log file
python main.py \
 --config ./configs/config.json \
 --tool-config-path ./configs/tool_config.json \
 --waypoints-path ./surveyed-waypoints/deerPark3x2.csv \
 --last-row-waypoint-index 3 \
 --turn-direction left \
 --row-spacing 6 \
 --headland-buffer 2.0 \
 --return-to-start 1 \
 --actuator-enabled --actuator-id 0 --actuator-rate-hz 5.0 2>&1 | tee "$LOG_FILE"

echo "Log saved to: $LOG_FILE"
