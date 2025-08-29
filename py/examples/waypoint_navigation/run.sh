#!/bin/bash
source ~/farm-ng-amiga/venv/bin/activate
python main.py \
 --config ./configs/config.json \
 --tool-config-path ./configs/tool_config.json \
 --waypoints-path ./surveyed-waypoints/physStraight.csv \
 --last-row-waypoint-index 4 \
 --turn-direction left \
 --row-spacing 6.0 \
 --headland-buffer 6.0 \
 --actuator-enabled --actuator-id 0 --actuator-open-seconds 6 --actuator-close-seconds 7
