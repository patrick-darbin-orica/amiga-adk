#!/bin/bash
python main.py --filter-config ./configs/filter_config.json \
 --controller-config ./configs/controller_config.json \
 --tool-config-path ./configs/tool_config.json \
 --waypoints-path ./surveyed-waypoints/2025-08-06_17-57-04_waypoints.json \
 --last-row-waypoint-index 4 \
 --turn-direction left \
 --row-spacing 6.0 \
 --headland-buffer 6.0 \
