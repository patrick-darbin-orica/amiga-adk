#!/bin/bash
python main.py --filter-config ./configs/filter_config.json \
 --controller-config ./configs/controller_config.json \
 --tool-config-path ./configs/tool_config.json \
 --canbus-config ./configs/canbus_config.json \
 --waypoints-path ./surveyed-waypoints/test1.csv \
 --last-row-waypoint-index 4 \
 --turn-direction left \
 --row-spacing 6.0 \
 --headland-buffer 6.0 \
 --actuator-enabled --actuator-id 0 --actuator-open-seconds 1.5 --actuator-close-seconds 1.5
