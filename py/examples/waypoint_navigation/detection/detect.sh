#!/bin/bash
source ~/Amiga/venv/bin/activate

echo "Detection script executed in headless mode (no matplotlib display)"
echo "Running with lower priority (nice +10) to avoid CAN bus starvation"

# Preload libgomp to avoid TLS allocation error on ARM platforms
export LD_PRELOAD=/mnt/managed_home/farm-ng-user-patrick-orica/Amiga/venv/lib/python3.8/site-packages/depthai.libs/libgomp-89466.so.1.0.0

DETECTION_HEADLESS=1 nice -n 10 python detectionPlot.py \
