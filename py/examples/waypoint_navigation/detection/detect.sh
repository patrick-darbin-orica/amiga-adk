#!/bin/bash
source ~/Amiga/venv/bin/activate

echo "Detection script executed in headless mode (no matplotlib display)"
echo "Running with lower priority (nice +10) to avoid CAN bus starvation"
DETECTION_HEADLESS=1 nice -n 10 python detectionPlot.py \
