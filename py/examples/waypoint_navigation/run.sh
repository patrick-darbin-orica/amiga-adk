#!/bin/bash
source ~/Amiga/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamped log filename
LOG_FILE="logs/navigation_$(date +%Y%m%d_%H%M%S).log"

# Set waypoint CSV for Flask GUI to match navigation
export WAYPOINT_CSV="deerPark.csv"

# Parse command-line arguments
SKIP_ALIGNMENT=false
if [[ "$1" == "--no-alignment" ]] || [[ "$1" == "--skip-alignment" ]]; then
    SKIP_ALIGNMENT=true
    echo "⚠️  Hole alignment will be DISABLED"
else
    echo "✓ Hole alignment is ENABLED (use --no-alignment to disable)"
fi

# Start hole alignment service in background (unless skipped)
if [ "$SKIP_ALIGNMENT" = false ]; then
    echo "🚀 Starting hole alignment service in background..."
    python -m utils.hole_alignment \
        --canbus-config ./configs/canbus_config.json \
        --model-path ./detection/best1.engine \
        > logs/alignment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    ALIGNMENT_PID=$!
    echo "✓ Alignment service started (PID: $ALIGNMENT_PID)"

    # Give it a moment to initialize
    sleep 2
else
    echo "⏭  Skipping hole alignment service"
fi

# Build main.py command with optional --no-hole-alignment flag
MAIN_CMD="python main.py \
 --config ./configs/config.json \
 --tool-config-path ./configs/tool_config.json \
 --waypoints-path ./surveyed-waypoints/deerPark.csv \
 --last-row-waypoint-index 3 \
 --turn-direction left \
 --row-spacing 6.0 \
 --headland-buffer 1.8 \
 --actuator-id 0 --actuator-rate-hz 5.0" 
  

if [ "$SKIP_ALIGNMENT" = true ]; then
    MAIN_CMD="$MAIN_CMD --no-hole-alignment"
fi

# Cleanup function to kill alignment service on exit
cleanup() {
    if [ "$SKIP_ALIGNMENT" = false ] && [ -n "$ALIGNMENT_PID" ]; then
        echo ""
        echo "🛑 Stopping hole alignment service (PID: $ALIGNMENT_PID)..."
        kill $ALIGNMENT_PID 2>/dev/null
        wait $ALIGNMENT_PID 2>/dev/null
        echo "✓ Alignment service stopped"
    fi
}

# Register cleanup on script exit
trap cleanup EXIT INT TERM

# Run navigation with output to both terminal and log file
echo "🚀 Starting navigation..."
eval "$MAIN_CMD" 2>&1 | tee "$LOG_FILE"

echo "Log saved to: $LOG_FILE"
