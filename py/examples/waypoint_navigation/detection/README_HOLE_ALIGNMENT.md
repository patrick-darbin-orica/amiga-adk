# Hole Alignment Integration

## Overview

The hole alignment system provides fine-grained positioning for the Amiga robot using the oak0 (downward-facing) camera. This ensures the dipbob tool is perfectly aligned over the collar hole before deployment.

## Navigation Workflow

The complete navigation sequence now includes hole alignment:

```
1. GPS Navigation → Search Zone
   ├─ Robot uses GPS waypoints to reach general area
   └─ Accuracy: ~1-2 meters

2. Front Camera (oak1) → Collar Detection & Approach
   ├─ YOLO detects collar in search zone
   ├─ Robot approaches collar using visual servoing
   └─ Accuracy: ~10-20cm

3. Robot Parks Over Collar
   └─ Initial positioning complete

4. ⭐ Downward Camera (oak0) → Fine Hole Alignment ⭐
   ├─ YOLO detects collar/hole from above
   ├─ Visual servoing for final positioning
   └─ Accuracy: ~1-2cm (±40 pixels at 1920x1080)

5. Deploy Dipbob
   └─ Tool perfectly aligned over hole

6. Move Forward (Tool → Origin)
   └─ Robot origin moves over hole

7. Open & Close Chute
   └─ Deployment complete
```

## Architecture

### Files Modified

1. **[navigation_manager.py](../utils/navigation_manager.py)**
   - Added hole alignment configuration parameters
   - Integrated `align_with_oak0()` into deployment sequence
   - Added command-line argument support

2. **[main.py](../main.py)**
   - Added hole alignment CLI arguments
   - Pass parameters to NavigationManager

3. **[oak0_alignment.py](../utils/oak0_alignment.py)** (existing)
   - Visual servoing alignment using oak0 camera
   - YOLO-based collar detection
   - PD controller for smooth alignment

### Integration Points

The hole alignment is triggered in `navigation_manager.py` at line 572-590, within the `execute_single_track()` post-actions:

```python
# 2) Perform hole alignment with oak0 (downward-facing camera)
if self.hole_alignment_enabled:
    logger.info("[HOLE ALIGN] Starting fine alignment using oak0 camera...")
    alignment_success = await align_with_oak0(
        canbus_client=self.canbus_client,
        model_path=self.hole_alignment_model_path,
        tolerance_px=self.hole_alignment_tolerance_px,
        move_gain=self.hole_alignment_move_gain,
        derivative_gain=self.hole_alignment_derivative_gain,
        max_velocity=self.hole_alignment_max_velocity,
        timeout_seconds=self.hole_alignment_timeout,
    )
```

## Configuration

### Command-Line Arguments

All hole alignment parameters can be configured via command-line:

```bash
python main.py \
  --config ./configs/config.json \
  --waypoints-path ./surveyed-waypoints/waypoints.csv \
  --tool-config-path ./configs/tool_config.json \
  # Hole Alignment Options:
  --hole-alignment-enabled \              # Enable alignment (default: True)
  --no-hole-alignment \                   # Disable alignment (override)
  --hole-alignment-model detection/best.engine \  # YOLO model path
  --hole-alignment-tolerance 40 \         # Tolerance in pixels (default: 40)
  --hole-alignment-gain 0.001 \           # Proportional gain (default: 0.001)
  --hole-alignment-max-velocity 0.15 \    # Max velocity m/s (default: 0.15)
  --hole-alignment-timeout 30.0           # Timeout seconds (default: 30.0)
```

### NavigationManager Parameters

Or configure programmatically when instantiating NavigationManager:

```python
nav_manager = NavigationManager(
    filter_client=filter_client,
    controller_client=controller_client,
    motion_planner=motion_planner,
    canbus_client=canbus_client,
    actuator=actuator,
    # Hole alignment configuration
    hole_alignment_enabled=True,
    hole_alignment_model_path=Path("detection/best.engine"),
    hole_alignment_tolerance_px=40,
    hole_alignment_move_gain=0.001,
    hole_alignment_derivative_gain=0.002,
    hole_alignment_max_velocity=0.15,
    hole_alignment_timeout=30.0,
)
```

## Default Parameters

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `hole_alignment_enabled` | `True` | Enable/disable hole alignment |
| `hole_alignment_model_path` | `detection/best.engine` | Path to YOLO model |
| `hole_alignment_tolerance_px` | `40` | Alignment tolerance (pixels) |
| `hole_alignment_move_gain` | `0.001` | Proportional gain (m/pixel) |
| `hole_alignment_derivative_gain` | `0.002` | Derivative gain for damping |
| `hole_alignment_max_velocity` | `0.15` | Maximum velocity (m/s) |
| `hole_alignment_timeout` | `30.0` | Timeout (seconds) |

## Camera Orientation

**CRITICAL**: The oak0 camera faces **BACKWARDS** (downward and rearward):

```
Robot Orientation:
    FRONT (oak1, forward-facing)
      ↑
      |
  [ROBOT]
      |
      ↓
    REAR (oak0, downward-facing)
```

This affects the control logic:
- **Collar higher in frame** → Robot **reverses** (moves backward)
- **Collar lower in frame** → Robot **drives forward**

The `oak0_alignment.py` module handles this correctly with the rear-facing convention.

## Visual Servoing Algorithm

The alignment uses a PD (Proportional-Derivative) controller:

1. **Detection**: YOLO model detects collar bounding box
2. **Error Calculation**: Vertical offset from target reticle position
3. **Control Law**:
   ```
   velocity = (offset_y × kp) - (dOffset/dt × kd)
   ```
   - `kp`: Proportional gain (default: 0.001 m/pixel)
   - `kd`: Derivative gain (default: 0.002 m/pixel)
4. **Command**: Send twist command to CAN bus
5. **Repeat**: Until aligned within tolerance or timeout

### Alignment Success Criteria

The robot is considered aligned when:
- Vertical offset ≤ tolerance (default: 40 pixels)
- Maintained for 3 consecutive frames
- Total time < timeout (default: 30 seconds)

## Failure Handling

The system is designed to be robust:

1. **Alignment Timeout**:
   - Logs warning
   - Proceeds with deployment anyway (graceful degradation)

2. **No Collar Detection**:
   - Stops robot
   - Waits for detection
   - Times out if collar not found

3. **CAN Bus Unavailable**:
   - Hole alignment automatically disabled
   - Warning logged at startup

## Disabling Hole Alignment

To disable hole alignment:

### Option 1: Command-line flag
```bash
python main.py --no-hole-alignment ...
```

### Option 2: Programmatic
```python
nav_manager = NavigationManager(
    ...,
    hole_alignment_enabled=False,
)
```

The system will skip the alignment step and proceed directly to dipbob deployment.

## Performance Tuning

### Increasing Alignment Speed
- **Increase `move_gain`**: Robot moves faster (may oscillate)
- **Decrease `tolerance_px`**: Less strict alignment (faster completion)
- **Increase `max_velocity`**: Higher speed limit

### Improving Alignment Accuracy
- **Decrease `move_gain`**: Slower, smoother movements
- **Increase `derivative_gain`**: More damping (less overshoot)
- **Decrease `tolerance_px`**: Stricter alignment requirement

### Example: Fast but Less Accurate
```bash
--hole-alignment-gain 0.002 \
--hole-alignment-tolerance 60 \
--hole-alignment-max-velocity 0.2
```

### Example: Slow but Very Accurate
```bash
--hole-alignment-gain 0.0005 \
--hole-alignment-derivative-gain 0.005 \
--hole-alignment-tolerance 20 \
--hole-alignment-max-velocity 0.1
```

## Logs and Debugging

The hole alignment system provides detailed logging:

```
[HOLE ALIGN] Starting fine alignment using oak0 camera...
[OAK0 ALIGN] Starting alignment (target: (958, 931), tolerance: ±40px)
[OAK0 ALIGN] Loaded YOLO model: best.engine
[OAK0 ALIGN] Connected to oak0: localhost:50010
[OAK0 ALIGN] Iteration 1: FORWARD (offset: +52.3px, velocity: +0.052m/s, conf: 0.87)
[OAK0 ALIGN] Iteration 2: FORWARD (offset: +38.1px, velocity: +0.038m/s, conf: 0.89)
[OAK0 ALIGN] Iteration 3: ALIGNED ✓ (offset: +12.5px, conf: 0.91, consecutive: 1/3)
[OAK0 ALIGN] Iteration 4: ALIGNED ✓ (offset: +8.2px, conf: 0.92, consecutive: 2/3)
[OAK0 ALIGN] Iteration 5: ALIGNED ✓ (offset: +5.1px, conf: 0.93, consecutive: 3/3)
[OAK0 ALIGN] ✓ Alignment confirmed after 5 iterations (1.2s)
[HOLE ALIGN] ✓ Hole alignment completed successfully
```

### Common Log Messages

- `[HOLE ALIGN] Hole alignment disabled, skipping...` → Feature disabled
- `[HOLE ALIGN] ⚠ Hole alignment failed or timed out, proceeding anyway...` → Timeout/failure, continuing
- `[OAK0 ALIGN] No collar detected` → YOLO not finding collar in frame
- `[OAK0 ALIGN] Timeout after 30.0s` → Alignment didn't converge in time

## Testing

### Standalone Test Script

The `hole_alignment.py` script can be run standalone for testing:

```bash
cd detection
python hole_alignment.py \
  --model-path best.engine \
  --target-x 958 \
  --target-y 931 \
  --tolerance-px 40 \
  --move-gain 0.001 \
  --auto-align  # Start aligned automatically
```

This allows testing the alignment without running full navigation.

## Related Documentation

- [Oak0 Camera Setup](README_OAK0_ALIGNMENT.md) - Camera calibration and setup
- [Navigation Manager](../utils/navigation_manager.py) - Main navigation orchestration
- [Oak0 Alignment Module](../utils/oak0_alignment.py) - Visual servoing implementation

## Troubleshooting

### Problem: Alignment always times out
- **Check**: Is the collar visible in oak0 camera feed?
- **Check**: Is the YOLO model detecting collars? (confidence threshold)
- **Fix**: Adjust `--hole-alignment-timeout` or `--conf` threshold

### Problem: Robot oscillates around target
- **Cause**: Proportional gain too high
- **Fix**: Decrease `--hole-alignment-gain` and increase `--hole-alignment-derivative-gain`

### Problem: Robot moves too slowly
- **Cause**: Gain too low or max velocity too conservative
- **Fix**: Increase `--hole-alignment-gain` and/or `--hole-alignment-max-velocity`

### Problem: Alignment disabled message at startup
- **Cause**: No CAN bus client available
- **Fix**: Ensure `--config` points to valid canbus configuration

## Future Enhancements

Potential improvements:
1. **Adaptive gains**: Adjust PD gains based on offset magnitude
2. **Multi-axis alignment**: Include lateral (X-axis) alignment
3. **Confidence-based weighting**: Trust high-confidence detections more
4. **Visual feedback**: Display alignment status in Flask GUI
5. **Kalman filtering**: Smooth noisy detections
