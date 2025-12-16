# Oak0 Visual Servoing Alignment

This directory contains the oak0 rear-facing camera alignment system for precise collar positioning using YOLO detection and visual servoing control.

## Overview

The oak0 alignment system allows the robot to fine-tune its position after GPS-based navigation by:
1. Detecting the collar using YOLO on the oak0 RGB camera feed
2. Calculating the offset between the collar center and a target reticle
3. Sending CAN bus twist commands to move the robot forward/backward
4. Confirming alignment when the collar is centered within tolerance

## Files

### Core Scripts

- **`test_oak0_alignment.py`** - Standalone test script for isolated testing
  - Interactive UI with real-time visualization
  - Manual control of auto-alignment
  - Adjustable parameters via command line
  - Perfect for tuning gain and tolerance values

- **`oak0_alignment.py`** (in `utils/`) - Production module for navigation integration
  - Clean API for use in navigation_manager.py
  - Automated alignment with configurable parameters
  - Returns success/failure status

### Utilities

- **`calculate_tolerance.py`** - Pixel tolerance calculator
  - Converts physical distance (cm) to pixels
  - Accounts for camera height and FOV
  - Recommended: ~28px for 2cm at 1m height

- **`test_oak0_detection.py`** - Basic detection test (existing)
  - Verifies YOLO model works on oak0 stream
  - No alignment logic

### Models

- **`best.pt`** - PyTorch YOLO model (slow, ~1 FPS on CPU)
- **`best.engine`** - TensorRT GPU-accelerated model (fast, ~10-20 FPS) ✅ **READY**
- **`best.onnx`** - Intermediate ONNX export

## Quick Start

### 1. Test Alignment in Isolation

First, test the alignment system standalone to tune parameters:

```bash
cd detection
python test_oak0_alignment.py \
  --tolerance-px 40 \
  --move-gain 0.001 \
  --max-velocity 0.15
```

**Controls:**
- `'a'` - Enable/disable auto-alignment (robot will move)
- `'s'` - Emergency stop
- `'q'` - Quit

**What you'll see:**
- Real-time oak0 camera feed with detection overlay
- Target reticle (crosshair) at your configured position
- Collar bounding box (if detected)
- Offset in pixels
- Current velocity command
- Alignment status

**Tuning tips:**
- Start with `--move-gain 0.001` (conservative)
- Increase gain if robot moves too slowly
- Decrease gain if robot overshoots
- Tolerance of 40px ≈ 2cm physical distance (at 1m camera height)

### 2. Calculate Optimal Tolerance

If you want to recalculate tolerance for a different physical distance or camera height:

```bash
python calculate_tolerance.py \
  --physical-distance 0.02 \
  --camera-height 1.0
```

Output will show recommended `--tolerance-px` value.

### 3. Integration into Navigation (Future)

When ready to integrate, add to [navigation_manager.py:531-540](../../utils/navigation_manager.py#L531-L540):

```python
# After robot parks over collar (GPS-based), before dipbob deployment:
from utils.oak0_alignment import align_with_oak0
from pathlib import Path

# Perform oak0 visual servoing alignment
alignment_success = await align_with_oak0(
    canbus_client=self.canbus_client,
    model_path=Path(__file__).parent / "detection" / "best.engine",
    target_reticle_y=931,
    tolerance_px=40,  # Tune based on isolated testing
    move_gain=0.001,   # Tune based on isolated testing
    max_velocity=0.15,
    timeout_seconds=30.0
)

if not alignment_success:
    logger.warning("[OAK0] Alignment failed, proceeding with GPS position")
    
# Proceed with dipbob deployment...
await trigger_dipbob(...)
```

## Camera Orientation

**CRITICAL**: oak0 faces **BACKWARDS** (rear-facing camera)

- Collar **higher** in frame → Robot **reverses** (negative velocity)
- Collar **lower** in frame → Robot **forward** (positive velocity)

This is handled automatically in the code.

## Parameters

### Target Reticle Position
- **target_reticle_x**: 958 (horizontal center where collar should align)
- **target_reticle_y**: 931 (vertical position where collar should align)

These values are camera-specific and should be measured based on your oakdipper tool offset and camera mounting.

### Tolerance
- **tolerance_px**: ±40 pixels (approximately 2cm physical)
- Calculated using `calculate_tolerance.py`
- Conservative (larger) values recommended for safety

### Move Gain
- **move_gain**: 0.001 m/px (proportional control gain)
- How much the robot moves per pixel of error
- Start conservative, increase if too slow
- Typical range: 0.0005 - 0.002

### Safety Limits
- **max_velocity**: 0.15 m/s (maximum forward/backward speed)
- **timeout_seconds**: 30.0 (maximum time to spend aligning)
- **max_iterations**: 20 (maximum alignment attempts)

## Troubleshooting

### No collar detected
- Check that collar is in camera FOV
- Verify YOLO model confidence threshold (default: 0.3)
- Test with `test_oak0_detection.py` first

### Robot doesn't move
- Verify CAN bus connection
- Check auto-mode is enabled on robot
- Confirm `'a'` key was pressed to enable auto-alignment

### Robot overshoots
- Decrease `--move-gain` value
- Reduce `--max-velocity`
- Increase `--tolerance-px` for more forgiving alignment

### Too slow
- Increase `--move-gain` value
- Use TensorRT engine (best.engine) instead of PyTorch model
- Check camera frame rate

## Performance

### Model Performance
- **best.pt** (PyTorch): ~1 FPS, 72ms inference
- **best.engine** (TensorRT): ~10-20 FPS, 5-10ms inference ✅

### Alignment Performance
- Typical alignment time: 5-15 seconds
- Depends on initial offset and gain tuning

## Next Steps

1. **Test in isolation** with `test_oak0_alignment.py`
2. **Tune parameters** (gain, tolerance) for your robot
3. **Verify reticle position** matches your dipper/camera setup
4. **Integrate into navigation** when confident
5. **Field test** with real collars

## Notes

- Only class 0 (Collar) detections are used
- Model has 36 class channels but only 1 is actively trained
- Spurious detections from other classes are filtered out
- System uses proportional control (P-only, no I or D terms)
- Consider adding derivative term if oscillations occur
