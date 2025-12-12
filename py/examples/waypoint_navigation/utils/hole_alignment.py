"""
Oak visual servoing alignment module for collar detection.

This module provides collar alignment functionality using the oak0 rear-facing
RGB camera with YOLO detection. The robot uses visual servoing to align the
dipper tool with the collar by moving forward/backward based on the collar's
vertical position in the camera frame.

Camera orientation: oak0 faces BACKWARDS
- Collar higher in frame → Robot reverses (negative velocity)
- Collar lower in frame → Robot drives forward (positive velocity)

Typical usage in navigation_manager.py:
    from utils.oak0_alignment import align_with_oak0

    # After robot parks over collar (GPS-based)
    success = await align_with_oak0(
        canbus_client=self.canbus_client,
        model_path=Path("detection/best.engine"),
        target_reticle_y=931,
        tolerance_px=40,
        move_gain=0.001
    )

    if success:
        # Proceed with dipbob deployment
        await trigger_dipbob(...)
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri
from farm_ng.canbus.canbus_pb2 import Twist2d

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


async def align_with_oak0(
    canbus_client: EventClient,
    model_path: Path,
    oak0_config_path: Optional[Path] = None,
    target_reticle_x: int = 958,
    target_reticle_y: int = 931,
    tolerance_px: int = 40,
    move_gain: float = 0.001,
    derivative_gain: float = 0.002,
    max_velocity: float = 0.15,
    conf_threshold: float = 0.3,
    max_iterations: int = 20,
    timeout_seconds: float = 30.0,
    min_detections_required: int = 3,
) -> bool:
    """
    Align robot with collar using oak0 visual servoing.

    This function:
    1. Subscribes to oak0 RGB camera feed
    2. Runs YOLO detection to find collar
    3. Calculates vertical offset from target reticle
    4. Sends CAN bus twist commands to align robot
    5. Returns when aligned within tolerance or timeout

    Args:
        canbus_client: EventClient for CAN bus service
        model_path: Path to YOLO model (.pt or .engine)
        oak0_config_path: Path to oak0 service config (default: ../../camera_client/service_config.json)
        target_reticle_x: Target X pixel position for collar center
        target_reticle_y: Target Y pixel position for collar center
        tolerance_px: Alignment tolerance in pixels
        move_gain: Proportional gain (meters per pixel offset)
        derivative_gain: Derivative gain for damping oscillations
        max_velocity: Maximum linear velocity (m/s)
        conf_threshold: YOLO confidence threshold
        max_iterations: Maximum alignment iterations
        timeout_seconds: Maximum time to spend aligning
        min_detections_required: Minimum consecutive aligned detections before success

    Returns:
        True if aligned successfully, False if timeout or failed
    """

    if not ULTRALYTICS_AVAILABLE:
        logger.error("Ultralytics not available, cannot perform oak0 alignment")
        return False

    logger.info(
        f"[OAK0 ALIGN] Starting alignment "
        f"(target: ({target_reticle_x}, {target_reticle_y}), tolerance: ±{tolerance_px}px)"
    )

    # Load YOLO model
    if not model_path.exists():
        logger.error(f"[OAK0 ALIGN] Model not found: {model_path}")
        return False

    try:
        model = YOLO(str(model_path))
        logger.info(f"[OAK0 ALIGN] Loaded YOLO model: {model_path.name}")
    except Exception as e:
        logger.error(f"[OAK0 ALIGN] Failed to load model: {e}")
        return False

    # Load oak0 config
    if oak0_config_path is None:
        oak0_config_path = Path(__file__).resolve().parents[2] / "camera_client" / "service_config.json"

    if not oak0_config_path.exists():
        logger.error(f"[OAK0 ALIGN] oak0 config not found: {oak0_config_path}")
        return False

    try:
        oak0_config = proto_from_json_file(oak0_config_path, EventServiceConfig())
        oak0_client = EventClient(oak0_config)
        logger.info(f"[OAK0 ALIGN] Connected to oak0: {oak0_config.host}:{oak0_config.port}")
    except Exception as e:
        logger.error(f"[OAK0 ALIGN] Failed to connect to oak0: {e}")
        return False

    # Create subscription
    subscription = SubscribeRequest(
        uri=Uri(path="/rgb", query="service_name=oak/0"),
        every_n=1
    )

    # Alignment state
    iteration = 0
    consecutive_aligned = 0
    start_time = asyncio.get_event_loop().time()
    last_offset_y = None

    try:
        async for event, message in oak0_client.subscribe(subscription, decode=True):
            iteration += 1
            elapsed = asyncio.get_event_loop().time() - start_time

            # Check timeout
            if elapsed > timeout_seconds:
                logger.warning(f"[OAK0 ALIGN] Timeout after {timeout_seconds}s")
                await _send_stop_command(canbus_client)
                return False

            # Check max iterations
            if iteration > max_iterations:
                logger.warning(f"[OAK0 ALIGN] Max iterations ({max_iterations}) exceeded")
                await _send_stop_command(canbus_client)
                return False

            # Decode image
            image = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.warning(f"[OAK0 ALIGN] Failed to decode frame {iteration}")
                continue

            # Run YOLO detection
            # Note: classes=[0] doesn't work with TensorRT engines, filter in post-processing
            # IMPORTANT: Specify imgsz=640 to match TensorRT engine input size
            try:
                results = model.predict(image, conf=conf_threshold, verbose=False, imgsz=640)
            except Exception as e:
                logger.error(f"[OAK0 ALIGN] Detection failed: {e}")
                continue

            # Extract collar detections (class 0 only)
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    if cls != 0:  # Only keep Collar (class 0)
                        continue

                    conf = float(boxes.conf[i])
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    detections.append({'confidence': conf, 'bbox': xyxy})

            # Check if collar detected
            if not detections:
                logger.warning(f"[OAK0 ALIGN] Iteration {iteration}: No collar detected")
                consecutive_aligned = 0
                await _send_stop_command(canbus_client)
                await asyncio.sleep(0.1)
                continue

            # Use highest confidence detection
            best_det = max(detections, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best_det['bbox']
            conf = best_det['confidence']

            # Calculate collar center
            collar_center_y = (y1 + y2) / 2

            # Calculate vertical offset from target reticle
            offset_y = collar_center_y - target_reticle_y

            # Check alignment
            is_aligned = abs(offset_y) <= tolerance_px

            # Calculate derivative term (rate of change of offset) for damping
            if last_offset_y is not None:
                offset_derivative = offset_y - last_offset_y
            else:
                offset_derivative = 0.0

            # Update last offset for next iteration
            last_offset_y = offset_y

            if is_aligned:
                consecutive_aligned += 1
                logger.info(
                    f"[OAK0 ALIGN] Iteration {iteration}: ALIGNED ✓ "
                    f"(offset: {offset_y:+.1f}px, conf: {conf:.2f}, "
                    f"consecutive: {consecutive_aligned}/{min_detections_required})"
                )

                # Stop robot
                await _send_stop_command(canbus_client)

                # Check if alignment confirmed
                if consecutive_aligned >= min_detections_required:
                    logger.info(
                        f"[OAK0 ALIGN] ✓ Alignment confirmed "
                        f"after {iteration} iterations ({elapsed:.1f}s)"
                    )
                    return True

                # Wait briefly before next check
                await asyncio.sleep(0.2)

            else:
                # Not aligned - send correction velocity
                consecutive_aligned = 0

                # Calculate velocity with derivative damping (oak0 faces BACKWARDS!)
                # Positive offset_y (collar lower in frame) → forward velocity
                # Negative offset_y (collar higher in frame) → reverse velocity
                # Derivative term opposes rapid changes, preventing overshoot
                proportional_term = offset_y * move_gain
                derivative_term = offset_derivative * derivative_gain

                velocity_cmd = np.clip(
                    proportional_term - derivative_term,
                    -max_velocity,
                    max_velocity
                )

                direction = "FORWARD" if velocity_cmd > 0 else "REVERSE"
                logger.info(
                    f"[OAK0 ALIGN] Iteration {iteration}: {direction} "
                    f"(offset: {offset_y:+.1f}px, velocity: {velocity_cmd:+.3f}m/s, conf: {conf:.2f})"
                )

                # Send velocity command
                await _send_twist_command(canbus_client, velocity_cmd)

                # Wait for movement
                await asyncio.sleep(0.2)

    except asyncio.CancelledError:
        logger.warning("[OAK0 ALIGN] Alignment cancelled")
        await _send_stop_command(canbus_client)
        return False
    except Exception as e:
        logger.error(f"[OAK0 ALIGN] Alignment error: {e}")
        await _send_stop_command(canbus_client)
        return False
    finally:
        # Ensure robot is stopped
        await _send_stop_command(canbus_client)

    logger.warning("[OAK0 ALIGN] Alignment loop exited without success")
    return False


async def _send_twist_command(canbus_client: EventClient, linear_velocity_x: float):
    """Send twist command to CAN bus."""
    twist = Twist2d()
    twist.linear_velocity_x = linear_velocity_x
    twist.linear_velocity_y = 0.0
    twist.angular_velocity = 0.0

    try:
        await canbus_client.request_reply("/twist", twist)
    except Exception as e:
        logger.warning(f"[OAK0 ALIGN] Failed to send twist command: {e}")


async def _send_stop_command(canbus_client: EventClient):
    """Send zero velocity command."""
    await _send_twist_command(canbus_client, 0.0)
