"""
Oak visual servoing alignment service for collar detection using DepthAI.

This module provides a continuous background alignment service that:
- Runs continuously with DepthAI pipeline (no startup delay)
- Processes frames and detects collars in real-time
- Only sends movement commands when alignment is enabled via flag
- Shares visualization frames with Flask GUI

The service runs independently and is controlled via flag files:
- enable_alignment() - Start aligning (called when robot reaches waypoint)
- disable_alignment() - Stop aligning (called after alignment or between waypoints)

Camera orientation: oak2 faces BACKWARDS
- Collar higher in frame → Robot reverses (negative velocity)
- Collar lower in frame → Robot drives forward (positive velocity)

Usage as background service:
    # Start the service (runs continuously)
    python -m utils.hole_alignment --canbus-config configs/canbus_config.json

    # In navigation_manager.py:
    from utils.oak2_camera_cache import enable_alignment, disable_alignment, is_alignment_enabled

    # When robot reaches waypoint
    enable_alignment()
    await wait_for_alignment()  # Wait until aligned
    disable_alignment()
"""

import asyncio
import argparse
import logging
import numpy as np
from pathlib import Path
import depthai as dai
import time

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.canbus.canbus_pb2 import Twist2d

# Import alignment control and frame sharing
from utils.oak2_camera_cache import (
    set_oak2_frame,
    set_inference_active,
    is_alignment_enabled,
)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ContinuousAlignmentService:
    """Continuous hole alignment service that runs in the background."""

    def __init__(
        self,
        canbus_client: EventClient,
        model_path: Path,
        device_id: str = "14442C10D14CFFD600",  # oak2 device ID
        target_reticle_x: int = 958,
        target_reticle_y: int = 931,
        tolerance_px: int = 40,
        dead_zone_px: int = 10,
        min_consecutive_aligned: int = 10,
        move_gain: float = 0.001,
        derivative_gain: float = 0.002,
        max_velocity: float = 0.15,
        conf_threshold: float = 0.3,
        min_detections_required: int = 3,
        img_size: int = 640,
        fps: int = 15,
        min_scan_height: int = 200,  # Only accept detections from this pixel height and below
    ):
        """
        Initialize continuous alignment service.

        Args:
            canbus_client: EventClient for CAN bus service
            model_path: Path to YOLO model (.pt or .engine)
            device_id: OAK device MxID (oak2 rear-facing camera)
            target_reticle_x: Target X pixel position for collar center
            target_reticle_y: Target Y pixel position for collar center
            tolerance_px: Alignment tolerance in pixels (outer boundary)
            dead_zone_px: Dead zone tolerance in pixels (inner boundary, no corrections applied)
            min_consecutive_aligned: Minimum consecutive frames in dead zone before stopping corrections
            move_gain: Proportional gain (meters per pixel offset)
            derivative_gain: Derivative gain for damping oscillations
            max_velocity: Maximum linear velocity (m/s)
            conf_threshold: YOLO confidence threshold
            min_detections_required: Minimum consecutive aligned detections
            img_size: Camera image size (640x640)
            fps: Camera frame rate
            min_scan_height: Minimum Y pixel height for detection (only scan from this height and below)
        """
        self.canbus_client = canbus_client
        self.model_path = model_path

        # Camera parameters
        self.device_id = device_id
        self.img_size = img_size
        self.fps = fps

        # Alignment parameters
        self.target_reticle_x = target_reticle_x
        self.target_reticle_y = target_reticle_y
        self.tolerance_px = tolerance_px
        self.dead_zone_px = dead_zone_px
        self.min_consecutive_aligned = min_consecutive_aligned
        self.move_gain = move_gain
        self.derivative_gain = derivative_gain
        self.max_velocity = max_velocity

        # Detection parameters
        self.conf_threshold = conf_threshold
        self.min_detections_required = min_detections_required
        self.min_scan_height = min_scan_height

        # State
        self.desired_velocity = 0.0
        self.last_offset_y = None
        self.consecutive_aligned = 0

        # Stats
        self.frame_count = 0
        self.detection_count = 0

        # Pipeline and model (initialized in setup)
        self.model = None
        self.pipeline = None
        self.qRgb = None

    async def setup(self):
        """Initialize model and DepthAI pipeline."""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics not available")
            return False

        # Load YOLO model
        if not self.model_path.exists():
            logger.error(f"[ALIGN SERVICE] Model not found: {self.model_path}")
            return False

        try:
            logger.info(f"[ALIGN SERVICE] Loading YOLO model: {self.model_path.name}")
            self.model = YOLO(str(self.model_path))
            logger.info(f"[ALIGN SERVICE] ✓ Model loaded")
        except Exception as e:
            logger.error(f"[ALIGN SERVICE] Failed to load model: {e}")
            return False

        # Warm up model
        logger.info("[ALIGN SERVICE] Warming up model...")
        dummy_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        _ = self.model.predict(dummy_frame, conf=self.conf_threshold, verbose=False, imgsz=self.img_size)
        logger.info("[ALIGN SERVICE] ✓ Model warmed up")

        # Create DepthAI pipeline
        logger.info(f"[ALIGN SERVICE] Creating DepthAI pipeline for device {self.device_id}")
        try:
            device = dai.Device(self.device_id)
            self.pipeline = dai.Pipeline(device)

            # RGB camera
            camRgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

            # Request RGB output
            xoutRgb = camRgb.requestOutput((self.img_size, self.img_size))

            # Create output queue
            self.qRgb = xoutRgb.createOutputQueue(maxSize=1, blocking=False)

            logger.info("[ALIGN SERVICE] ✓ DepthAI pipeline created")
        except Exception as e:
            logger.error(f"[ALIGN SERVICE] Failed to create pipeline: {e}")
            return False

        return True

    async def run(self):
        """Main service loop - runs continuously."""
        logger.info("[ALIGN SERVICE] Starting continuous alignment service...")

        # Signal that inference is active
        set_inference_active(True)

        # Start background twist sender
        twist_sender_task = asyncio.create_task(self._twist_command_sender())

        # Track frame timing
        last_frame_time = time.time()

        try:
            # Start pipeline
            self.pipeline.start()
            with self.pipeline:
                logger.info("[ALIGN SERVICE] ✓ Service running - waiting for alignment requests...")

                while self.pipeline.isRunning():
                    # Drain latest frame from queue
                    latestRgb = None
                    while self.qRgb.has():
                        latestRgb = self.qRgb.get()

                    if latestRgb is None:
                        await asyncio.sleep(0.001)
                        continue

                    # Get frame
                    frame = latestRgb.getCvFrame()
                    if frame is None:
                        await asyncio.sleep(0.001)
                        continue

                    self.frame_count += 1
                    current_time = time.time()

                    # Calculate FPS
                    processing_fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
                    last_frame_time = current_time

                    # Process frame
                    await self._process_frame(frame, processing_fps)

                    # Yield to event loop
                    await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("[ALIGN SERVICE] Interrupted by user")
        finally:
            # Cancel twist sender
            twist_sender_task.cancel()
            try:
                await twist_sender_task
            except asyncio.CancelledError:
                pass

            # Stop robot
            await self._send_stop_command()

            # Clear inference flag
            set_inference_active(False)
            logger.info("[ALIGN SERVICE] Service stopped")

    async def _process_frame(self, frame: np.ndarray, fps: float = 0.0):
        """Process a single frame: detect, visualize, and optionally align."""
        import cv2

        # Check if alignment is enabled
        alignment_active = is_alignment_enabled()

        # Run YOLO detection
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=self.img_size
        )

        # Extract collar detections (class 0 only) AND below min_scan_height
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls != 0:  # Only keep Collar
                    continue

                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()

                # Filter: only keep detections where the center Y is at or below min_scan_height
                center_y = (xyxy[1] + xyxy[3]) / 2
                if center_y < self.min_scan_height:
                    continue  # Skip detections above the minimum scan height

                detections.append({'confidence': conf, 'bbox': xyxy})

        # Update stats
        if detections:
            self.detection_count += 1

        # Calculate offset and alignment
        offset_y = None
        is_aligned = False
        in_dead_zone = False
        velocity_cmd = 0.0

        if detections and alignment_active:
            # Use highest confidence detection
            best_det = max(detections, key=lambda d: d['confidence'])
            _, y1, _, y2 = best_det['bbox']

            # Calculate collar center
            collar_center_y = (y1 + y2) / 2

            # Calculate vertical offset
            offset_y = collar_center_y - self.target_reticle_y

            # Check if in dead zone (tight tolerance for stable holding)
            in_dead_zone = abs(offset_y) <= self.dead_zone_px

            # Check if in broader alignment tolerance
            is_aligned = abs(offset_y) <= self.tolerance_px

            # Calculate derivative term
            if self.last_offset_y is not None:
                offset_derivative = offset_y - self.last_offset_y
            else:
                offset_derivative = 0.0

            self.last_offset_y = offset_y

            # Dead zone logic: if within dead zone, hold at 0 and count consecutive frames
            if in_dead_zone:
                self.consecutive_aligned += 1

                # Only stop corrections after sufficient consecutive frames in dead zone
                if self.consecutive_aligned >= self.min_consecutive_aligned:
                    self.desired_velocity = 0.0
                    if self.consecutive_aligned % 10 == self.min_consecutive_aligned:  # Log occasionally
                        logger.info(
                            f"[ALIGN SERVICE] LOCKED ✓ "
                            f"(offset: {offset_y:+.1f}px, locked for {self.consecutive_aligned} frames)"
                        )
                else:
                    # Still building up consecutive frames, apply gentle correction
                    proportional_term = offset_y * self.move_gain
                    derivative_term = offset_derivative * self.derivative_gain
                    velocity_cmd = np.clip(
                        proportional_term - derivative_term,
                        -self.max_velocity,
                        self.max_velocity
                    )
                    self.desired_velocity = velocity_cmd
                    logger.info(
                        f"[ALIGN SERVICE] STABILIZING "
                        f"(offset: {offset_y:+.1f}px, frames: {self.consecutive_aligned}/"
                        f"{self.min_consecutive_aligned})"
                    )
            else:
                # Outside dead zone - reset counter and apply corrections
                self.consecutive_aligned = 0

                # Calculate velocity command with PD control
                proportional_term = offset_y * self.move_gain
                derivative_term = offset_derivative * self.derivative_gain

                velocity_cmd = np.clip(
                    proportional_term - derivative_term,
                    -self.max_velocity,
                    self.max_velocity
                )

                # Apply minimum velocity threshold to overcome static friction
                MIN_VELOCITY = 0.045
                if abs(velocity_cmd) > 0.001:
                    if abs(velocity_cmd) < MIN_VELOCITY:
                        velocity_cmd = MIN_VELOCITY if velocity_cmd > 0 else -MIN_VELOCITY

                self.desired_velocity = velocity_cmd
                direction = "FORWARD" if velocity_cmd > 0 else "REVERSE"
                logger.info(
                    f"[ALIGN SERVICE] {direction} "
                    f"(offset: {offset_y:+.1f}px, vel: {velocity_cmd:+.4f}m/s)"
                )
        else:
            # Alignment not active or no detection
            self.desired_velocity = 0.0
            self.last_offset_y = None
            self.consecutive_aligned = 0

        # Create visualization and share with Flask
        vis_frame = self._create_visualization(frame, detections, offset_y, is_aligned, velocity_cmd, fps, alignment_active)
        set_oak2_frame(vis_frame)

    def _create_visualization(
        self, image: np.ndarray, detections: list, offset_y: float,
        is_aligned: bool, velocity_cmd: float, fps: float, alignment_active: bool
    ):
        """Create visualization frame with detection overlay."""
        import cv2

        vis_image = image.copy()

        # Draw FPS and status
        status_color = (0, 255, 0) if alignment_active else (100, 100, 100)
        align_status = "ALIGNMENT ACTIVE" if alignment_active else "ALIGNMENT IDLE"
        cv2.putText(vis_image, f"FPS: {fps:.1f} | {align_status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Draw scan region boundary
        cv2.line(
            vis_image,
            (0, self.min_scan_height),
            (vis_image.shape[1], self.min_scan_height),
            (255, 0, 0),  # Blue line
            2
        )
        cv2.putText(
            vis_image,
            "SCAN REGION BELOW",
            (10, self.min_scan_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        # Draw target reticle
        reticle_color = (0, 255, 0) if is_aligned else (0, 165, 255)
        reticle_size = 30
        # cv2.line(vis_image, (self.target_reticle_x - reticle_size, self.target_reticle_y),
        #         (self.target_reticle_x + reticle_size, self.target_reticle_y), reticle_color, 2)
        # cv2.line(vis_image, (self.target_reticle_x, self.target_reticle_y - reticle_size),
        #         (self.target_reticle_x, self.target_reticle_y + reticle_size), reticle_color, 2)
        cv2.circle(vis_image, (self.target_reticle_x, self.target_reticle_y), 5, reticle_color, -1)

        # Draw tolerance zone (outer boundary) - red horizontal lines only, 40px wide
        tolerance_color = (0, 0, 255)  # Red
        tolerance_line_width = 40
        tolerance_x_start = self.target_reticle_x - tolerance_line_width // 2
        tolerance_x_end = self.target_reticle_x + tolerance_line_width // 2

        # Top tolerance line
        cv2.line(
            vis_image,
            (tolerance_x_start, self.target_reticle_y - self.tolerance_px),
            (tolerance_x_end, self.target_reticle_y - self.tolerance_px),
            tolerance_color,
            2
        )
        # Bottom tolerance line
        cv2.line(
            vis_image,
            (tolerance_x_start, self.target_reticle_y + self.tolerance_px),
            (tolerance_x_end, self.target_reticle_y + self.tolerance_px),
            tolerance_color,
            2
        )

        # Draw dead zone (inner boundary) - horizontal lines only, 35px wide
        in_dead_zone = offset_y is not None and abs(offset_y) <= self.dead_zone_px
        is_locked = in_dead_zone and self.consecutive_aligned >= self.min_consecutive_aligned
        dead_zone_color = (0, 255, 0) if is_locked else (0, 255, 255)  # Green if locked, yellow if stabilizing
        dead_zone_line_width = 30
        dead_zone_x_start = self.target_reticle_x - dead_zone_line_width // 2
        dead_zone_x_end = self.target_reticle_x + dead_zone_line_width // 2

        # Top dead zone line
        cv2.line(
            vis_image,
            (dead_zone_x_start, self.target_reticle_y - self.dead_zone_px),
            (dead_zone_x_end, self.target_reticle_y - self.dead_zone_px),
            dead_zone_color,
            2
        )
        # Bottom dead zone line
        cv2.line(
            vis_image,
            (dead_zone_x_start, self.target_reticle_y + self.dead_zone_px),
            (dead_zone_x_end, self.target_reticle_y + self.dead_zone_px),
            dead_zone_color,
            2
        )

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            conf = det['confidence']

            bbox_color = (0, 255, 0) if is_aligned else (0, 255, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, 2)
            cv2.putText(vis_image, f"Collar {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

            # Draw collar center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(vis_image, (center_x, center_y), 8, (255, 0, 255), -1)
            cv2.line(vis_image, (center_x, center_y), (self.target_reticle_x, self.target_reticle_y),
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Status overlay
        status_y = 60
        line_height = 30

        cv2.putText(vis_image, f"Detections: {len(detections)}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += line_height

        if offset_y is not None:
            offset_color = (0, 255, 0) if is_aligned else (0, 165, 255)
            cv2.putText(vis_image, f"Offset Y: {offset_y:+.1f}px", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)
            status_y += line_height

        # Status text with dead zone info
        if offset_y is not None and alignment_active:
            in_dead_zone = abs(offset_y) <= self.dead_zone_px
            is_locked = in_dead_zone and self.consecutive_aligned >= self.min_consecutive_aligned

            if is_locked:
                cv2.putText(
                    vis_image,
                    f"STATUS: LOCKED ✓ ({self.consecutive_aligned} frames)",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            elif in_dead_zone:
                cv2.putText(
                    vis_image,
                    f"STATUS: STABILIZING ({self.consecutive_aligned}/{self.min_consecutive_aligned})",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
            else:
                direction = "FORWARD" if velocity_cmd > 0 else "REVERSE"
                cv2.putText(
                    vis_image,
                    f"STATUS: {direction}",
                    (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
        status_y += line_height

        cv2.putText(vis_image, f"Velocity: {self.desired_velocity:+.3f} m/s", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis_image

    async def _twist_command_sender(self):
        """Background task that sends twist commands at 20 Hz when alignment is enabled."""
        logger.info("[ALIGN SERVICE] Twist sender started (20 Hz)")
        while True:
            try:
                # Only send commands if alignment is enabled
                if is_alignment_enabled():
                    twist = Twist2d()
                    twist.linear_velocity_x = self.desired_velocity
                    twist.linear_velocity_y = 0.0
                    twist.angular_velocity = 0.0

                    await self.canbus_client.request_reply("/twist", twist)

                await asyncio.sleep(0.05)  # 20 Hz
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[ALIGN SERVICE] Failed to send twist: {e}")
                await asyncio.sleep(0.05)

    async def _send_stop_command(self):
        """Send zero velocity command."""
        twist = Twist2d()
        twist.linear_velocity_x = 0.0
        twist.linear_velocity_y = 0.0
        twist.angular_velocity = 0.0
        try:
            await self.canbus_client.request_reply("/twist", twist)
        except Exception as e:
            logger.warning(f"[ALIGN SERVICE] Failed to send stop: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Continuous hole alignment service using oak2 camera and DepthAI"
    )

    # Model config
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent.parent / "detection" / "best1.engine",
        help="Path to YOLO model (.pt or .engine)"
    )

    # CAN bus config
    parser.add_argument(
        "--canbus-config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "canbus_config.json",
        help="Path to canbus service config"
    )

    # Camera parameters
    parser.add_argument(
        "--device-id",
        type=str,
        default="14442C10D14CFFD600",
        help="OAK device MxID (oak2 rear-facing)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Camera image size"
    )

    # Alignment parameters
    parser.add_argument(
        "--target-x",
        type=int,
        default=330,
        help="Target reticle X position"
    )
    parser.add_argument(
        "--target-y",
        type=int,
        default=260,
        help="Target reticle Y position"
    )
    parser.add_argument(
        "--tolerance-px",
        type=int,
        default=15,
        help="Alignment tolerance (pixels, outer boundary)"
    )
    parser.add_argument(
        "--dead-zone-px",
        type=int,
        default=10,
        help="Dead zone tolerance (pixels, inner boundary for stable holding)"
    )
    parser.add_argument(
        "--min-consecutive-aligned",
        type=int,
        default=10,
        help="Minimum consecutive frames in dead zone before stopping corrections"
    )
    parser.add_argument(
        "--move-gain",
        type=float,
        default=0.001,
        help="Proportional gain"
    )
    parser.add_argument(
        "--derivative-gain",
        type=float,
        default=0.002,
        help="Derivative gain"
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.15,
        help="Maximum linear velocity (m/s)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--min-scan-height",
        type=int,
        default=100,
        help="Minimum Y pixel height for detection (only scan from this height and below)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load canbus config
    if not args.canbus_config.exists():
        logger.error(f"Canbus config not found: {args.canbus_config}")
        return

    canbus_config = proto_from_json_file(args.canbus_config, EventServiceConfig())
    canbus_client = EventClient(canbus_config)
    logger.info(f"Connected to canbus: {canbus_config.host}:{canbus_config.port}")

    # Create and run service
    service = ContinuousAlignmentService(
        canbus_client=canbus_client,
        model_path=args.model_path,
        device_id=args.device_id,
        target_reticle_x=args.target_x,
        target_reticle_y=args.target_y,
        tolerance_px=args.tolerance_px,
        dead_zone_px=args.dead_zone_px,
        min_consecutive_aligned=args.min_consecutive_aligned,
        move_gain=args.move_gain,
        derivative_gain=args.derivative_gain,
        max_velocity=args.max_velocity,
        conf_threshold=args.conf,
        img_size=args.img_size,
        min_scan_height=args.min_scan_height,
    )

    if await service.setup():
        await service.run()


if __name__ == "__main__":
    asyncio.run(main())
