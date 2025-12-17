#!/usr/bin/env python3
"""
TESTING ONLY
Standalone test script for oak2 collar alignment using visual servoing.

This script uses a depthai pipeline (like detectionPlot.py) instead of farm-ng
framework subscription to access the camera.

This script:
1. Creates a depthai pipeline for oak2 RGB camera feed (front-facing)
2. Runs YOLO detection to find collar bounding box
3. Compares collar center to target reticle position
4. Sends CAN bus twist commands to align robot forward/backward
5. Displays real-time visualization with reticle and bbox

Camera orientation: oak2 faces FORWARD
- Collar higher in frame → Robot drives forward (positive velocity)
- Collar lower in frame → Robot reverses (negative velocity)

Usage:
    python test_oak2_alignment.py --move-gain 0.001 --tolerance-px 40

Controls:
    'a' - Enable auto-alignment (sends CAN bus commands)
    's' - Stop robot (zero velocity)
    'q' - Quit
"""

import asyncio
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
import sys
import depthai as dai

# Add parent directory to import detection classes
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import frame sharing for Flask GUI
from utils.oak0_camera_cache import set_oak0_frame, set_inference_active

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.canbus.canbus_pb2 import Twist2d

# Import YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False
    sys.exit(1)


class oak2AlignmentTester:
    """Visual servoing alignment tester for oak2 collar detection using depthai pipeline."""

    def __init__(
        self,
        model_path: Path,
        canbus_config_path: Path,
        target_reticle_x: int = 320,
        target_reticle_y: int = 480,
        tolerance_px: int = 40,
        move_gain: float = 0.001,
        derivative_gain: float = 0.002,
        conf_threshold: float = 0.3,
        max_velocity: float = 0.2,
        visualization_scale: float = 1.0,
        headless: bool = False,
        auto_align_at_startup: bool = False,
        img_size: int = 640,
        fps: int = 15,
        device_id: str = "14442C10D14CFFD600",  # oak2 device ID
        min_scan_height: int = 200,  # Only accept detections from this pixel height and below
    ):
        """
        Initialize the oak2 alignment tester.

        Args:
            model_path: Path to YOLO model (.pt or .engine)
            canbus_config_path: Path to canbus service config
            target_reticle_x: Target X pixel position for collar center
            target_reticle_y: Target Y pixel position for collar center
            tolerance_px: Alignment tolerance in pixels
            move_gain: Proportional gain (meters per pixel offset)
            derivative_gain: Derivative gain for damping
            conf_threshold: YOLO confidence threshold
            max_velocity: Maximum linear velocity (m/s)
            visualization_scale: Scale factor for display window
            headless: Run without display window
            auto_align_at_startup: Enable auto-alignment at startup
            img_size: Camera image size (640x640)
            fps: Camera frame rate
            device_id: OAK device MxID (oak2)
            min_scan_height: Minimum Y pixel height for detection (only scan from this height and below)
        """
        self.model_path = model_path
        self.canbus_config_path = canbus_config_path

        # Alignment parameters
        self.target_reticle_x = target_reticle_x
        self.target_reticle_y = target_reticle_y
        self.tolerance_px = tolerance_px
        self.move_gain = move_gain
        self.derivative_gain = derivative_gain
        self.max_velocity = max_velocity

        # Detection parameters
        self.conf_threshold = conf_threshold
        self.min_scan_height = min_scan_height

        # Visualization
        self.visualization_scale = visualization_scale
        self.headless = headless

        # Camera parameters
        self.img_size = img_size
        self.fps = fps
        self.device_id = device_id

        # State
        self.auto_align_enabled = auto_align_at_startup
        self.last_offset_y = None
        self.desired_velocity = 0.0

        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.alignment_count = 0

    async def run(self):
        """Main run loop."""
        print(f"\n{'='*70}")
        print("oak2 COLLAR ALIGNMENT TESTER (DepthAI Pipeline)")
        print(f"{'='*70}")
        print(f"Model:            {self.model_path}")
        print(f"Device ID:        {self.device_id}")
        print(f"Target Reticle:   ({self.target_reticle_x}, {self.target_reticle_y})")
        print(f"Tolerance:        ±{self.tolerance_px}px")
        print(f"Move Gain (kp):   {self.move_gain} m/px")
        print(f"Derivative (kd):  {self.derivative_gain} m/px")
        print(f"Max Velocity:     {self.max_velocity} m/s")
        print(f"{'='*70}\n")

        # Load YOLO model
        if not self.model_path.exists():
            print(f"❌ Model not found: {self.model_path}")
            return

        print(f"Loading YOLO model...")
        self.model = YOLO(str(self.model_path))
        print(f"✓ Loaded model with {len(self.model.names)} classes: {list(self.model.names.values())}")

        # Warm up model
        print("Warming up model...")
        dummy_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        _ = self.model.predict(dummy_frame, conf=self.conf_threshold, verbose=False, imgsz=self.img_size)
        print("✓ Model ready\n")

        # Load canbus config
        if not self.canbus_config_path.exists():
            print(f"❌ canbus config not found: {self.canbus_config_path}")
            return

        canbus_config = proto_from_json_file(self.canbus_config_path, EventServiceConfig())
        print(f"✓ canbus config: {canbus_config.host}:{canbus_config.port}\n")

        # Create canbus client
        self.canbus_client = EventClient(canbus_config)

        # Signal Flask GUI that inference is active (use oak0 feed for display)
        set_inference_active(True)
        print("✓ Inference flag set (Flask will display oak2 frames on oak0 feed)")

        # Create depthai pipeline for oak2
        print("Creating DepthAI pipeline for oak2...")
        pipeline, xoutRgb = self._create_pipeline()

        # Create output queue BEFORE starting pipeline (V3 requirement)
        qRgb = xoutRgb.createOutputQueue(maxSize=1, blocking=False)
        print("✓ Pipeline created\n")

        print("🚀 Starting alignment test...")
        if not self.headless:
            print("\nControls:")
            print("  'a' - Enable auto-alignment (robot will move)")
            print("  's' - Stop robot (zero velocity)")
            print("  'q' - Quit")
        if self.auto_align_enabled:
            print("\n⚠️  AUTO-ALIGNMENT ENABLED - Robot will move to align!")
        print()

        start_time = time.time()

        # Start background task to send twist commands at 20 Hz
        twist_sender_task = asyncio.create_task(self._twist_command_sender())
        print("✓ Twist command sender started (20 Hz)\n")

        # Track frame timing
        last_frame_time = time.time()

        try:
            # Start pipeline
            pipeline.start()
            with pipeline:

                while pipeline.isRunning():
                    # Drain latest frame from queue
                    latestRgb = None
                    while qRgb.has():
                        latestRgb = qRgb.get()

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

                    # Run detection and alignment
                    await self._process_frame(frame, processing_fps)

                    # Yield to event loop to allow twist sender to run
                    await asyncio.sleep(0.001)

                    # Handle key presses (non-blocking check) - only if not headless
                    key = 255  # Default: no key pressed
                    if not self.headless:
                        try:
                            key = cv2.waitKey(1) & 0xFF
                        except:
                            key = 255

                    if key == ord('q'):
                        print("\n✓ User requested exit")
                        break
                    elif key == ord('a'):
                        self.auto_align_enabled = not self.auto_align_enabled
                        status = "ENABLED" if self.auto_align_enabled else "DISABLED"
                        print(f"\n{'='*70}")
                        print(f"AUTO-ALIGNMENT {status}")
                        print(f"{'='*70}\n")
                        if not self.auto_align_enabled:
                            await self._send_stop_command()
                    elif key == ord('s'):
                        print("\n⏹  STOP commanded")
                        await self._send_stop_command()
                        self.auto_align_enabled = False

        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        finally:
            # Cancel background task and stop robot
            twist_sender_task.cancel()
            try:
                await twist_sender_task
            except asyncio.CancelledError:
                pass
            await self._send_stop_command()

            # Clear inference flag so Flask GUI can resume normal operation
            set_inference_active(False)
            print("✓ Inference flag cleared")

        # Final stats
        elapsed = time.time() - start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        print(f"\n{'='*70}")
        print("TEST COMPLETE")
        print(f"{'='*70}")
        print(f"Frames processed:  {self.frame_count}")
        print(f"Detections found:  {self.detection_count}")
        print(f"Aligned frames:    {self.alignment_count}")
        print(f"Runtime:           {elapsed:.1f}s")
        print(f"Average FPS:       {fps:.1f}")
        print(f"{'='*70}\n")

        cv2.destroyAllWindows()

    def _create_pipeline(self):
        """Create OAK-D pipeline for oak2 RGB camera (DepthAI V3 API)."""
        # Create pipeline for specific device
        device = dai.Device(self.device_id)
        pipeline = dai.Pipeline(device)

        # RGB camera
        camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        # Request RGB output at specified size (returns Output object in V3)
        xoutRgb = camRgb.requestOutput((self.img_size, self.img_size))

        return pipeline, xoutRgb

    async def _process_frame(self, image: np.ndarray, fps: float = 0.0):
        """Process a single frame: detect, visualize, and optionally align."""

        h, w = image.shape[:2]

        # Run YOLO detection
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=self.img_size
        )

        # Extract detections - filter to ONLY class 0 (Collar) AND below min_scan_height
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls != 0:  # Only keep Collar detections
                    continue

                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()

                # Filter: only keep detections where the center Y is at or below min_scan_height
                center_y = (xyxy[1] + xyxy[3]) / 2
                if center_y < self.min_scan_height:
                    continue  # Skip detections above the minimum scan height

                detections.append({
                    'confidence': conf,
                    'bbox': xyxy
                })

        # Update stats
        if detections:
            self.detection_count += 1

        # Calculate offset and alignment status
        offset_y = None
        is_aligned = False
        velocity_cmd = 0.0

        if detections:
            # Use highest confidence detection
            best_det = max(detections, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best_det['bbox']

            # Calculate collar center
            collar_center_x = (x1 + x2) / 2
            collar_center_y = (y1 + y2) / 2

            # Calculate vertical offset from target reticle
            offset_y = collar_center_y - self.target_reticle_y

            # Check alignment
            is_aligned = abs(offset_y) <= self.tolerance_px
            if is_aligned:
                self.alignment_count += 1

            # Calculate derivative term (rate of change of offset) for damping
            if self.last_offset_y is not None:
                offset_derivative = offset_y - self.last_offset_y
            else:
                offset_derivative = 0.0

            # Update last offset for next iteration
            self.last_offset_y = offset_y

            # Calculate velocity command with derivative damping (oak2 faces FORWARD!)
            # Forward-facing camera: Positive offset_y (collar lower) → forward velocity (positive)
            # Negative offset_y (collar higher) → reverse velocity (negative)
            proportional_term = offset_y * self.move_gain  # Positive for forward-facing
            derivative_term = offset_derivative * self.derivative_gain

            velocity_cmd = np.clip(
                proportional_term - derivative_term,
                -self.max_velocity,
                self.max_velocity
            )

            # Apply minimum velocity threshold to overcome static friction
            # If commanded velocity is too small, boost it to minimum effective velocity
            MIN_VELOCITY = 0.045  # Minimum velocity to actually move the robot (m/s)
            if abs(velocity_cmd) > 0.001:  # Not zero
                if abs(velocity_cmd) < MIN_VELOCITY:
                    # Boost to minimum velocity while preserving direction
                    velocity_cmd = MIN_VELOCITY if velocity_cmd > 0 else -MIN_VELOCITY

            # Print debug info
            direction = "FORWARD" if velocity_cmd > 0 else "REVERSE"
            auto_status = "AUTO-ON" if self.auto_align_enabled else "AUTO-OFF"
            print(
                f"[Align] Offset: {offset_y:+.1f}px | "
                f"dOffset: {offset_derivative:+.1f}px | "
                f"Cmd: {velocity_cmd:+.4f}m/s | "
                f"{direction} | Aligned: {is_aligned} | {auto_status}"
            )

            # Update desired velocity if auto-align enabled
            if self.auto_align_enabled:
                if is_aligned:
                    # Already aligned, set zero velocity
                    self.desired_velocity = 0.0
                    print(f"[Align] ✓ ALIGNED within tolerance ({self.tolerance_px:.0f}px)")
                else:
                    # Set alignment velocity
                    self.desired_velocity = velocity_cmd
                    print(f"[Align] → Setting velocity: {self.desired_velocity:+.4f}m/s")
            else:
                self.desired_velocity = 0.0
        else:
            # No detection
            if self.auto_align_enabled:
                # Stop if no collar detected
                self.desired_velocity = 0.0

        # Create visualization frame and share with Flask (always create, even in headless mode)
        vis_frame = self._create_visualization_frame(image, detections, offset_y, is_aligned, velocity_cmd, fps)

        # Share frame with Flask GUI (using oak0 feed)
        set_oak0_frame(vis_frame)

        # Display locally only if not headless
        if not self.headless:
            cv2.namedWindow("oak2 Alignment Test", cv2.WINDOW_NORMAL)
            cv2.imshow("oak2 Alignment Test", vis_frame)

    def _create_visualization_frame(self, image: np.ndarray, detections: list, offset_y: float, is_aligned: bool, velocity_cmd: float, fps: float = 0.0):
        """Create visualization frame with detection overlay."""

        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # Draw FPS and frame counter
        fps_text = f"FPS: {fps:.1f} | Frame #{self.frame_count}"
        cv2.putText(vis_image, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

        # Draw target reticle (crosshair)
        reticle_color = (0, 255, 0) if is_aligned else (0, 165, 255)  # Green if aligned, orange otherwise
        reticle_size = 30
        reticle_thickness = 2

        # Horizontal line
        cv2.line(
            vis_image,
            (self.target_reticle_x - reticle_size, self.target_reticle_y),
            (self.target_reticle_x + reticle_size, self.target_reticle_y),
            reticle_color,
            reticle_thickness
        )
        # Vertical line
        cv2.line(
            vis_image,
            (self.target_reticle_x, self.target_reticle_y - reticle_size),
            (self.target_reticle_x, self.target_reticle_y + reticle_size),
            reticle_color,
            reticle_thickness
        )
        # Center dot
        cv2.circle(vis_image, (self.target_reticle_x, self.target_reticle_y), 5, reticle_color, -1)

        # Draw tolerance zone
        tolerance_rect_color = (100, 255, 100) if is_aligned else (100, 100, 100)
        cv2.rectangle(
            vis_image,
            (0, self.target_reticle_y - self.tolerance_px),
            (vis_image.shape[1], self.target_reticle_y + self.tolerance_px),
            tolerance_rect_color,
            1
        )

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            conf = det['confidence']

            # Bounding box
            bbox_color = (0, 255, 0) if is_aligned else (0, 255, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, 2)

            # Label
            label = f"Collar {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

            # Draw collar center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(vis_image, (center_x, center_y), 8, (255, 0, 255), -1)

            # Draw line from collar center to reticle
            cv2.line(vis_image, (center_x, center_y), (self.target_reticle_x, self.target_reticle_y),
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Status overlay
        status_y = 60
        line_height = 30

        # Auto-align status
        align_status = "AUTO-ALIGN: ON" if self.auto_align_enabled else "AUTO-ALIGN: OFF"
        align_color = (0, 255, 0) if self.auto_align_enabled else (0, 0, 255)
        cv2.putText(vis_image, align_status, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, align_color, 2)
        status_y += line_height

        # Detection count
        cv2.putText(vis_image, f"Detections: {len(detections)}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += line_height

        # Offset
        if offset_y is not None:
            offset_text = f"Offset Y: {offset_y:+.1f}px"
            offset_color = (0, 255, 0) if is_aligned else (0, 165, 255)
            cv2.putText(vis_image, offset_text, (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)
            status_y += line_height

        # Alignment status
        if is_aligned:
            cv2.putText(vis_image, "STATUS: ALIGNED ✓", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif offset_y is not None:
            direction = "FORWARD" if velocity_cmd > 0 else "REVERSE"
            cv2.putText(vis_image, f"STATUS: {direction} ↕", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        status_y += line_height

        # Velocity command
        vel_text = f"Velocity: {self.desired_velocity:+.3f} m/s"
        cv2.putText(vis_image, vel_text, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize for display if needed
        if self.visualization_scale != 1.0:
            display_width = int(vis_image.shape[1] * self.visualization_scale)
            display_height = int(vis_image.shape[0] * self.visualization_scale)
            vis_image = cv2.resize(vis_image, (display_width, display_height))

        return vis_image

    async def _twist_command_sender(self):
        """Background task that sends twist commands at 20 Hz."""
        print("[CAN] Twist sender task started")
        last_logged_velocity = None
        while True:
            try:
                # Send the current desired velocity
                twist = Twist2d()
                twist.linear_velocity_x = self.desired_velocity
                twist.linear_velocity_y = 0.0
                twist.angular_velocity = 0.0

                await self.canbus_client.request_reply("/twist", twist)

                # Log velocity changes (avoid spam)
                if self.desired_velocity != last_logged_velocity:
                    if abs(self.desired_velocity) > 0.001:
                        print(f"[CAN] Sending twist: {self.desired_velocity:+.4f}m/s")
                    elif last_logged_velocity is not None and abs(last_logged_velocity) > 0.001:
                        print(f"[CAN] Sending twist: 0.0000m/s (STOPPED)")
                    last_logged_velocity = self.desired_velocity

                # Send at 20 Hz
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                # Task is being cancelled, exit gracefully
                break
            except Exception as e:
                print(f"⚠️  Failed to send twist command: {e}")
                await asyncio.sleep(0.05)

    async def _send_twist_command(self, linear_velocity_x: float):
        """Send twist command to CAN bus."""
        twist = Twist2d()
        twist.linear_velocity_x = linear_velocity_x
        twist.linear_velocity_y = 0.0
        twist.angular_velocity = 0.0

        try:
            await self.canbus_client.request_reply("/twist", twist)
        except Exception as e:
            print(f"⚠️  Failed to send twist command: {e}")

    async def _send_stop_command(self):
        """Send zero velocity command."""
        await self._send_twist_command(0.0)


async def main():
    parser = argparse.ArgumentParser(
        description="Test oak2 visual servoing alignment with YOLO collar detection (DepthAI pipeline)"
    )

    # Model config
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent / "best1.engine",
        help="Path to YOLO model (.pt or .engine)"
    )

    # CAN bus config
    parser.add_argument(
        "--canbus-config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "canbus_config.json",
        help="Path to canbus service config"
    )

    # Alignment parameters
    parser.add_argument(
        "--target-x",
        type=int,
        default=330,
        help="Target reticle X position (pixels, center for 640x640)"
    )
    parser.add_argument(
        "--target-y",
        type=int,
        default=260,
        help="Target reticle Y position (pixels, center for 640x640)"
    )
    parser.add_argument(
        "--tolerance-px",
        type=int,
        default=8,
        help="Alignment tolerance (pixels)"
    )
    parser.add_argument(
        "--move-gain",
        type=float,
        default=0.001,
        help="Proportional gain: meters per pixel offset"
    )
    parser.add_argument(
        "--derivative-gain",
        type=float,
        default=0.02,
        help="Derivative gain for damping oscillations"
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.05,
        help="Maximum linear velocity (m/s)"
    )

    # Detection parameters
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--min-scan-height",
        type=int,
        default=200,
        help="Minimum Y pixel height for detection (only scan from this height and below)"
    )

    # Camera parameters
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Camera image size (640x640)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Camera frame rate"
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="14442C10D14CFFD600",
        help="OAK device MxID (oak2)"
    )

    # Visualization
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display window scale factor"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without local visualization window"
    )
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help="Enable auto-alignment at startup (robot will move automatically)"
    )

    args = parser.parse_args()

    # Create tester
    tester = oak2AlignmentTester(
        model_path=args.model_path,
        canbus_config_path=args.canbus_config,
        target_reticle_x=args.target_x,
        target_reticle_y=args.target_y,
        tolerance_px=args.tolerance_px,
        move_gain=args.move_gain,
        derivative_gain=args.derivative_gain,
        conf_threshold=args.conf,
        max_velocity=args.max_velocity,
        visualization_scale=args.scale,
        headless=args.headless,
        auto_align_at_startup=args.auto_align,
        img_size=args.img_size,
        fps=args.fps,
        device_id=args.device_id,
        min_scan_height=args.min_scan_height,
    )

    # Run
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
