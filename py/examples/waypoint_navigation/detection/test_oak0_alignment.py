#!/usr/bin/env python3
"""
Standalone test script for oak0 collar alignment using visual servoing.

This script:
1. Subscribes to oak0 RGB camera feed (rear-facing)
2. Runs YOLO detection to find collar bounding box
3. Compares collar center to target reticle position
4. Sends CAN bus twist commands to align robot forward/backward
5. Displays real-time visualization with reticle and bbox

Camera orientation: oak0 faces BACKWARDS
- Collar higher in frame → Robot reverses (negative velocity)
- Collar lower in frame → Robot drives forward (positive velocity)

Usage:
    python test_oak0_alignment.py --move-gain 0.001 --tolerance-px 40

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
import hashlib

# Add parent directory to import detection classes
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import frame sharing for Flask GUI
from utils.oak0_camera_cache import set_oak0_frame, set_inference_active

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri
from farm_ng.canbus.canbus_pb2 import Twist2d

# Import YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False
    sys.exit(1)


class Oak0AlignmentTester:
    """Visual servoing alignment tester for oak0 collar detection."""

    def __init__(
        self,
        model_path: Path,
        oak0_config_path: Path,
        canbus_config_path: Path,
        target_reticle_x: int = 573,
        target_reticle_y: int = 470,
        tolerance_px: int = 40,
        move_gain: float = 0.001,
        derivative_gain: float = 0.002,
        conf_threshold: float = 0.3,
        max_velocity: float = 0.2,
        visualization_scale: float = 0.6,
        headless: bool = False,
        auto_align_at_startup: bool = False,
        roi_top_pct: float = 0.0,
        roi_bottom_pct: float = 1.0,
        roi_left_pct: float = 0.0,
        roi_right_pct: float = 1.0,
        move_interval: float = 0.0,
        disable_flask: bool = False,
    ):
        """
        Initialize the oak0 alignment tester.

        Args:
            model_path: Path to YOLO model (.pt or .engine)
            oak0_config_path: Path to oak0 camera service config
            canbus_config_path: Path to canbus service config
            move_interval: If >0, use snapshot mode (move, pause, capture)
            target_reticle_x: Target X pixel position for collar center
            target_reticle_y: Target Y pixel position for collar center
            tolerance_px: Alignment tolerance in pixels
            move_gain: Proportional gain (meters per pixel offset)
            conf_threshold: YOLO confidence threshold
            max_velocity: Maximum linear velocity (m/s)
            visualization_scale: Scale factor for display window
        """
        self.model_path = model_path
        self.oak0_config_path = oak0_config_path
        self.canbus_config_path = canbus_config_path

        # Alignment parameters
        self.target_reticle_x = target_reticle_x
        self.target_reticle_y = target_reticle_y
        self.tolerance_px = tolerance_px
        self.move_gain = move_gain
        self.max_velocity = max_velocity

        # Detection parameters
        self.conf_threshold = conf_threshold

        # Visualization
        self.visualization_scale = visualization_scale
        self.headless = headless
        self.disable_flask = disable_flask

        # ROI (Region of Interest) for detection
        self.roi_top_pct = roi_top_pct
        self.roi_bottom_pct = roi_bottom_pct
        self.roi_left_pct = roi_left_pct
        self.roi_right_pct = roi_right_pct

        # Movement control
        self.move_interval = move_interval  # Snapshot mode interval
        self.snapshot_mode = move_interval > 0  # Use snapshot mode if interval specified

        # State
        self.auto_align_enabled = auto_align_at_startup
        self.last_offset_y = None
        self.last_velocity = 0.0
        self.desired_velocity = 0.0  # Target velocity to send
        self.latest_frame = None  # Store latest frame for processing
        self.latest_vis_frame = None  # Store latest visualization for Flask GUI updates
        self.processing_frame = False  # Flag to track if YOLO is running

        # Derivative damping for oscillation prevention
        self.derivative_gain = derivative_gain  # kd - derivative damping coefficient

        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.alignment_count = 0
        self.frames_skipped = 0

    async def run(self):
        """Main run loop."""
        print(f"\n{'='*70}")
        print("OAK0 COLLAR ALIGNMENT TESTER")
        print(f"{'='*70}")
        print(f"Model:            {self.model_path}")
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

        # Warm up model with dummy inference to initialize all TensorRT contexts
        print("Warming up model (initializing GPU contexts)...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(dummy_frame, conf=self.conf_threshold, verbose=False, imgsz=640)
        print("✓ Model ready\n")

        # Load configs
        if not self.oak0_config_path.exists():
            print(f"❌ oak0 config not found: {self.oak0_config_path}")
            return

        if not self.canbus_config_path.exists():
            print(f"❌ canbus config not found: {self.canbus_config_path}")
            return

        oak0_config = proto_from_json_file(self.oak0_config_path, EventServiceConfig())
        canbus_config = proto_from_json_file(self.canbus_config_path, EventServiceConfig())

        print(f"✓ oak0 config: {oak0_config.host}:{oak0_config.port}")
        print(f"✓ canbus config: {canbus_config.host}:{canbus_config.port}\n")

        # Create clients
        self.oak0_client = EventClient(oak0_config)
        self.canbus_client = EventClient(canbus_config)

        # Create subscription - camera delivers at ~10 FPS
        # Process all frames since rate is already limited
        subscription = SubscribeRequest(
            uri=Uri(path="/rgb", query="service_name=oak/0"),
            every_n=1  # Camera service delivers at 10 FPS, process all
        )

        # Signal Flask GUI that inference is active (don't overwrite our processed frames)
        if not self.disable_flask:
            set_inference_active(True)
            print("✓ Inference flag set (Flask GUI will not overwrite frames)")

            # Wait a moment for Flask thread to see the flag and stop writing
            await asyncio.sleep(0.5)

            # Clear any existing raw frames from cache to avoid showing uninferred frames
            from pathlib import Path
            cache_file = Path("/tmp/amiga_oak0_frame.jpg")
            if cache_file.exists():
                cache_file.unlink()
                print("✓ Cleared old frames from cache")

            # Verify the flag is set correctly
            from utils.oak0_camera_cache import is_inference_active
            if is_inference_active():
                print("✓ Flask GUI oak0 updater is paused\n")
            else:
                print("⚠️  Warning: Inference flag not detected by Flask GUI\n")
        else:
            print("✓ Flask streaming DISABLED for maximum performance\n")

        print("🚀 Starting alignment test...")
        if self.snapshot_mode:
            print(f"📸 SNAPSHOT MODE ENABLED: Move for {self.move_interval}s, pause for fresh frame")
            print(f"   This compensates for camera latency by waiting for current visual feedback")
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

        # Track frame timing
        last_frame_time = time.time()
        frame_receive_time = time.time()
        camera_frame_count = 0
        frames_drained = 0
        stale_frames_skipped = 0

        # Frame buffer for draining
        latest_frame = None
        latest_event = None
        latest_message = None

        # Latency tracking
        max_acceptable_latency = 0.5  # Skip frames older than 500ms

        # Frame uniqueness tracking (detect duplicate/repeated frames)
        last_frame_hash = None
        duplicate_frames = 0

        try:
            async for event, message in self.oak0_client.subscribe(subscription, decode=True):
                camera_frame_count += 1
                current_time = time.time()

                # Check frame timestamp to detect stale frames
                # event.header.stamp is in nanoseconds
                if hasattr(event, 'header') and hasattr(event.header, 'stamp'):
                    frame_timestamp_sec = event.header.stamp.seconds + event.header.stamp.nanos / 1e9
                    frame_age = current_time - frame_timestamp_sec

                    # Skip stale frames (older than max_acceptable_latency)
                    if frame_age > max_acceptable_latency:
                        stale_frames_skipped += 1
                        if stale_frames_skipped % 10 == 0:
                            print(f"⚠️  Frame latency: {frame_age:.2f}s (skipped {stale_frames_skipped} stale frames)")
                        continue

                # Store this frame as latest
                latest_event = event
                latest_message = message

                # If YOLO is still processing, keep draining but don't process yet
                if self.processing_frame:
                    frames_drained += 1
                    continue

                # Decode the latest frame
                image = cv2.imdecode(np.frombuffer(latest_message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue

                # Check if this is a duplicate frame (same image content as previous)
                frame_hash = hashlib.md5(image.tobytes()).hexdigest()
                if frame_hash == last_frame_hash:
                    duplicate_frames += 1
                    if duplicate_frames % 10 == 0:
                        print(f"⚠️  Duplicate frame detected ({duplicate_frames} total duplicates)")
                    continue  # Skip duplicate frames
                last_frame_hash = frame_hash

                # Track camera frame arrival time
                camera_fps = 1.0 / (current_time - frame_receive_time) if (current_time - frame_receive_time) > 0 else 0
                frame_receive_time = current_time

                self.frame_count += 1
                self.processing_frame = True

                try:
                    # Calculate processing FPS
                    processing_fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
                    last_frame_time = current_time

                    # Log timing every 5 frames
                    if self.frame_count % 5 == 0:
                        print(f"[Timing] Cam: {camera_fps:.1f} FPS | Proc: {processing_fps:.1f} FPS | Skipped: {self.frames_skipped} | Drained: {frames_drained} | Stale: {stale_frames_skipped} | Dupes: {duplicate_frames}")
                        frames_drained = 0  # Reset drain counter after logging
                        stale_frames_skipped = 0  # Reset stale counter after logging
                        duplicate_frames = 0  # Reset duplicate counter after logging

                    # Run detection and alignment
                    await self._process_frame(image, processing_fps)
                finally:
                    self.processing_frame = False

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

            # Clear inference flag so Flask GUI can resume writing raw oak0 frames
            if not self.disable_flask:
                set_inference_active(False)
                print("✓ Inference flag cleared (Flask GUI can resume)")

        # Final stats
        elapsed = time.time() - start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        print(f"\n{'='*70}")
        print("TEST COMPLETE")
        print(f"{'='*70}")
        print(f"Frames processed:  {self.frame_count}")
        print(f"Frames skipped:    {self.frames_skipped}")
        print(f"Detections found:  {self.detection_count}")
        print(f"Aligned frames:    {self.alignment_count}")
        print(f"Runtime:           {elapsed:.1f}s")
        print(f"Average FPS:       {fps:.1f}")
        print(f"{'='*70}\n")

        cv2.destroyAllWindows()

    async def _process_frame(self, image: np.ndarray, fps: float = 0.0):
        """Process a single frame: detect, visualize, and optionally align."""

        h, w = image.shape[:2]

        # Extract ROI for detection
        roi_top = int(h * self.roi_top_pct)
        roi_bottom = int(h * self.roi_bottom_pct)
        roi_left = int(w * self.roi_left_pct)
        roi_right = int(w * self.roi_right_pct)

        # Crop to ROI
        roi_image = image[roi_top:roi_bottom, roi_left:roi_right]
        roi_h, roi_w = roi_image.shape[:2]

        # TensorRT engines require exact input dimensions
        # For a 640x640 engine, resize the ENTIRE ROI to 640x640
        # Since ROI is now 1:1 aspect ratio (1080x1080), resizing maintains proportions
        crop_size = 640

        # Use the entire ROI and resize it to 640x640
        crop_image = cv2.resize(roi_image, (crop_size, crop_size))

        # The inference region is the entire ROI
        crop_offset_x = roi_left
        crop_offset_y = roi_top
        crop_dim_w = roi_w
        crop_dim_h = roi_h

        # Scale factors for mapping bboxes back to original ROI size
        scale_x = roi_w / crop_size
        scale_y = roi_h / crop_size

        # Run YOLO detection on 640x640 crop (no resizing needed by YOLO)
        # Note: classes=[0] doesn't work well with TensorRT engines
        results = self.model.predict(
            crop_image,
            conf=self.conf_threshold,
            verbose=False
        )

        # Extract detections - filter to ONLY class 0 (Collar)
        # Map coordinates back to full frame
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls != 0:  # Only keep Collar detections
                    continue

                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()

                # Map crop coordinates back to full frame
                # Scale if resized, then offset by crop position
                x1_full = xyxy[0] * scale_x + crop_offset_x
                y1_full = xyxy[1] * scale_y + crop_offset_y
                x2_full = xyxy[2] * scale_x + crop_offset_x
                y2_full = xyxy[3] * scale_y + crop_offset_y

                detections.append({
                    'confidence': conf,
                    'bbox': np.array([x1_full, y1_full, x2_full, y2_full])
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

            # Calculate collar center (in full-resolution coordinates)
            collar_center_x = (x1 + x2) / 2
            collar_center_y = (y1 + y2) / 2

            # Scale target reticle to full-resolution for comparison
            # (target_reticle coordinates are in display window space)
            scale_factor = 1.0 / self.visualization_scale
            target_reticle_y_full = self.target_reticle_y * scale_factor
            tolerance_px_full = self.tolerance_px * scale_factor

            # Calculate vertical offset from target reticle (both in full-res coordinates)
            offset_y = collar_center_y - target_reticle_y_full

            # Check alignment
            is_aligned = abs(offset_y) <= tolerance_px_full
            if is_aligned:
                self.alignment_count += 1

            # Calculate derivative term (rate of change of offset) for damping
            if self.last_offset_y is not None:
                offset_derivative = offset_y - self.last_offset_y
            else:
                offset_derivative = 0.0

            # Update last offset for next iteration
            self.last_offset_y = offset_y

            # Calculate velocity command with derivative damping (oak0 faces BACKWARDS!)
            # Positive offset_y (collar lower in frame) → forward velocity (positive)
            # Negative offset_y (collar higher in frame) → reverse velocity (negative)
            # Derivative term opposes rapid changes, preventing overshoot and oscillation
            proportional_term = offset_y * self.move_gain
            derivative_term = offset_derivative * self.derivative_gain

            velocity_cmd = np.clip(
                proportional_term - derivative_term,
                -self.max_velocity,
                self.max_velocity
            )

            # Print debug info
            direction = "FORWARD" if offset_y > 0 else "REVERSE"
            print(
                f"[Align] Offset: {offset_y:+.1f}px | "
                f"dOffset: {offset_derivative:+.1f}px | "
                f"Cmd: {velocity_cmd:+.4f}m/s | "
                f"{direction} | Aligned: {is_aligned}"
            )

            # Update desired velocity if auto-align enabled
            if self.auto_align_enabled:
                if is_aligned:
                    # Already aligned, set zero velocity
                    self.desired_velocity = 0.0
                    print(f"[Align] ✓ ALIGNED within tolerance ({tolerance_px_full:.0f}px)")
                else:
                    # Set alignment velocity
                    self.desired_velocity = velocity_cmd

                    # Snapshot mode: Send command, wait for movement, then stop to capture fresh frame
                    if self.snapshot_mode:
                        print(f"[Snapshot] Moving {velocity_cmd:.3f} m/s for {self.move_interval}s...")
                        await asyncio.sleep(self.move_interval)
                        self.desired_velocity = 0.0  # Stop robot

                        # Shorter wait - just let camera catch up (no Flask refresh needed)
                        wait_time = 2.0  # Reduced from 5s to 2s
                        print(f"[Snapshot] Stopped. Waiting {wait_time}s for camera...")
                        await asyncio.sleep(wait_time)
                        print(f"[Snapshot] Ready for next frame")
            else:
                self.desired_velocity = 0.0
        else:
            # No detection
            if self.auto_align_enabled:
                # Stop if no collar detected
                self.desired_velocity = 0.0

        # Create visualization only if needed (for display or Flask)
        if not self.headless or not self.disable_flask:
            # Pass inference region info for visualization (shows what the model sees)
            inference_region = {
                'offset_x': crop_offset_x,
                'offset_y': crop_offset_y,
                'width': crop_dim_w,  # Actual ROI width
                'height': crop_dim_h  # Actual ROI height
            }
            vis_frame = self._create_visualization_frame(image, detections, offset_y, is_aligned, velocity_cmd, fps, inference_region)

            # Share frame with Flask GUI (unless disabled for performance)
            if not self.disable_flask:
                set_oak0_frame(vis_frame)
                # Store latest vis_frame for continuous updates during snapshot wait
                self.latest_vis_frame = vis_frame

            # Display locally only if not headless
            if not self.headless:
                cv2.namedWindow("oak0 Alignment Test", cv2.WINDOW_NORMAL)
                cv2.imshow("oak0 Alignment Test", vis_frame)

    def _create_visualization_frame(self, image: np.ndarray, detections: list, offset_y: float, is_aligned: bool, velocity_cmd: float, fps: float = 0.0, crop_region: dict = None):
        """Create visualization frame with detection overlay (for display and Flask GUI)."""

        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # Draw ROI rectangle
        roi_top = int(h * self.roi_top_pct)
        roi_bottom = int(h * self.roi_bottom_pct)
        roi_left = int(w * self.roi_left_pct)
        roi_right = int(w * self.roi_right_pct)

        # Draw ROI boundary (cyan color)
        cv2.rectangle(vis_image, (roi_left, roi_top), (roi_right, roi_bottom), (255, 255, 0), 2)
        cv2.putText(vis_image, "Detection ROI", (roi_left + 5, roi_top + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw inference region (what gets resized to 640x640 and fed to model) - magenta/pink color
        if crop_region is not None:
            crop_x1 = int(crop_region['offset_x'])
            crop_y1 = int(crop_region['offset_y'])
            crop_x2 = crop_x1 + crop_region['width']
            crop_y2 = crop_y1 + crop_region['height']
            cv2.rectangle(vis_image, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 255), 3)
            label = f"Inference ({crop_region['width']}x{crop_region['height']}->640x640)"
            cv2.putText(vis_image, label, (crop_x1 + 5, crop_y1 + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Note: Target reticle coordinates are specified for the DISPLAY window,
        # but we need to scale them up to the original image resolution for drawing
        scale_factor = 1.0 / self.visualization_scale
        reticle_x_full = int(self.target_reticle_x * scale_factor)
        reticle_y_full = int(self.target_reticle_y * scale_factor)

        # Draw FPS, frame counter, and timestamp in top right
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        fps_text = f"FPS: {fps:.1f} | Frame #{self.frame_count} | {timestamp}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fps_x = vis_image.shape[1] - text_size[0] - 10
        cv2.putText(vis_image, fps_text, (fps_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw target reticle (crosshair) at scaled-up coordinates
        reticle_color = (0, 255, 0) if is_aligned else (0, 165, 255)  # Green if aligned, orange otherwise
        reticle_size = int(30 * scale_factor)  # Scale reticle size too
        reticle_thickness = max(2, int(2 * scale_factor))

        # Horizontal line
        cv2.line(
            vis_image,
            (reticle_x_full - reticle_size, reticle_y_full),
            (reticle_x_full + reticle_size, reticle_y_full),
            reticle_color,
            reticle_thickness
        )
        # Vertical line
        cv2.line(
            vis_image,
            (reticle_x_full, reticle_y_full - reticle_size),
            (reticle_x_full, reticle_y_full + reticle_size),
            reticle_color,
            reticle_thickness
        )
        # Center dot
        cv2.circle(vis_image, (reticle_x_full, reticle_y_full), int(5 * scale_factor), reticle_color, -1)

        # Draw tolerance zone (scaled)
        tolerance_px_full = int(self.tolerance_px * scale_factor)
        tolerance_rect_color = (100, 255, 100) if is_aligned else (100, 100, 100)
        cv2.rectangle(
            vis_image,
            (0, reticle_y_full - tolerance_px_full),
            (vis_image.shape[1], reticle_y_full + tolerance_px_full),
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

            # Draw line from collar center to reticle (use scaled reticle coordinates)
            cv2.line(vis_image, (center_x, center_y), (reticle_x_full, reticle_y_full),
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Status overlay
        status_y = 30
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
            direction = "FORWARD" if offset_y > 0 else "REVERSE"
            cv2.putText(vis_image, f"STATUS: {direction} ↕", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        status_y += line_height

        # Velocity command
        vel_text = f"Velocity: {self.desired_velocity:+.3f} m/s"
        cv2.putText(vis_image, vel_text, (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize for display
        display_width = int(vis_image.shape[1] * self.visualization_scale)
        display_height = int(vis_image.shape[0] * self.visualization_scale)
        vis_image_resized = cv2.resize(vis_image, (display_width, display_height))

        # Return visualization frame
        return vis_image_resized

    async def _twist_command_sender(self):
        """Background task that sends twist commands at 20 Hz."""
        while True:
            try:
                # Send the current desired velocity
                twist = Twist2d()
                twist.linear_velocity_x = self.desired_velocity
                twist.linear_velocity_y = 0.0
                twist.angular_velocity = 0.0

                await self.canbus_client.request_reply("/twist", twist)
                self.last_velocity = self.desired_velocity

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
        description="Test oak0 visual servoing alignment with YOLO collar detection"
    )

    # Model config
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent / "best1.engine",
        help="Path to YOLO model (.pt or .engine)"
    )

    # Camera config
    parser.add_argument(
        "--oak0-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "camera_client" / "service_config.json",
        help="Path to oak0 camera service config"
    )

    # CAN bus config
    parser.add_argument(
        "--canbus-config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "canbus_config.json",
        help="Path to canbus service config"
    )

    # Alignment parameters
    # NOTE: Target coordinates are in DISPLAY window space (after 0.6x scaling)
    # Display window is 1152x648 (1920x1080 × 0.6)
    # Center X: 576, Center Y: 324
    parser.add_argument(
        "--target-x",
        type=int,
        default=576,
        help="Target reticle X position in display coordinates (pixels, center=576 for 1920x1080 camera)"
    )
    parser.add_argument(
        "--target-y",
        type=int,
        default=470,
        help="Target reticle Y position in display coordinates (pixels, lower than center for ground-level targets)"
    )
    parser.add_argument(
        "--tolerance-px",
        type=int,
        default=20,
        help="Alignment tolerance (pixels, ~2cm physical)"
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
        default=0.002,
        help="Derivative gain for damping oscillations (prevents overshoot)"
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

    # Visualization
    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Display window scale factor"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without local visualization window (faster, no X11 required)"
    )
    parser.add_argument(
        "--disable-flask",
        action="store_true",
        help="Disable Flask GUI streaming for maximum performance (no web visualization)"
    )
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help="Enable auto-alignment at startup (robot will move automatically)"
    )
    parser.add_argument(
        "--move-interval",
        type=float,
        default=0.0,
        help="If >0, use snapshot mode: move for this duration (seconds), then pause to capture fresh frame. Recommended: 0.5-1.0s for latency compensation"
    )

    # ROI parameters
    # Default: 1080x1080 square centered horizontally (perfect 1:1 aspect ratio for TensorRT)
    # For 1920x1080 camera: creates centered 1080x1080 square
    # Left: 420px (0.21875), Right: 1500px (0.78125) → exactly 1080px wide
    # This 1080x1080 square gets downsampled to 640x640 with no aspect ratio distortion
    parser.add_argument(
        "--roi-top",
        type=float,
        default=0.0,
        help="ROI top edge as percentage (0.0-1.0, 0.0=top of frame)"
    )
    parser.add_argument(
        "--roi-bottom",
        type=float,
        default=1.0,
        help="ROI bottom edge as percentage (0.0-1.0, 1.0=bottom of frame)"
    )
    parser.add_argument(
        "--roi-left",
        type=float,
        default=0.21875,
        help="ROI left edge as percentage (0.0-1.0, 0.0=left of frame)"
    )
    parser.add_argument(
        "--roi-right",
        type=float,
        default=0.78125,
        help="ROI right edge as percentage (0.0-1.0, 1.0=right of frame)"
    )

    args = parser.parse_args()

    # Create tester
    tester = Oak0AlignmentTester(
        model_path=args.model_path,
        oak0_config_path=args.oak0_config,
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
        roi_top_pct=args.roi_top,
        roi_bottom_pct=args.roi_bottom,
        roi_left_pct=args.roi_left,
        roi_right_pct=args.roi_right,
        move_interval=args.move_interval,
        disable_flask=args.disable_flask,
    )

    # Run
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
