# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import math
import socket
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, EventServiceConfigList
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.track.track_pb2 import (
    RobotStatus,
    Track,
    TrackFollowerState,
    TrackFollowRequest,
    TrackStatusEnum,
)
from utils.actuator import BaseActuator, NullActuator, CanHBridgeActuator
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty
from motion_planner import MotionPlanner
from utils.navigation_manager import NavigationManager
from utils.multiclient import MultiClientSubscriber as multi
from utils.pose_cache import set_latest_pose
from utils.navigation_state import set_navigation_state
from utils.canbus import imu_wiggle

logger = logging.getLogger("Navigation Manager")

def setup_signal_handlers(nav_manager: NavigationManager) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"\nReceived signal {signum}, initiating shutdown...")
        nav_manager.shutdown_requested = True

        if nav_manager.main_task and not nav_manager.main_task.done():
            nav_manager.main_task.cancel()

        if hasattr(signal_handler, "call_count"):
            signal_handler.call_count += 1
            if signal_handler.call_count > 1:
                logger.info("Second signal received, forcing exit")
                sys.exit(1)
        else:
            signal_handler.call_count = 1

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def _update_pose_from_filter(filter_msg) -> None:
    """
    Call this from your existing filter/localization callback.
    Replace field accessors below to match your message.
    """
    # Example mappings (adjust to your types):
    # If you have a Pose3F64: pose.a_from_b.translation = [y, x, z] or [x,y,z] depending on your convention.
    x_m = float(filter_msg.pose.translation[0])  # map/world X (meters)
    y_m = float(filter_msg.pose.translation[1])  # map/world Y (meters)

    # Heading (yaw, radians). If you have a rotation SO3, the z-yaw is log()[-1]
    # yaw_rad = float(filter_msg.pose.rotation.log()[-1])
    # or if heading is provided directly:
    yaw_rad = float(getattr(filter_msg, "pose_yaw_rad", 0.0))

    # Filter convergence status
    converged = bool(getattr(filter_msg, "has_converged", False))

    set_latest_pose(x_m, y_m, yaw_rad, converged)
        
# Vision configuration
VISION_SEARCH_RADIUS_M = 1.0  # Search radius around CSV waypoints for cone detection
VISION_WAIT_TIMEOUT_S = 10.0  # How long to wait for cone detection before skipping waypoint

def is_vision_enabled() -> bool:
    """Check if detectionPlot.py is running to determine if vision mode is active."""
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "detectionPlot.py"],
            capture_output=True,
            text=True,
            timeout=1.0
        )
        return result.returncode == 0  # Returns 0 if process found
    except Exception:
        return False

# TODO: Refactor into a separate module
async def vision_goal_listener(motion_planner, controller_client, nav_manager, proximity_m=2.0):
    """
    Listens for UDP 'cone_goal' messages from detectionPlot.py and overrides waypoint positions.

    NEW BEHAVIOR (Phase 2):
    - Continuously buffers collar detections for each waypoint from 2-6m distance
    - motion_planner.next_track_segment() queries buffer BEFORE creating approach segment
    - Approach segment aims directly at detected collar (not CSV waypoint)
    - Robot still refines position at 1.5m for final accuracy

    When vision is enabled (detectionPlot.py is running):
    - CSV waypoints act as 'search zone centers' with radius VISION_SEARCH_RADIUS_M
    - When a cone is detected within the search radius, it's added to the buffer
    - Navigation manager handles all execution, deployment, and state management
    - Robot drives to cones instead of CSV waypoints when vision is enabled

    When vision is disabled:
    - Robot drives directly to CSV waypoint positions (normal behavior)
    """

    # Track when cone was last detected in search zone for current waypoint
    if not hasattr(nav_manager, "cone_detected_for_current_wp"):
        nav_manager.cone_detected_for_current_wp = False
    if not hasattr(nav_manager, "current_waypoint_start_time"):
        nav_manager.current_waypoint_start_time = None

    # Track which waypoint index we last successfully triggered vision for
    if not hasattr(nav_manager, "vision_completed_waypoints"):
        nav_manager.vision_completed_waypoints = set()

    # Track all cone detections for visualization
    if not hasattr(nav_manager, "cone_detections"):
        nav_manager.cone_detections = []

    # NEW: Vision detection buffer for pre-approach planning
    # Structure: {waypoint_idx: [(timestamp, x, y, confidence, distance), ...]}
    if not hasattr(motion_planner, "vision_detection_buffer"):
        motion_planner.vision_detection_buffer = {}

    # Buffer retention time (keep detections from last 10 seconds)
    BUFFER_RETENTION_S = 10.0

    # --- socket setup -------------------------------------------------------
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception: pass
    sock.bind(("127.0.0.1", 41234))
    sock.settimeout(1.0)

    last_goal = None
    last_sent_t = 0.0
    MIN_DIST_DELTA = 0.35
    MIN_PERIOD_S   = 0.8

    # print("Started vision goal listener")

    try:
        while True:
            # ---- receive next message ----
            try:
                data, _ = await loop.run_in_executor(None, sock.recvfrom, 4096)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[VISION] recv error: {e}")
                continue

            # ---- parse ----
            try:
                msg = json.loads(data.decode())
                # Accept class_id 0 (collars from visual prompting)
                # Previously filtered for class_id 6 (Safety Cone)
                if msg.get("class_id") != int(0):
                    continue
                name = str(msg["class_name"])
                x = float(msg["x_fwd_m"])
                y = float(msg["y_left_m"])

                conf = float(msg.get("confidence", 1.0))
            except Exception as e:
                print(f"[VISION] bad msg: {e}")
                continue

            # ---- pose check ----
            pose = await motion_planner._get_current_pose()
            if pose is None:
                print("[VISION] skip: pose None (filter down)")
                continue
            
            yaw = pose.a_from_b.rotation.log()[-1]
            xr, yr = pose.a_from_b.translation[0], pose.a_from_b.translation[1]
            c, s = math.cos(yaw), math.sin(yaw)

            # robot frame (x = forward, y = left)  -> world/map (X, Y)
            
            X_obj = xr + x * c - y * s
            Y_obj = yr + x * s + y * c
            print(f"[VISION] WP Location: Y={X_obj:.2f} m, X={-Y_obj:.2f} m  | robot @ ({xr:.2f}, {-yr:.2f}), ψ={math.degrees(yaw):.2f} deg")
                    
            # ---- distance from detected cone to ORIGINAL CSV waypoint (search zone center) ----
            def cone_distance_from_csv_waypoint():
                """
                Calculate distance from detected cone to the ORIGINAL CSV waypoint.
                This is used to validate if the cone is within the search zone.
                The search zone is centered at the current target waypoint.
                """
                # Get the original CSV waypoint for the current target waypoint
                # Since _create_ab_segment_to_next_waypoint() increments BEFORE building,
                # current_waypoint_index already points to the waypoint we're navigating toward
                csv_waypoint = None
                idx = max(1, motion_planner.current_waypoint_index)

                if hasattr(motion_planner, "original_csv_waypoints"):
                    csv_waypoint = motion_planner.original_csv_waypoints.get(idx)

                # Fallback to regular waypoints if original_csv_waypoints not available
                if csv_waypoint is None and hasattr(motion_planner, "waypoints"):
                    if hasattr(motion_planner.waypoints, "get"):
                        csv_waypoint = motion_planner.waypoints.get(idx)
                    elif isinstance(motion_planner.waypoints, (list, tuple)) and idx < len(motion_planner.waypoints):
                        csv_waypoint = motion_planner.waypoints[idx]

                # Calculate distance from detected cone to CSV waypoint
                if csv_waypoint is not None:
                    csv_x = float(csv_waypoint.a_from_b.translation[0])
                    csv_y = float(csv_waypoint.a_from_b.translation[1])
                    distance = math.hypot(X_obj - csv_x, Y_obj - csv_y)
                    return distance, idx
                return None, None

            dist_goal_wp, target_wp_idx = cone_distance_from_csv_waypoint()

            # ---- cone validation gates ----
            r = math.hypot(x, y)

            # Debug output
            if dist_goal_wp is not None:
                # Get waypoint coordinates for display (target_wp_idx is the NEXT waypoint)
                csv_waypoint = motion_planner.original_csv_waypoints.get(target_wp_idx)
                if csv_waypoint is not None:
                    wp_x = float(csv_waypoint.a_from_b.translation[0])
                    wp_y = float(csv_waypoint.a_from_b.translation[1])
                    print(f"[VISION] Cone @ ({X_obj:.2f}, {Y_obj:.2f}) is {dist_goal_wp:.2f}m from CSV waypoint {target_wp_idx} @ ({wp_x:.2f}, {wp_y:.2f}) (search radius: {VISION_SEARCH_RADIUS_M:.2f}m)")
                else:
                    print(f"[VISION] Cone @ ({X_obj:.2f}, {Y_obj:.2f}) is {dist_goal_wp:.2f}m from CSV waypoint {target_wp_idx} (search radius: {VISION_SEARCH_RADIUS_M:.2f}m)")
            else:
                print(f"[VISION] Warning: Could not determine CSV waypoint location (current_idx={motion_planner.current_waypoint_index}, target_idx={max(1, motion_planner.current_waypoint_index)})")
            print(f"[VISION] Cone rf: x={x:.2f} y={y:.2f} r={r:.2f} conf={conf:.2f}")

            # NEW BEHAVIOR: Cone must be within VISION_SEARCH_RADIUS_M of the CSV waypoint (search zone)
            IN_SEARCH_ZONE = (dist_goal_wp is not None and dist_goal_wp <= VISION_SEARCH_RADIUS_M)

            # Cone quality checks (robot frame): reasonable distance, not too far left/right, good confidence
            CONE_OK = (0.3 <= r <= 6.0 and abs(y) <= 3.0 and x >= 0.3 and conf >= 0.5)

            # Combined gate: cone must be in search zone AND pass quality checks
            if not (IN_SEARCH_ZONE and CONE_OK):
                if not IN_SEARCH_ZONE:
                    if dist_goal_wp is not None:
                        print(f"[VISION] skip: cone outside search zone (dist={dist_goal_wp:.2f}m > {VISION_SEARCH_RADIUS_M:.2f}m)")
                    else:
                        print(f"[VISION] skip: no CSV waypoint found for search zone validation")
                elif not CONE_OK:
                    print(f"[VISION] skip: cone quality check failed (r={r:.2f}, y={y:.2f}, conf={conf:.2f})")
                continue

            # ---- Check if actuator is deploying ----
            if getattr(nav_manager, "actuator_deploying", False):
                # print("[VISION] skip: actuator is deploying")
                continue

            # ---- Check if we've already processed this waypoint ----
            # target_wp_idx was already calculated by cone_distance_from_csv_waypoint() above
            # and represents the CSV waypoint we're checking against
            if target_wp_idx in nav_manager.vision_completed_waypoints:
                # print(f"[VISION] skip: waypoint {target_wp_idx} already processed by vision")
                continue

            now = asyncio.get_event_loop().time()

            # ---- NEW: Add detection to buffer for pre-approach planning ----
            # Buffer detections from planning range (2-6m) for use BEFORE approach segment creation
            PLANNING_RANGE_MIN = 2.0  # Minimum distance for planning detections
            PLANNING_RANGE_MAX = 6.0  # Maximum distance for planning detections

            if PLANNING_RANGE_MIN <= r <= PLANNING_RANGE_MAX:
                # Add to buffer
                if target_wp_idx not in motion_planner.vision_detection_buffer:
                    motion_planner.vision_detection_buffer[target_wp_idx] = []

                # Clean old detections from buffer (older than BUFFER_RETENTION_S)
                motion_planner.vision_detection_buffer[target_wp_idx] = [
                    det for det in motion_planner.vision_detection_buffer[target_wp_idx]
                    if now - det[0] < BUFFER_RETENTION_S
                ]

                # Add new detection: (timestamp, x_world, y_world, confidence, distance, robot_dist)
                motion_planner.vision_detection_buffer[target_wp_idx].append(
                    (now, X_obj, Y_obj, conf, dist_goal_wp, r)
                )

                print(f"[VISION BUFFER] Added detection for WP {target_wp_idx}: "
                      f"({X_obj:.2f}, {Y_obj:.2f}), dist_from_csv={dist_goal_wp:.2f}m, "
                      f"robot_dist={r:.2f}m, buffer_size={len(motion_planner.vision_detection_buffer[target_wp_idx])}")

            # ---- Check if robot is waiting at approach position for collar detection (REFINEMENT) ----
            # Only allow waypoint overrides when robot has stopped at approach waypoint (1.5m before collar)
            # This is the REFINEMENT step for final accuracy
            if not getattr(nav_manager, "waiting_for_collar_detection", False):
                # Not at approach position - just buffer the detection and continue
                continue

            # Robot is at approach position - perform refinement override
            # ---- debounce for refinement ----
            if last_goal is not None:
                moved = math.hypot(x - last_goal[0], y - last_goal[1])
                if moved < MIN_DIST_DELTA and (now - last_sent_t) < MIN_PERIOD_S:
                    # print(f"[VISION] skip: debounce (moved={moved:.2f}, dt={now-last_sent_t:.2f})")
                    continue

            # --- Override waypoint position with refined cone location ---
            try:
                # Override the next waypoint position with the detected cone position
                modified_idx = await motion_planner.override_next_waypoint_world_xy(X_obj, Y_obj, yaw_rad=None)
                # print(f"[VISION] Overrode waypoint {modified_idx} position to cone at ({X_obj:.2f}, {Y_obj:.2f})")

                # Mark that we detected a valid cone for the current waypoint
                nav_manager.cone_detected_for_current_wp = True
                nav_manager.vision_completed_waypoints.add(target_wp_idx)

                # Log cone detection for visualization
                cone_detection_record = {
                    "x": X_obj,
                    "y": Y_obj,
                    "waypoint_index": target_wp_idx,
                    "confidence": conf,
                    "robot_x": xr,
                    "robot_y": yr,
                    "robot_heading": yaw
                }
                nav_manager.cone_detections.append(cone_detection_record)
                logger.info(f"[VISION REFINEMENT] Collar detected at ({X_obj:.2f}, {Y_obj:.2f}) for waypoint {target_wp_idx}, refining position")

                # Update last goal for debouncing
                last_goal = (x, y)
                last_sent_t = now

            except Exception as e:
                print(f"[VISION] Failed to override waypoint: {e}")
                continue

    except asyncio.CancelledError:
        pass
    finally:
        try: sock.close()
        except Exception: pass

async def main(args) -> None:
    """Main function to orchestrate waypoint navigation."""
    nav_manager = None
    actuator: BaseActuator = NullActuator()
    
    service_config_list = proto_from_json_file(args.config, EventServiceConfigList())
    mc = multi(service_config_list)

    filter_client = mc.clients["filter"]
    controller_client = mc.clients["track_follower"]
    canbus_client = mc.clients.get("canbus")

    try:
        # Initialize motion planner
        logger.info("Initializing motion planner...")
        motion_planner = MotionPlanner(
            client=filter_client,
            tool_config_path=args.tool_config_path, # offset of centre of robot to dipper
            # camera_offset_path=args.camera_offset_path,
            waypoints_path=args.waypoints_path,
            last_row_waypoint_index=args.last_row_waypoint_index,
            turn_direction=args.turn_direction,
            row_spacing=args.row_spacing,
            headland_buffer=args.headland_buffer,
            return_to_start=bool(args.return_to_start),  # Convert 0/1 to False/True
        )

        actuator: BaseActuator = (
            CanHBridgeActuator(client=canbus_client,
                               actuator_id=args.actuator_id)
            if args.actuator_enabled and canbus_client is not None else
            NullActuator()
        )

        # Visual waypoint follower
        # asyncio.create_task(vision_goal_listener(motion_planner, controller_client, nav_manager))


        # Create nav_manager and inject actuator
        nav_manager = NavigationManager(
            filter_client=filter_client,
            controller_client=controller_client,
            motion_planner=motion_planner,
            no_stop=args.no_stop,
            canbus_client=canbus_client,
            actuator=actuator,
        )

        asyncio.create_task(
        vision_goal_listener(motion_planner, controller_client, nav_manager, proximity_m=2.0)
        )

        setup_signal_handlers(nav_manager=nav_manager)
        nav_manager.main_task = asyncio.current_task()

        # Check if filter has converged, and wiggle if needed
        logger.info("Checking filter convergence before starting navigation...")
        if canbus_client is not None:
            converged = await imu_wiggle(
                canbus_client=canbus_client,
                filter_client=filter_client,
                duration_seconds=3.0,
                angular_velocity=0.3,
                check_convergence=True,
                max_attempts=3
            )

            if not converged:
                logger.warning("Filter did not converge after wiggle attempts. Navigation may fail.")
                logger.warning("Consider manually shaking the robot or restarting the filter service.")
            else:
                logger.info("Filter converged successfully, ready for navigation!")
        else:
            logger.warning("CAN bus client not available, skipping IMU wiggle")

        # Run navigation
        await nav_manager.run_navigation()

    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt in main")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")

    finally:
        # Save navigation progress to JSON file
        progress_path = Path("./visualization/navigation_progress.json")
        try:
            serializable_progress = {}
            for segment_name, track in (nav_manager.navigation_progress if nav_manager else {}).items():
                x: List[float] = []
                y: List[float] = []
                heading: List[float] = []

                track_waypoints = [Pose3F64.from_proto(
                    pose) for pose in track.waypoints]
                for pose in track_waypoints:
                    x.append(pose.a_from_b.translation[0])
                    y.append(pose.a_from_b.translation[1])
                    heading.append(pose.a_from_b.rotation.log()[-1])

                serializable_progress[segment_name] = {
                    "waypoints_count": len(track.waypoints),
                    "x": x,
                    "y": y,
                    "heading": heading,
                }

            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "w") as f:
                json.dump(serializable_progress, f, indent=2)
            logger.info(f"Navigation progress saved to {progress_path}")
        except Exception as e:
            logger.error(f"FAILED to save navigation progress: {e}")

        positions_path = Path("./visualization/robot_positions.json")
        try:
            positions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(positions_path, "w") as f:
                json.dump(
                    getattr(nav_manager, "robot_positions", []), f, indent=2)
            logger.info(f"Robot positions saved to {positions_path}")
        except Exception as e:
            logger.error(f"FAILED to save robot positions: {e}")

        # Save cone detections
        cone_detections_path = Path("./visualization/cone_detections.json")
        try:
            cone_detections_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cone_detections_path, "w") as f:
                json.dump(
                    getattr(nav_manager, "cone_detections", []), f, indent=2)
            logger.info(f"Cone detections saved to {cone_detections_path}")
        except Exception as e:
            logger.error(f"FAILED to save cone detections: {e}")

        if nav_manager and not nav_manager.shutdown_requested:
            await nav_manager._cleanup()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Waypoint navigation using MotionPlanner and track_follower service"
    )
    # Required
    parser.add_argument("--waypoints-path", type=Path, required=True,
                        help="Path to waypoints JSON or CSV file (Track format)")
    parser.add_argument("--tool-config-path", type=Path,
                        required=True, help="Path to tool configuration JSON file")
    parser.add_argument(
        "--actuator-enabled",
        action="store_true",
        help="If set, pulse H-bridge OPEN briefly when the robot reaches each waypoint.",
    )
    parser.add_argument("--actuator-id", type=int, default=0,
                        help="H-bridge actuator ID (default: 0)")
    parser.add_argument(
        "--actuator-rate-hz",
        type=float,
        default=10.0,
        help="Command publish rate to CAN bus while reversing (Hz).",
    )

    # MotionPlanner configuration
    parser.add_argument(
        "--last-row-waypoint-index",
        type=int,
        default=6,
        help="Index of the last waypoint in the current row (default: 6)",
    )
    parser.add_argument(
        "--turn-direction",
        choices=["left", "right"],
        default="left",
        help="Direction to turn at row ends (default: left)",
    )
    parser.add_argument(
        "--row-spacing",
        type=float,
        default=6.0,
        help="Spacing between rows in meters (default: 6.0)",
    )
    parser.add_argument(
        "--headland-buffer",
        type=float,
        default=2.0,
        help="Buffer distance for headland maneuvers in meters (default: 2.0)",
    )
    parser.add_argument("--no-stop", action="store_true",
                        help="Disable stopping at each waypoint"
    )
    parser.add_argument("--return-to-start",
                        type=int,
                        choices=[0, 1],
                        default=1,
                        help="After last hole, perform row-end sequence to drive back to start (0=disabled, 1=enabled, default: 1)"
    )
    parser.add_argument("--config",
                        type=Path,
                        required=True,
                        help="The system config."
    )
    args = parser.parse_args()
    
    # xfm = Transforms(args.camera_offset_path)
    # os.environ["CAMERA_OFFSET_CONFIG"] = str(args.camera_offset_path)
    
    # Run the main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("\nFinal keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        logger.info("Script terminated")
        sys.exit(0)
