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
# from utils.canbus import move_robot_forward
from utils.navigation_manager import NavigationManager
from utils.multiclient import MultiClientSubscriber as multi

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

async def vision_goal_listener(motion_planner, controller_client, nav_manager, proximity_m=10.0):
    """
    Listens for UDP 'cone_goal' messages and overrides the follower by:
      /cancel -> /set_track (short vision track) -> /start
    Now also /pause if auto mode is disabled (robot not controllable).
    """

    # --- helpers ------------------------------------------------------------
    async def get_state() -> TrackFollowerState:
        return await controller_client.request_reply("/get_state", Empty(), decode=True)

    def _is_terminal(st: TrackFollowerState) -> bool:
        return st.status.track_status in (
            TrackStatusEnum.TRACK_COMPLETE,
            TrackStatusEnum.TRACK_ABORTED,
            TrackStatusEnum.TRACK_FAILED,
            TrackStatusEnum.TRACK_TIMEOUT,
        )
        
    async def wait_until(pred, timeout: float, poll_s: float = 0.05):
        """
        Poll follower state until pred(state) is True, or timeout elapses.
        Assumes get_state() returns a decoded TrackFollowerState (decode=True).
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout

        # initial sample
        st = await get_state()  # TrackFollowerState
        while not pred(st):
            if loop.time() >= deadline:
                raise TimeoutError("wait condition not met")

            await asyncio.sleep(poll_s)

            try:
                st = await get_state()  # keep sampling
                if st is None:
                    # extremely defensive: treat as transient failure
                    continue
            except Exception as e:
                # transient RPC/wire hiccup; keep trying until deadline
                # (optionally log: print(f"[vision] get_state failed: {e}"))
                continue

        return st


    # One mutex to rule all controller RPCs (prevents races with nav loop)
    if not hasattr(nav_manager, "_controller_lock"):
        nav_manager._controller_lock = asyncio.Lock()
    ctl_lock: asyncio.Lock = nav_manager._controller_lock

    # Latch state on the nav_manager object so it’s visible across loops
    if not hasattr(nav_manager, "vision_latched"):
        nav_manager.vision_latched = False
    if not hasattr(nav_manager, "vision_latch_deadline"):
        nav_manager.vision_latch_deadline = 0.0

    LATCH_MAX_S = 20.0  # safety timeout for a vision retarget (tune as you like)

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

    print("Started vision goal listener")

    try:
        while True:
            # --- while latched, just watch state and drop messages ---
            if nav_manager.vision_latched:
                # auto-unlatch on terminal state or timeout
                try:
                    st = await get_state()
                    if _is_terminal(st) or (asyncio.get_event_loop().time() >= nav_manager.vision_latch_deadline):
                        nav_manager.vision_latched = False
                        # small grace to avoid immediate re-trigger on the same frame
                        await asyncio.sleep(0.2)
                except Exception:
                    # if state read hiccups, keep latch until deadline
                    pass

                # drain/ignore incoming UDP quickly, then continue loop
                try:
                    _ = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(None, sock.recvfrom, 4096), timeout=0.01
                    )
                except Exception:
                    await asyncio.sleep(0.03)
                continue

            # ---- receive next message (non-latched path) ----
            try:
                data, _ = await loop.run_in_executor(None, sock.recvfrom, 4096)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[vision] recv error: {e}")
                continue

            # ---- parse ----
            try:
                msg = json.loads(data.decode())
                if msg.get("type") != "cone_goal":
                    continue
                x = float(msg["x_fwd_m"])
                y = float(msg["y_left_m"])
                conf = float(msg.get("confidence", 1.0))
            except Exception as e:
                print(f"[vision] bad msg: {e}")
                continue

            # ---- pose check ----
            pose = await motion_planner._get_current_pose()
            if pose is None:
                print("[vision] skip: pose None (filter down)")
                continue

            # ---- distance to next nominal waypoint (if available) ----
            def distance_to_nominal_or_none():
                idx = motion_planner.current_waypoint_index + 1
                next_goal = motion_planner.waypoints.get(idx) if hasattr(motion_planner.waypoints, "get") else (
                    motion_planner.waypoints[idx] if 0 <= idx < len(motion_planner.waypoints) else None
                )
                if next_goal is not None:
                    dx, dy = (next_goal.a_from_b.translation - pose.a_from_b.translation)[:2]
                    return math.hypot(dx, dy)
                return None

            d_nom = distance_to_nominal_or_none()
            if d_nom is not None:
                print(f"[vision] proximity: {d_nom:.2f} m (threshold {proximity_m:.2f})")

            # ---- cone gates (robot frame) ----
            r = math.hypot(x, y)
            print(f"[vision] cone rf: x={x:.2f} y={y:.2f} r={r:.2f} conf={conf:.2f}")

            PROX_OK   = (d_nom is not None and d_nom <= proximity_m)  # near nominal path
            CONE_NEAR = (0.3 <= r <= 2.0)                              # very close
            CONE_OK   = (0.3 <= r <= 6.0 and abs(y) <= 3.0 and x >= 0.3 and conf >= 0.5)

            # follower status (for FOLLOWING_OK gate)
            try:
                st = await get_state()
                is_following = (st.status.track_status == TrackStatusEnum.TRACK_FOLLOWING)
            except Exception:
                is_following = False
            FOLLOWING_OK = (is_following and CONE_OK)

            if not (PROX_OK or CONE_NEAR or FOLLOWING_OK):
                print("[vision] skip: gate (need near nominal OR cone near OR following+good cone)")
                continue

            # ---- debounce ----
            now = asyncio.get_event_loop().time()
            if last_goal is not None:
                moved = math.hypot(x - last_goal[0], y - last_goal[1])
                if moved < MIN_DIST_DELTA and (now - last_sent_t) < MIN_PERIOD_S:
                    print(f"[vision] skip: debounce (moved={moved:.2f}, dt={now-last_sent_t:.2f})")
                    continue

            # ---- don't stack overrides ----
            if getattr(nav_manager, "vision_active", False):
                print("[vision] skip: vision_active already true")
                continue

            # ---- Build the short track (world standoff) ----
            try:
                track, goal = await motion_planner.build_track_to_robot_relative_goal(
                    x, y, standoff_m=0.75, spacing=0.1
                )
            except Exception as e:
                print(f"[vision] build track failed: {e}")
                continue

            print("[vision] OK → cancel + set_track + start")

            # ---- Run the override safely ----
            if hasattr(nav_manager, "vision_active"):
                nav_manager.vision_active = True

            try:
                async with ctl_lock:
                    st = await get_state()

                    # Only pause if not controllable AND currently following
                    if not st.status.robot_status.controllable:
                        if st.status.track_status == TrackStatusEnum.TRACK_FOLLOWING:
                            try:
                                await controller_client.request_reply("/pause", Empty())
                                print("[vision] paused follower (auto mode disabled / not controllable)")
                            except Exception as e:
                                print(f"[vision] pause failed: {e}")
                        # Skip this cycle entirely; don't set a new track while not controllable
                        continue

                    # cancel current (ignore errors if idle)
                    try:
                        await controller_client.request_reply("/cancel", Empty())
                    except Exception:
                        pass

                    # wait until NOT FOLLOWING to avoid “already following” error on /start
                    try:
                        await wait_until(lambda s: s.status.track_status != TrackStatusEnum.TRACK_FOLLOWING, timeout=3.0)
                    except TimeoutError:
                        print("[vision] warn: still FOLLOWING after cancel; proceeding anyway")

                    # set → wait LOADED → start → wait FOLLOWING
                    req = TrackFollowRequest(track=track)
                    await controller_client.request_reply("/set_track", req)
                    await wait_until(lambda s: s.status.track_status == TrackStatusEnum.TRACK_LOADED, timeout=2.0)
                    await controller_client.request_reply("/start", Empty())
                    await wait_until(lambda s: s.status.track_status == TrackStatusEnum.TRACK_FOLLOWING, timeout=2.0)

                    print("[vision] start sent; goal world = (%.2f, %.2f)" %
                          (goal.a_from_b.translation[0], goal.a_from_b.translation[1]))

                    # LATCH NOW: ignore further cone messages until terminal or timeout
                    nav_manager.vision_latched = True
                    nav_manager.vision_latch_deadline = asyncio.get_event_loop().time() + LATCH_MAX_S

                # optional: light polling (purely for console feedback)
                t0 = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - t0 < 4.0:
                    try:
                        st = await get_state()
                        print(f"[vision] follower status: {TrackStatusEnum.Name(st.status.track_status)}")
                        if _is_terminal(st):
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.25)

            finally:
                if hasattr(nav_manager, "vision_active"):
                    nav_manager.vision_active = False
                await asyncio.sleep(0.1)

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
        vision_goal_listener(motion_planner, controller_client, nav_manager, proximity_m=10.0)
        )
        
        setup_signal_handlers(nav_manager=nav_manager)
        nav_manager.main_task = asyncio.current_task()

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
        "--actuator-open-seconds",
        type=float,
        default=6,
        help="Duration to drive actuator in OPEN after reaching a waypoint (seconds).",
    )
    parser.add_argument(
        "--actuator-close-seconds",
        type=float,
        default=6,
        help="Duration to drive actuator in CLOSE after reaching a waypoint (seconds).",
    )
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
