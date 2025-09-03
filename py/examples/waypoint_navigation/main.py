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
import time
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
from utils.canbus import move_robot_forward
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
            tool_config_path=args.tool_config_path,
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

        # Create nav_manager and inject actuator
        nav_manager = NavigationManager(
            filter_client=filter_client,
            controller_client=controller_client,
            motion_planner=motion_planner,
            no_stop=args.no_stop,
            canbus_client=canbus_client,
            actuator=actuator,
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
