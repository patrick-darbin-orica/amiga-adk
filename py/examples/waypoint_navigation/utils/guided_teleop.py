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
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty

logger = logging.getLogger("Guided Teleop")


async def get_current_pose(client: EventClient | None = None, timeout: float = 5.0) -> Optional[Pose3F64]:
    """Get the current pose for the track.

    Args:
        client: A EventClient for the required service (filter)
    Returns:
        The current pose (Pose3F64) if available, otherwise None.
    """

    if client is not None:
        try:
            # Get the current state of the filter
            state: FilterState = await asyncio.wait_for(
                client.request_reply("/get_state", Empty(), decode=True), timeout=timeout
            )
            return Pose3F64.from_proto(state.pose)
        except asyncio.TimeoutError:
            logger.info("Timeout while getting filter state. Using default start pose.")
        except Exception as e:
            logger.error(f"Error getting filter state: {e}. Using default start pose.")

    return None


async def process_movement(start_pose: Pose3F64, end_pose: Pose3F64) -> None:
    logger.info("Processing movement from start to end pose...")
    await asyncio.sleep(1.0)  # The robot wiggles a bit after we stop it, so we wait a bit before processing

    translation_array_start = np.asarray(start_pose.a_from_b.translation)
    # Extract x, y from numpy array
    x_s = float(translation_array_start[0])
    y_s = float(translation_array_start[1])
    # Extract heading from rotation (this should work as before)
    heading_s = float(start_pose.a_from_b.rotation.log()[-1])

    translation_array_final = np.asarray(end_pose.a_from_b.translation)
    # Extract x, y from numpy array
    x_f = float(translation_array_final[0])
    y_f = float(translation_array_final[1])
    # Extract heading from rotation (this should work as before)
    heading_f = float(end_pose.a_from_b.rotation.log()[-1])

    dx = x_f - x_s
    dy = y_f - y_s
    d_heading = heading_f - heading_s

    logger.info("Movement Summary: dx: {:.3f}, dy: {:.3f}, d_heading: {:.3f}".format(dx, dy, d_heading))


async def move_robot(direction: str = "forward") -> None:
    """Util function to move the robot forward in case it gets stuck.

    Args:
        direction: The direction to move the robot. Default is "forward".
    """

    linear_speed: float = 0.0
    angular_speed: float = 0.0

    if direction == "forward":
        linear_speed = 1.0
    elif direction == "backward":
        linear_speed = -1.0
    elif direction == "left":
        angular_speed = 1.0
    elif direction == "right":
        angular_speed = -1.0

    scale = 0.04  # Scale factor for the speed
    linear_speed *= scale
    angular_speed *= scale

    # Initialize the command to send
    time_goal = 20.0  # seconds to hold the command
    twist = Twist2d(linear_velocity_x=linear_speed, angular_velocity=angular_speed)

    # create a client to the canbus service
    canbus_config_path = Path("../configs/canbus_config.json")
    filter_config_path = Path("../configs/filter_config.json")
    canbus_config: EventServiceConfig = proto_from_json_file(canbus_config_path, EventServiceConfig())
    filter_config: EventServiceConfig = proto_from_json_file(filter_config_path, EventServiceConfig())
    canbus_client: EventClient = EventClient(canbus_config)
    filter_client: EventClient = EventClient(filter_config)

    # Get the start pose
    start_pose: Optional[Pose3F64] = None
    while start_pose is None:
        start_pose = await get_current_pose(filter_client)

    # Start moving the robot
    start = time.monotonic()

    # Hold the loop for the duration
    while time.monotonic() - start < time_goal:
        # Update and send the twist command
        logger.info(
            f"Sending linear velocity: {twist.linear_velocity_x:.3f}, angular velocity: {twist.angular_velocity:.3f}"
        )
        await canbus_client.request_reply("/twist", twist)

        # Sleep to maintain a constant rate
        await asyncio.sleep(0.1)

    # Bring the robot to a stop
    twist = Twist2d(linear_velocity_x=0.0, angular_velocity=0.0)
    logger.info("Stopping the robot")
    await canbus_client.request_reply("/twist", twist)

    # Get the final pose after moving
    final_pose: Optional[Pose3F64] = None
    while final_pose is None:
        final_pose = await get_current_pose(filter_client)

    await process_movement(start_pose, final_pose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python guided_teleop.py", description="Waypoint navigation using MotionPlanner and track_follower service"
    )

    parser.add_argument(
        "--turn-direction",
        choices=[
            "left",
            "right",
            "forward",
            "backward",
            "forward-left",
            "forward-right",
            "backward-left",
            "backward-right",
        ],
        default="forward",
        help="Direction to turn at (default: forward)",
    )

    args = parser.parse_args()

    # Run the main function
    try:
        asyncio.run(move_robot(args.turn_direction))
    except KeyboardInterrupt:
        logger.info("\n🛑 Final keyboard interrupt")
    except Exception as e:
        logger.error(f"💥 Unhandled exception: {e}")
    finally:
        logger.info("👋 Script terminated")
        sys.exit(0)
