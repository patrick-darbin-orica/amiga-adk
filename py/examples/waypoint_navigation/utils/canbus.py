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

import asyncio
import time
from pathlib import Path
from typing import Optional

from farm_ng.canbus.canbus_pb2 import Twist2d, RawCanbusMessage
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file


async def move_robot_forward(time_goal: float = 1.5) -> None:
    """Util function to move the robot forward in case it gets stuck.

    Args:
        service_config_path (Path): The path to the canbus service config.
    """
    # Initialize the command to send
    twist = Twist2d(linear_velocity_x=0.7)

    # create a client to the canbus service
    service_config_path = Path("./configs/canbus_config.json")
    config: EventServiceConfig = proto_from_json_file(
        service_config_path, EventServiceConfig())
    client: EventClient = EventClient(config)
    start = time.monotonic()
    # Hold the loop for the duration
    while time.monotonic() - start < time_goal:
        # Update and send the twist command
        print(
            f"Sending linear velocity: {twist.linear_velocity_x:.3f}, angular velocity: {twist.angular_velocity:.3f}")
        await client.request_reply("/twist", twist)

        # Sleep to maintain a constant rate
        await asyncio.sleep(0.1)


async def trigger_dipbob(
    *,
    can_id: int = 0x18FF0007,
    trigger_byte: int = 0x02,
    repeat: int = 1,            # bump to 2–3 if the RPi sometimes misses a frame
    period_s: float = 0.02,     # spacing between repeats
    raw_uri: str = "/raw",      # adjust if your service uses a different path
    service_config_path: Path = Path("./configs/canbus_config.json"),
) -> None:
    """Send the Dipbob trigger CAN frame (like move_robot_forward style)."""
    # Build the raw CAN message
    msg = RawCanbusMessage()
    # 0x18FF0007 (29-bit extended; service infers from value)
    msg.id = can_id
    msg.data = bytes([trigger_byte])   # one-byte payload: 0x02
    msg.error = False
    msg.remote_transmission = False

    # Create a client to the canbus service (same pattern as move_robot_forward)
    config: EventServiceConfig = proto_from_json_file(
        service_config_path, EventServiceConfig())
    client: EventClient = EventClient(config)

    # Send once (or a few times)
    for i in range(repeat):
        print(
            f"Sending Dipbob trigger: id=0x{can_id:08X}, data={msg.data.hex()}")
        await client.request_reply(raw_uri, msg)
        if i + 1 < repeat:
            await asyncio.sleep(period_s)
