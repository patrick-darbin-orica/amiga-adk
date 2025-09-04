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
import shutil
from pathlib import Path
import can

from farm_ng.canbus.canbus_pb2 import Twist2d, RawCanbusMessage
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file


# async def move_robot_forward(time_goal: float = 1.5) -> None:
#     """Util function to move the robot forward in case it gets stuck.

#     Args:
#         service_config_path (Path): The path to the canbus service config.
#     """
#     # Initialize the command to send
#     twist = Twist2d(linear_velocity_x=0.7)

#     # create a client to the canbus service
#     service_config_path = Path("./configs/canbus_config.json")
#     config: EventServiceConfig = proto_from_json_file(
#         service_config_path, EventServiceConfig())
#     client: EventClient = EventClient(config)
#     start = time.monotonic()
#     # Hold the loop for the duration
#     while time.monotonic() - start < time_goal:
#         # Update and send the twist command
#         print(
#             f"Sending linear velocity: {twist.linear_velocity_x:.3f}, angular velocity: {twist.angular_velocity:.3f}")
#         await client.request_reply("/twist", twist)

#         # Sleep to maintain a constant rate
#         await asyncio.sleep(0.1)


def _send_once_socketcan(bus: can.Bus, data: bytes) -> None:
    msg = can.Message(
        arbitration_id=0x18FF0007,
        data=data,
        is_extended_id=True,  # 0x18FF0007 is 29-bit
    )
    bus.send(msg, timeout=0.1)

async def trigger_dipbob(iface: str = "can0") -> None:
    # open/close per call to keep it simple; you can hold the bus persistently elsewhere
    bus = can.interface.Bus(channel=iface, bustype="socketcan")
    try:
        _send_once_socketcan(bus, bytes([0x06, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00]))
        await asyncio.sleep(0.02)
        _send_once_socketcan(bus, bytes([0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00]))
    finally:
        bus.shutdown()