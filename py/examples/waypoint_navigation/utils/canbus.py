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
from pathlib import Path
import argparse

from farm_ng.canbus.canbus_pb2 import RawCanbusMessage
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file


CAN_EFF_FLAG = 0x80000000       # SocketCAN "extended frame" flag
CAN_EFF_MASK = 0x1FFFFFFF       # 29-bit ID mask

def eff_id(arb29: int) -> int:
    if arb29 & ~CAN_EFF_MASK:
        raise ValueError(f"ID {arb29:#x} exceeds 29 bits")
    return CAN_EFF_FLAG | arb29

async def _send_once_farmng(client, arb29: int, payload: bytes) -> None:
    msg = RawCanbusMessage()
    msg.id = eff_id(arb29)               # <-- set extended flag here
    msg.remote_transmission = False      # RTR=0 (data frame)
    msg.error = False
    msg.data = payload                   # 8 bytes
    await client.request_reply("/can_message", msg, decode=True)

async def trigger_dipbob(service_config_path: str = "can0") -> None:
    # If the argument isn't a JSON path, fall back to your default config
    cfg_path = Path(service_config_path)
    if cfg_path.suffix.lower() != ".json":
        cfg_path = Path("./configs/canbus_config.json")  # <— your real canbus service config

    cfg: EventServiceConfig = proto_from_json_file(cfg_path, EventServiceConfig())
    client = EventClient(cfg)

    await _send_once_farmng(client, 0x18FF0007, b"\x06\x00\x02\x00\x00\x00\x00\x00")
    await asyncio.sleep(0.02)
    await _send_once_farmng(client, 0x18FF0007, b"\x07\x00\x02\x00\x00\x00\x00\x00")
