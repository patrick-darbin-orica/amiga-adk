# Replace your socketcan helpers with these farm-ng API versions.

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

async def trigger_dipbob(service_config_path: str | Path = "./configs/canbus_config.json") -> None:
    cfg: EventServiceConfig = proto_from_json_file(Path(service_config_path), EventServiceConfig())
    client = EventClient(cfg)
    await _send_once_farmng(client, 0x18FF0007, bytes([0x06,0x00,0x02,0x00,0x00,0x00,0x00,0x00]))
    await asyncio.sleep(0.02)
    await _send_once_farmng(client, 0x18FF0007, bytes([0x07,0x00,0x02,0x00,0x00,0x00,0x00,0x00]))
async def main(args):
    # just call the function with the config path provided
    await trigger_dipbob(args.config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test sending dipbob CAN frames via farm-ng API")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./canbus_config.json"),
        help="Path to canbus service config JSON"
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("Interrupted by user")