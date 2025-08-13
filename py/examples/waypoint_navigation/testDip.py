#!/usr/bin/env python3
import asyncio
from pathlib import Path

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.canbus.canbus_pb2 import RawCanbusMessage


async def trigger_dipbob(
    service_config_path: Path = Path("./configs/canbus_config.json"),
    raw_uri: str = "/raw",      # change if your service exposes a different route
    repeat_each: int = 1,       # send each frame N times (reliability)
    period_s: float = 0.02,     # spacing between repeats
) -> None:
    # Load CAN service config and create client
    cfg: EventServiceConfig = proto_from_json_file(
        service_config_path, EventServiceConfig())
    client = EventClient(cfg)

    # Two frames to send, in order (exact bytes you observed on bus)
    frames = [
        bytes([0x06, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00]),
        bytes([0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00]),
    ]
    # extended (29-bit) ID; most stacks infer extended from >0x7FF
    can_id = 0x18FF0007

    for data in frames:
        # ✅ construct the proto, don’t leave RawCanbusMessage(id=) half-filled
        msg = RawCanbusMessage()
        msg.id = can_id
        msg.data = data
        msg.error = False
        msg.remote_transmission = False

        for i in range(repeat_each):
            print(
                f"Sending 18FF0007 [{len(data)}] {data.hex(' ').upper()} ({i+1}/{repeat_each})")
            # Usually no typed reply for raw CAN -> don't decode
            await client.request_reply(raw_uri, msg, decode=False)
            if i + 1 < repeat_each:
                await asyncio.sleep(period_s)


if __name__ == "__main__":
    try:
        asyncio.run(trigger_dipbob())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user.")
