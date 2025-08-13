# WIP

# utils/dipbob_trigger.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import asyncio
import logging

from farm_ng.core.event_client import EventClient

logger = logging.getLogger("DipbobTrigger")


@dataclass
class DipbobCanConfig:
    can_id: int                 # e.g. 0x321
    data_hex: str               # e.g. "DEADBEEF" or "01FF0000AA55AA55"
    is_extended_id: bool = False
    repeat: int = 1             # how many times to send (some nodes like 2–3x)
    period_s: float = 0.02      # spacing between repeats


class DipbobCanTrigger:
    """Sends the CAN frame Dipbob is listening for."""

    def __init__(self, canbus_client: EventClient, cfg: DipbobCanConfig):
        self.client = canbus_client
        self.cfg = cfg

    async def __call__(self, ctx: Dict[str, Any]) -> None:
        if self.client is None:
            logger.warning(
                "Dipbob trigger skipped: no CAN bus client available.")
            return

        # Convert hex string to bytes
        payload = bytes.fromhex(self.cfg.data_hex)

        # Build & publish your CAN message using your project’s CAN publish helper.
        # Replace this with your actual CAN protobuf/topic API you use elsewhere.
        for i in range(self.cfg.repeat):
            try:
                # Example: publish on a topic your canbus service listens on
                # await self.client.request("canbus/send", CanFrame(id=..., data=..., extended=...))
                await self._publish_can_frame(self.cfg.can_id, payload, self.cfg.is_extended_id)
            except Exception as e:
                logger.error(f"Dipbob CAN send failed: {e}")
                break
            if i + 1 < self.cfg.repeat:
                await asyncio.sleep(self.cfg.period_s)

    async def _publish_can_frame(self, can_id: int, data: bytes, extended: bool) -> None:
        # TODO: adapt to your existing canbus helper. If you already have utils.canbus.move_robot_forward,
        # mirror that style here. For example:
        #
        # from utils.canbus import send_can_frame
        # await send_can_frame(self.client, can_id=can_id, data=data, extended=extended)
        #
        # For now, raise if not implemented:
        raise NotImplementedError(
            "Wire this to your existing CAN publish API.")
