"""
Actuator abstraction for Amiga waypoint navigation.

- Provides a no-op actuator (NullActuator) when CAN is not available.
- Provides a CAN H-bridge actuator (CanHBridgeActuator) that drives an
  H-bridge via the farm-ng /control_tools endpoint.

Drop this file next to main.py, then modify NavigationManager to accept
an `actuator` object and replace direct calls to `_pulse_actuator_open`
/ `_pulse_actuator_close` with `await actuator.pulse_sequence(...)` or the
more granular `pulse_open`/`pulse_close` calls.

This module does not import anything from your app code, so it is reusable.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from farm_ng.core.event_client import EventClient
from farm_ng.canbus.tool_control_pb2 import (
    ActuatorCommands,
    HBridgeCommand,
    HBridgeCommandType,
)

logger = logging.getLogger(__name__)


# ------------------------------
# Utility
# ------------------------------

def _build_hbridge_cmd(hbridge_id: int, cmd: HBridgeCommandType.ValueType) -> ActuatorCommands:
    commands: ActuatorCommands = ActuatorCommands()
    commands.hbridges.append(HBridgeCommand(id=hbridge_id, command=cmd))
    return commands


# ------------------------------
# Public API
# ------------------------------

# TODO: Relabel functions as open/close
class BaseActuator:
    """Abstract actuator interface."""

    async def pulse_open(self, seconds: float, rate_hz: float = 10.0) -> None:
        raise NotImplementedError

    async def pulse_close(self, seconds: float, rate_hz: float = 10.0) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def _wait_for_enter(self, prompt: str = "Press ENTER to close the chute...",
                              timeout: float | None = None) -> None:
        """
        Non-blocking wait for a single ENTER keypress (runs input() in a thread).
        If no TTY or an error occurs, we fall back to a short sleep.
        """
        loop = asyncio.get_running_loop()

        def _input_blocking():
            try:
                return input(prompt + " ")
            except Exception:
                # e.g., no TTY / redirected stdin
                return None

        try:
            if timeout is None:
                await loop.run_in_executor(None, _input_blocking)
            else:
                await asyncio.wait_for(loop.run_in_executor(None, _input_blocking), timeout=timeout)
        except (asyncio.TimeoutError, Exception):
            # Timeout or no-stdin: don’t derail the pipeline; just continue.
            logger.warning("ENTER wait skipped (timeout or no interactive stdin)")

    async def pulse_sequence(
        self,
        open_seconds: float,
        close_seconds: float,
        rate_hz: float = 10.0,
        settle_before: float = 0.0,
        settle_between: float = 1.0,
        settle_after: float = 0.0,
        *,
        wait_for_enter_between: bool = False,
        enter_prompt: str = "Press ENTER to close the chute...",
        enter_timeout: float | None = None,
    ) -> None:
        """Convenience: [optional wait] → open → (wait|ENTER) → close → [optional wait]."""
        if settle_before > 0:
            await asyncio.sleep(settle_before)

        if open_seconds > 0:
            await self.pulse_open(open_seconds, rate_hz)

        # Replace the fixed sleep with an optional required ENTER keypress.
        if wait_for_enter_between:
            # Try non-blocking console wait; if unavailable, fall back to settle_between.
            await self._wait_for_enter(prompt=enter_prompt, timeout=enter_timeout)
        elif settle_between > 0:
            await asyncio.sleep(settle_between)

        if close_seconds > 0:
            await self.pulse_close(close_seconds, rate_hz)

        if settle_after > 0:
            await asyncio.sleep(settle_after)


class NullActuator(BaseActuator):
    """No-op actuator used when CAN is unavailable or disabled."""

    async def pulse_open(self, seconds: float, rate_hz: float = 10.0) -> None:  # noqa: ARG002
        logger.debug("NullActuator: pulse_open(%ss) ignored", seconds)

    async def pulse_close(self, seconds: float, rate_hz: float = 10.0) -> None:  # noqa: ARG002
        logger.debug("NullActuator: pulse_close(%ss) ignored", seconds)

    async def stop(self) -> None:
        logger.debug("NullActuator: stop() ignored")


@dataclass
class CanHBridgeActuator(BaseActuator):
    """Drives an H-bridge via farm-ng CAN /control_tools service."""

    client: Optional[EventClient]
    actuator_id: int = 0

    async def _drive_for(self, command: HBridgeCommandType.ValueType, seconds: float, rate_hz: float) -> None:
        if self.client is None:
            logger.warning(
                "CanHBridgeActuator: No CAN client; skipping command %s", command)
            return
        seconds = max(0.0, float(seconds))
        if seconds == 0.0:
            return
        rate_hz = max(0.1, float(rate_hz))
        period = 1.0 / rate_hz
        t_end = time.monotonic() + seconds
        name = HBridgeCommandType.Name(command) if hasattr(
            HBridgeCommandType, "Name") else str(command)
        logger.info("Actuator %d: %s for %.2fs @ %.1f Hz",
                    self.actuator_id, name, seconds, rate_hz)
        try:
            while time.monotonic() < t_end:
                await self.client.request_reply(
                    "/control_tools",
                    _build_hbridge_cmd(self.actuator_id, command),
                    decode=True,
                )
                await asyncio.sleep(period)
        finally:
            # Always attempt a STOP at the end of any drive
            await self.stop()

    async def pulse_open(self, seconds: float, rate_hz: float = 10.0) -> None:
        await self._drive_for(HBridgeCommandType.HBRIDGE_FORWARD, seconds, rate_hz)

    async def pulse_close(self, seconds: float, rate_hz: float = 10.0) -> None:
        await self._drive_for(HBridgeCommandType.HBRIDGE_REVERSE, seconds, rate_hz)

    async def stop(self) -> None:
        if self.client is None:
            return
        logger.info("Actuator %d: STOP", self.actuator_id)
        await self.client.request_reply(
            "/control_tools",
            _build_hbridge_cmd(
                self.actuator_id, HBridgeCommandType.HBRIDGE_STOPPED),
            decode=True,
        )
