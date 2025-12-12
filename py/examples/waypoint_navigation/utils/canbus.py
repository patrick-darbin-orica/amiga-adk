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
import logging
from pathlib import Path

from farm_ng.canbus.canbus_pb2 import RawCanbusMessage, Twist2d
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.filter.filter_pb2 import FilterState
from google.protobuf.empty_pb2 import Empty

logger = logging.getLogger(__name__)


CAN_EFF_FLAG = 0x80000000       # SocketCAN "extended frame" flag
CAN_EFF_MASK = 0x1FFFFFFF       # 29-bit ID mask

def eff_id(arb29: int) -> int:
    if arb29 & ~CAN_EFF_MASK:
        raise ValueError(f"ID {arb29:#x} exceeds 29 bits")
    return CAN_EFF_FLAG | arb29

async def _send_sig(client, arb29: int, payload: bytes, timeout: float = 5.0) -> None:
    msg = RawCanbusMessage()
    msg.id = eff_id(arb29)               # <-- set extended flag here
    msg.remote_transmission = False      # RTR=0 (data frame)
    msg.error = False
    msg.data = payload                   # 8 bytes
    await asyncio.wait_for(client.request_reply("/can_message", msg, decode=True), timeout=timeout)

async def trigger_dipbob(service_config_path: str = "can0", timeout: float = 5.0) -> None:
    """Trigger the dipbob deployment.

    Args:
        service_config_path: Path to canbus config or "can0" to use default
        timeout: Timeout in seconds for CAN message (default 5.0)

    Raises:
        asyncio.TimeoutError: If the CAN message times out (device may be unplugged)
    """
    logger = logging.getLogger(__name__)

    # If the argument isn't a JSON path, fall back to your default config
    cfg_path = Path(service_config_path)
    if cfg_path.suffix.lower() != ".json":
        cfg_path = Path("./configs/canbus_config.json")  # <— your real canbus service config

    cfg: EventServiceConfig = proto_from_json_file(cfg_path, EventServiceConfig())
    client = EventClient(cfg)

    try:
        logger.info(f"[DIPBOB] Sending trigger signal (timeout: {timeout}s)...")
        await _send_sig(client, 0x18FF0007, b"\x06\x00\x02\x00\x00\x00\x00\x00", timeout=timeout)
        await asyncio.sleep(0.02)
        logger.info("[DIPBOB] Dipbob trigger signal sent successfully")
        # Whie pressing button A sends both sigs, this sometimes triggers two drops of the dipper.
        # await _send_sig(client, 0x18FF0007, b"\x07\x00\x02\x00\x00\x00\x00\x00")
    except asyncio.TimeoutError:
        logger.error(f"[DIPBOB] Timeout after {timeout}s - dipbob device may be unplugged or unresponsive")
        raise
    except Exception as e:
        logger.error(f"[DIPBOB] Error sending trigger signal: {e}")
        raise


async def check_filter_convergence(filter_client: EventClient, timeout: float = 5.0) -> bool:
    """Check if the filter has converged.

    Args:
        filter_client: EventClient for the filter service
        timeout: Timeout for getting filter state

    Returns:
        True if filter has converged, False otherwise
    """
    try:
        state: FilterState = await asyncio.wait_for(
            filter_client.request_reply("/get_state", Empty(), decode=True),
            timeout=timeout
        )
        converged = bool(getattr(state, "has_converged", False))
        return converged
    except asyncio.TimeoutError:
        logger.warning("Timeout checking filter convergence")
        return False
    except Exception as e:
        logger.error(f"Error checking filter convergence: {e}")
        return False


async def imu_wiggle(
    canbus_client: EventClient,
    filter_client: EventClient | None = None,
    duration_seconds: float = 1.0,
    angular_velocity: float = 0.3,
    check_convergence: bool = True,
    max_attempts: int = 3
) -> bool:
    """Wiggle the robot back and forth to help the UKF filter converge.

    The filter fuses wheel odometry, GPS, and IMU data. When diverged (typically at startup),
    GPS-based waypoint navigation doesn't work, but basic twist commands do. This function
    mimics manually shaking the robot by sending alternating left/right angular velocity
    commands to excite the IMU and help the filter converge.

    Args:
        canbus_client: EventClient for the canbus service
        filter_client: Optional EventClient for checking filter convergence
        duration_seconds: How long to wiggle for each attempt (default 3.0s)
        angular_velocity: Angular velocity for wiggling (rad/s, default 0.3)
        check_convergence: Whether to check if filter converged after wiggling
        max_attempts: Maximum number of wiggle attempts before giving up

    Returns:
        True if filter converged (or if not checking), False if still diverged after max attempts
    """
    # logger.info("Starting IMU wiggle to help filter converge...")

    # Check initial convergence state
    initial_converged = False
    if filter_client and check_convergence:
        initial_converged = await check_filter_convergence(filter_client)
        if initial_converged:
            # logger.info("Filter already converged, no wiggle needed")
            return True

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        # logger.info(f"Wiggle attempt {attempt}/{max_attempts} - Duration: {duration_seconds}s, Angular vel: ±{angular_velocity} rad/s")

        # Wiggle pattern: left -> right -> left -> right
        wiggle_cycle_duration = duration_seconds / 4  # Quarter of total time per direction
        directions = [angular_velocity, -angular_velocity, angular_velocity, -angular_velocity]

        for ang_vel in directions:
            # Send twist command for this direction
            twist = Twist2d(linear_velocity_x=0.0, angular_velocity=ang_vel)

            start_time = asyncio.get_event_loop().time()
            end_time = start_time + wiggle_cycle_duration

            # Hold this direction for the cycle duration
            while asyncio.get_event_loop().time() < end_time:
                await canbus_client.request_reply("/twist", twist)
                await asyncio.sleep(0.05)  # Send at 20 Hz

        # Stop the robot
        stop_twist = Twist2d(linear_velocity_x=0.0, angular_velocity=0.0)
        await canbus_client.request_reply("/twist", stop_twist)
        # logger.info("Wiggle complete, robot stopped")

        # Wait a moment for filter to settle
        await asyncio.sleep(0.5)

        # Check if filter converged
        if filter_client and check_convergence:
            converged = await check_filter_convergence(filter_client)
            if converged:
                # logger.info(f"✓ Filter converged after {attempt} wiggle attempt(s)!")
                return True
            else:
                # logger.warning(f"Filter still diverged after attempt {attempt}/{max_attempts}")
                pass  # Continue to next attempt
        else:
            # If not checking convergence, assume success after wiggling
            # logger.info("Wiggle complete (convergence check disabled)")
            return True

    # Failed to converge after max attempts
    # logger.error(f"✗ Filter did not converge after {max_attempts} wiggle attempts")
    return False 
