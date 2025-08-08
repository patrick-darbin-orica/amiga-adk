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
from pathlib import Path
from typing import Optional
from typing import Tuple

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty

sys.path.append(str(Path(__file__).parent.parent))
from track_planner import TrackBuilder  # noqa: E402

logger = logging.getLogger("Investigate Turns")


async def get_current_pose(client: EventClient, timeout: float = 5.0) -> Optional[Pose3F64]:
    try:
        state: FilterState = await asyncio.wait_for(
            client.request_reply("/get_state", Empty(), decode=True), timeout=timeout
        )
        return Pose3F64.from_proto(state.pose)
    except asyncio.TimeoutError:
        logger.error("Timeout while getting filter state. Using default start pose.")
        return None  # ← Explicitly return None
    except Exception as e:
        logger.error(f"Error getting filter state: {e}.")
        return None


async def setup_clients() -> Tuple[EventClient, EventClient]:
    """Setup EventClients for filter and controller services.

    Args:
        filter_config_path: Path to filter service config
        controller_config_path: Path to controller service config

    Returns:
        Tuple of (filter_client, controller_client)
    """
    print("🔧 Setting up service clients...")

    filter_config_path = Path("../configs/filter_config.json")
    controller_config_path = Path("../configs/controller_config.json")

    # Load filter service config
    filter_config = proto_from_json_file(filter_config_path, EventServiceConfig())
    if filter_config.name != "filter":
        raise RuntimeError(f"Expected filter service config, got {filter_config.name}")
    filter_client = EventClient(filter_config)

    # Load controller service config
    controller_config = proto_from_json_file(controller_config_path, EventServiceConfig())
    if controller_config.name != "track_follower":
        raise RuntimeError(f"Expected track_follower service config, got {controller_config.name}")
    controller_client = EventClient(controller_config)

    logger.info(f"✅ Filter client: {filter_config.name}")
    logger.info(f"✅ Controller client: {controller_config.name}")

    return filter_client, controller_client


async def set_track(controller_client: EventClient, track: Track) -> None:
    """Set the track for the track_follower to follow.

    Args:
        track: The track to follow
    """
    logger.info(f"📤 Setting track with {len(track.waypoints)} waypoints...")
    try:
        await controller_client.request_reply("/set_track", TrackFollowRequest(track=track))
        logger.info("✅ Track set successfully")
    except Exception as e:
        logger.error(f"❌ Failed to set track: {e}")
        raise


async def start_following(controller_client: EventClient) -> None:
    """Start following the currently set track."""
    logger.info("🚀 Starting track following...")
    try:
        await controller_client.request_reply("/start", Empty())
        logger.info("✅ Track following started")
    except Exception as e:
        logger.error(f"❌ Failed to start track following: {e}")
        raise


async def main(args) -> None:
    """Main function to orchestrate waypoint navigation."""

    distance: float = args.distance

    try:
        # Setup clients
        filter_client, controller_client = await setup_clients()

        current_pose: Optional[Pose3F64] = None
        while current_pose is None:
            current_pose = await get_current_pose(filter_client)
            if current_pose is None:
                print("Retrying to get current pose...")
                await asyncio.sleep(1.0)

        # Create track
        track_builder = TrackBuilder(start=current_pose)
        track_builder.create_straight_segment(f"{distance}_m_straight_segment", distance=distance, spacing=0.10)
        track = track_builder.track
        await set_track(controller_client, track)
        await asyncio.sleep(1.0)  # Give some time for the track to be set
        await start_following(controller_client)

    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Waypoint navigation using MotionPlanner and track_follower service"
    )

    # Required argument
    parser.add_argument("--distance", type=float, default=10.0, help="Distance to travel (default 10 m)")

    args = parser.parse_args()

    # Run the main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
