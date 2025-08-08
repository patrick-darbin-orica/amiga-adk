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
import signal
import sys
from math import radians
from pathlib import Path
from typing import Optional
from typing import Tuple

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng.track.track_pb2 import RobotStatus
from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowerState
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng.track.track_pb2 import TrackStatusEnum
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty
from track_planner import TrackBuilder


async def get_current_pose(client: EventClient, timeout: float = 5.0) -> Optional[Pose3F64]:
    try:
        state: FilterState = await asyncio.wait_for(
            client.request_reply("/get_state", Empty(), decode=True), timeout=timeout
        )
        return Pose3F64.from_proto(state.pose)
    except asyncio.TimeoutError:
        print("Timeout while getting filter state. Using default start pose.")
        return None  # ← Explicitly return None
    except Exception as e:
        print(f"Error getting filter state: {e}. Using default start pose.")
        return None  # ← Explicitly return None


class NavigationManager:
    """Orchestrates waypoint navigation using MotionPlanner and track_follower service."""

    def __init__(self, filter_client: EventClient, controller_client: EventClient, number_of_turns: int, delay: float):
        self.filter_client = filter_client
        self.controller_client = controller_client
        self.current_track_status: Optional[TrackStatusEnum] = None
        self.track_complete_event = asyncio.Event()
        self.track_failed_event = asyncio.Event()
        self.shutdown_requested = False
        self.delay = delay
        self.number_of_turns = number_of_turns

    async def get_current_pose(self) -> Pose3F64:
        current_pose: Pose3F64 | None = None
        while current_pose is None:
            try:
                current_pose = await get_current_pose(self.filter_client)
            except Exception as e:
                print(f"Error getting current pose: {e}")
                await asyncio.sleep(1.0)
        return current_pose

    async def set_track(self, track: Track) -> None:
        """Set the track for the track_follower to follow.

        Args:
            track: The track to follow
        """
        print(f"📤 Setting track with {len(track.waypoints)} waypoints...")
        try:
            await self.controller_client.request_reply("/set_track", TrackFollowRequest(track=track))
            print("✅ Track set successfully")
        except Exception as e:
            print(f"❌ Failed to set track: {e}")
            raise

    async def start_following(self) -> None:
        """Start following the currently set track."""
        print("🚀 Starting track following...")
        try:
            await self.controller_client.request_reply("/start", Empty())
            print("✅ Track following started")
        except Exception as e:
            print(f"❌ Failed to start track following: {e}")
            raise

    async def monitor_track_state(self) -> None:
        """Monitor the track_follower state and set events based on status."""
        print("👁️  Starting track state monitoring...")

        try:
            config = self.controller_client.config
            subscription = config.subscriptions[0] if config.subscriptions else "/state"

            async for event, message in self.controller_client.subscribe(subscription, decode=True):
                if self.shutdown_requested:
                    break

                if isinstance(message, TrackFollowerState):
                    await self._process_track_state(message)

        except Exception as e:
            print(f"❌ Error monitoring track state: {e}")
            self.track_failed_event.set()

    async def _process_track_state(self, state: TrackFollowerState) -> None:
        """Process incoming track follower state messages.

        Args:
            state: The TrackFollowerState message
        """
        track_status = state.status.track_status
        robot_controllable = state.status.robot_status.controllable

        # Update current status
        prev_status = self.current_track_status
        self.current_track_status = track_status

        # Log status changes
        if prev_status != track_status:
            try:
                status_name = TrackStatusEnum.Name(track_status)
                print(f"📊 Track status changed: {status_name}")
            except Exception as e:
                print(f"❌ Error getting status name: {e}")

        # Check for completion or failure
        if track_status == TrackStatusEnum.TRACK_COMPLETE:
            print("🎉 Track completed successfully!")
            self.track_complete_event.set()

        elif track_status in [
            TrackStatusEnum.TRACK_FAILED,
            TrackStatusEnum.TRACK_ABORTED,
            TrackStatusEnum.TRACK_CANCELLED,
        ]:
            try:
                status_name = TrackStatusEnum.Name(track_status)
                print(f"💥 Track failed with status: {status_name}")
            except Exception as e:
                print(f"❌ Error getting status name: {e}")

            if not robot_controllable:
                try:
                    failure_modes = []
                    for mode in state.status.robot_status.failure_modes:
                        try:
                            mode_name = RobotStatus.FailureMode.Name(mode)
                            failure_modes.append(mode_name)
                        except Exception as e:
                            # Fallback to the integer value if enum name lookup fails
                            failure_modes.append(f"UNKNOWN({mode})")
                            print(f"❌ Error getting failure mode name: {e}")

                    print(f"Robot not controllable. Failure modes: {failure_modes}")
                except Exception as e:
                    print(f"Robot not controllable. Failed to get failure modes: {e}")
            self.track_failed_event.set()

        # Log cross-track error if available
        if (
            hasattr(state, 'progress')
            and state.progress
            and hasattr(state.progress, 'cross_track_error')
            and state.progress.cross_track_error
        ):
            error = state.progress.cross_track_error
            if error.total_distance > 0.5:  # Only log if significant error
                print(
                    f"⚠️  Cross-track error: {error.total_distance:.2f}m "
                    f"(lateral: {error.lateral_distance:.2f}m, "
                    f"longitudinal: {error.longitudinal_distance:.2f}m)"
                )

    async def wait_for_track_completion(self, timeout: float = 60.0) -> bool:
        """Wait for track to complete or fail.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if track completed successfully, False if failed or timed out
        """
        print(f"⏳ Waiting for track completion (timeout: {timeout}s)...")

        try:
            # Wait for either completion or failure
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self.track_complete_event.wait()),
                    asyncio.create_task(self.track_failed_event.wait()),
                ],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            if not done:
                print("⏰ Timeout waiting for track completion")
                return False

            # Check which event was set
            if self.track_complete_event.is_set():
                return True
            elif self.track_failed_event.is_set():
                return False

        except Exception as e:
            print(f"❌ Error waiting for track completion: {e}")
            return False

        return False

    async def execute_single_track(self, track: Track, timeout: float = 30.0) -> bool:
        """Execute a single track segment and wait for completion.

        Args:
            track: The track to execute
            timeout: Maximum time to wait for completion

        Returns:
            True if successful, False otherwise
        """
        # Reset events
        self.track_complete_event.clear()
        self.track_failed_event.clear()

        try:
            # Set and start the track
            await self.set_track(track)
            await asyncio.sleep(1.0)  # Brief pause to ensure track is set
            await self.start_following()

            # Wait for completion
            success = await self.wait_for_track_completion(timeout)

            if success:
                print("✅ Track segment completed successfully")
            else:
                print("❌ Track segment failed or timed out")

            return success

        except Exception as e:
            print(f"❌ Error executing track: {e}")
            return False

    async def run_navigation(self) -> None:
        """Run the complete waypoint navigation sequence."""
        print("🚁 Starting waypoint navigation...")

        # Start monitoring track state
        monitor_task = asyncio.create_task(self.monitor_track_state())

        try:
            segment_count = 0

            for _ in range(0, self.number_of_turns):
                if self.shutdown_requested:
                    break

                print(f"\n--- Turn {segment_count + 1} ---")

                try:
                    current_pose = await self.get_current_pose()
                    if current_pose is None:
                        print("❌ Failed to get current pose, skipping turn")
                        continue

                    track_builder = TrackBuilder(start=current_pose)
                    if track_builder is None:
                        print("❌ Failed to create track builder, skipping turn")
                        continue

                    track_builder.create_turn_segment(next_frame_b="90_deg_turn", angle=radians(90), spacing=0.1)
                    track_segment = track_builder.track
                    if track_segment is None:
                        print("❌ Failed to create track segment, skipping turn")
                        continue

                    print(f"📍 Executing turn {segment_count}")
                    success = await self.execute_single_track(track_segment)

                    while not success:
                        print(f"💥 Failed to execute segment {segment_count}. Retrying...")
                        success = await self.execute_single_track(track_segment)

                    segment_count += 1
                    await asyncio.sleep(self.delay)

                except Exception as e:
                    print(f"❌ Error in turn {segment_count + 1}: {e}")
                    import traceback

                    traceback.print_exc()
                    break

            print(f"🎯 Navigation completed after {segment_count} segments")

        except Exception as e:
            print(f"💥 Navigation failed with error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.shutdown_requested = True
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass


async def setup_clients(filter_config_path: Path, controller_config_path: Path) -> Tuple[EventClient, EventClient]:
    """Setup EventClients for filter and controller services.

    Args:
        filter_config_path: Path to filter service config
        controller_config_path: Path to controller service config

    Returns:
        Tuple of (filter_client, controller_client)
    """
    print("🔧 Setting up service clients...")

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

    print(f"✅ Filter client: {filter_config.name}")
    print(f"✅ Controller client: {controller_config.name}")

    return filter_client, controller_client


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        # Let the main loop handle the shutdown
        return

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main(args) -> None:
    """Main function to orchestrate waypoint navigation."""

    setup_signal_handlers()
    delay = args.delay
    number_of_turns = args.turns

    try:
        # Setup clients
        filter_client, controller_client = await setup_clients(args.filter_config, args.controller_config)

        # Create orchestrator
        orchestrator = NavigationManager(
            filter_client=filter_client,
            controller_client=controller_client,
            delay=delay,
            number_of_turns=number_of_turns,
        )

        # Run navigation
        await orchestrator.run_navigation()

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Waypoint navigation using MotionPlanner and track_follower service"
    )

    # Required arguments
    parser.add_argument("--filter-config", type=Path, required=True, help="Path to filter service config JSON file")
    parser.add_argument(
        "--controller-config", type=Path, required=True, help="Path to track_follower service config JSON file"
    )
    parser.add_argument("--turns", type=int, default=5, help="Number of turns")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between turns")

    args = parser.parse_args()

    # Validate file paths
    for path_arg in [args.filter_config, args.controller_config]:
        if not path_arg.exists():
            print(f"❌ File not found: {path_arg}")
            sys.exit(1)

    # Run the main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
