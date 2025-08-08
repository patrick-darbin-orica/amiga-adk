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
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.track.track_pb2 import RobotStatus
from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowerState
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng.track.track_pb2 import TrackStatusEnum
from farm_ng_core_pybind import Pose3F64
from google.protobuf.empty_pb2 import Empty
from motion_planner import MotionPlanner
from utils.canbus import move_robot_forward

logger = logging.getLogger("Navigation Manager")


class NavigationManager:
    """Orchestrates waypoint navigation using MotionPlanner and track_follower service."""

    def __init__(
        self,
        filter_client: EventClient,
        controller_client: EventClient,
        motion_planner: MotionPlanner,
        no_stop: bool = False,
    ):
        self.filter_client = filter_client
        self.controller_client = controller_client
        self.motion_planner = motion_planner
        self.current_track_status: Optional[TrackStatusEnum] = None
        self.track_complete_event = asyncio.Event()
        self.track_failed_event = asyncio.Event()
        self.shutdown_requested = False
        self.navigation_progress: Dict[str, Track] = {}
        self.robot_positions: List[Dict] = []
        self.main_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.curr_segment_name: str = "start"
        self.no_stop = no_stop

    def record_robot_position(self, segment_name: str) -> None:
        """Record robot position before starting a track segment.

        Args:
            segment_name: Name of the track segment about to be executed
        """
        current_pose_obj = self.motion_planner.current_pose
        if current_pose_obj is not None:
            try:
                translation_array = np.asarray(current_pose_obj.a_from_b.translation)
                x = float(translation_array[0])
                y = float(translation_array[1])
                heading = float(current_pose_obj.a_from_b.rotation.log()[-1])

                position_record = {'segment_name': segment_name, 'x': x, 'y': y, 'heading': heading}

                self.robot_positions.append(position_record)
                logger.info(
                    f"📍 Recorded robot position for segment '{segment_name}': ({x:.2f}, {y:.2f}, "
                    f"{np.degrees(heading):.1f}°)"
                )

            except Exception as e:
                logger.error(f"❌ Failed to record robot position: {e}")

    async def set_track(self, track: Track) -> None:
        """Set the track for the track_follower to follow.

        Args:
            track: The track to follow
        """
        logger.info(f"📤 Setting track with {len(track.waypoints)} waypoints...")
        try:
            await self.controller_client.request_reply("/set_track", TrackFollowRequest(track=track))
            logger.info("✅ Track set successfully")
        except Exception as e:
            logger.error(f"❌ Failed to set track: {e}")
            raise

    async def start_following(self) -> None:
        """Start following the currently set track."""
        logger.info("🚀 Starting track following...")
        try:
            await self.controller_client.request_reply("/start", Empty())
            logger.info("✅ Track following started")
        except Exception as e:
            logger.error(f"❌ Failed to start track following: {e}")
            raise

    async def monitor_track_state(self) -> None:
        """Monitor the track_follower state and set events based on status."""
        logger.info("👁️  Starting track state monitoring...")

        try:
            config = self.controller_client.config
            subscription = config.subscriptions[0] if config.subscriptions else "/state"

            async for event, message in self.controller_client.subscribe(subscription, decode=True):
                if self.shutdown_requested:
                    logger.info("🛑 Monitor task received shutdown signal")
                    break

                if isinstance(message, TrackFollowerState):
                    await self._process_track_state(message)

        except asyncio.CancelledError:
            logger.info("🛑 Monitor task cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ Error monitoring track state: {e}")
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
                logger.info(f"📊 Track status changed: {status_name}")
            except Exception as e:
                logger.error(f"❌ Error getting status name: {e}")

        # Check for completion or failure
        if track_status == TrackStatusEnum.TRACK_COMPLETE:
            logger.info("🎉 Track completed successfully!")
            self.track_complete_event.set()

        elif track_status in [
            TrackStatusEnum.TRACK_FAILED,
            TrackStatusEnum.TRACK_ABORTED,
            TrackStatusEnum.TRACK_CANCELLED,
        ]:
            try:
                status_name = TrackStatusEnum.Name(track_status)
                logger.info(f"💥 Track failed with status: {status_name}")
            except Exception as e:
                logger.error(f"❌ Error getting status name: {e}")

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
                            logger.error(f"❌ Error getting failure mode name: {e}")

                    logger.info(f"Robot not controllable. Failure modes: {failure_modes}")
                except Exception as e:
                    logger.error(f"Robot not controllable. Failed to get failure modes: {e}")
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
                logger.warning(
                    f"⚠️  Cross-track error: {error.total_distance:.2f}m "
                    f"(lateral: {error.lateral_distance:.2f}m, "
                    f"longitudinal: {error.longitudinal_distance:.2f}m)"
                )

    async def _cleanup(self):
        """Clean up resources and cancel tasks."""
        logger.info("🧹 Starting cleanup...")

        self.shutdown_requested = True

        # Cancel monitor task
        if self.monitor_task and not self.monitor_task.done():
            logger.info("🛑 Cancelling monitor task...")
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        # Shutdown motion planner
        try:
            await self.motion_planner._shutdown()
        except Exception as e:
            logger.error(f"❌ Error shutting down motion planner: {e}")

        logger.info("✅ Cleanup completed")

    def get_user_choice(self) -> str:
        """Get user input for navigation choice.

        Returns:
            'continue' to continue to next waypoint, 'redo' to redo current segment
        """

        if self.no_stop or "waypoint" not in self.curr_segment_name:
            logger.info(
                "🚀 Either no stop mode enabled or going to the next row, automatically continuing to next waypoint"
            )
            return 'continue'

        print("\n" + "=" * 50)
        print("🤖 NAVIGATION CHOICE")
        print("=" * 50)
        print("What would you like to do next?")
        print("  1. Continue to the next waypoint")
        print("  2. Redo the current segment")
        print("  q. Quit navigation")
        print("-" * 50)

        while True:
            try:
                choice = input("Enter your choice (1/2/q): ").strip().lower()

                if choice in ['1', 'c', 'continue']:
                    print("➡️  Continuing to next waypoint...")
                    return 'continue'
                elif choice in ['2', 'r', 'redo']:
                    print("🔄 Redoing current segment...")
                    return 'redo'
                elif choice in ['q', 'quit', 'exit']:
                    print("🛑 Quitting navigation...")
                    return 'quit'
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or q.")

            except (EOFError, KeyboardInterrupt):
                print("\n🛑 Navigation interrupted by user")
                return 'quit'

    async def wait_for_track_completion(self, timeout: float = 60.0) -> bool:
        """Wait for track to complete or fail.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if track completed successfully, False if failed or timed out
        """
        logger.info(f"⏳ Waiting for track completion (timeout: {timeout}s)...")

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
                logger.warning("⏰ Timeout waiting for track completion")
                return False

            # Check which event was set
            if self.track_complete_event.is_set():
                return True
            elif self.track_failed_event.is_set():
                return False

        except Exception as e:
            logger.error(f"❌ Error waiting for track completion: {e}")
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
                logger.info("✅ Track segment completed successfully")
            else:
                logger.warning("❌ Track segment failed or timed out")

            return success

        except Exception as e:
            logger.error(f"❌ Error executing track: {e}")
            return False

    async def run_navigation(self) -> None:
        """Run the complete waypoint navigation sequence."""
        logger.info("🚁 Starting waypoint navigation...")

        # Start monitoring track state
        self.monitor_task = asyncio.create_task(self.monitor_track_state())

        try:
            segment_count = 0

            while not self.shutdown_requested:
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping navigation")
                    break

                # Get user choice before proceeding
                user_choice: str = self.get_user_choice()

                if user_choice == 'quit':
                    logger.info("🛑 User requested quit, stopping navigation")
                    self.shutdown_requested = True
                    break

                if user_choice == 'redo':
                    # Redo the last segment with recalculated path
                    logger.info("🔄 Redoing last segment with recalculated path...")

                    # Get a new track segment to the same target
                    (track_segment, segment_name) = await self.motion_planner.redo_last_segment()

                else:
                    (track_segment, segment_name) = await self.motion_planner.next_track_segment()

                # Get next track segment
                logger.info(f"\n--- Segment {segment_count + 1} ---")

                if track_segment is None:
                    logger.info("🏁 No more track segments. Navigation complete!")
                    self.record_robot_position("Final waypoint")
                    break

                self.record_robot_position(segment_name)
                logger.info(f"Got track segment '{segment_name}' with {len(track_segment.waypoints)} waypoints")
                self.curr_segment_name = segment_name
                self.navigation_progress[segment_name] = track_segment

                segment_count += 1
                logger.info(f"📍 Executing track segment {segment_count} with {len(track_segment.waypoints)} waypoints")

                # Execute the track segment
                success = await self.execute_single_track(track_segment)
                failed_attempts: int = 0

                while not success:
                    if self.shutdown_requested:
                        break
                    logger.warning(f"💥 Failed to execute segment {segment_count}. Stopping navigation.")
                    # We might have failed because the filter diverged or CANBUS timed out.
                    # We will try again
                    failed_attempts += 1
                    if segment_count == 1 and failed_attempts > 5:
                        # We're probably just getting stuck because the robot is too far from the path
                        # Let's move give it a "little push"
                        await move_robot_forward(time_goal=1.5)
                        logger.info(f"Moving robot forward | Failed attempts: {failed_attempts}")
                        failed_attempts = 0
                    track_segment, segment_name = await self.motion_planner.redo_last_segment()
                    success = await self.execute_single_track(track_segment)

            logger.info(f"🎯 Navigation completed after {segment_count} segments")

        except asyncio.CancelledError:
            logger.info("🛑 Navigation task cancelled")
            raise
        except KeyboardInterrupt:
            logger.info("\n🛑 Navigation interrupted by user")
        except Exception as e:
            logger.error(f"💥 Navigation failed with error: {e}")
        finally:
            # Cleanup
            await self._cleanup()


async def setup_clients(filter_config_path: Path, controller_config_path: Path) -> Tuple[EventClient, EventClient]:
    """Setup EventClients for filter and controller services.

    Args:
        filter_config_path: Path to filter service config
        controller_config_path: Path to controller service config

    Returns:
        Tuple of (filter_client, controller_client)
    """
    logger.info("🔧 Setting up service clients...")

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


def setup_signal_handlers(nav_manager: NavigationManager) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"\n🛑 Received signal {signum}, initiating shutdown...")
        nav_manager.shutdown_requested = True

        # Cancel the main navigation task
        if nav_manager.main_task and not nav_manager.main_task.done():
            nav_manager.main_task.cancel()

        # For immediate termination on second signal
        if hasattr(signal_handler, 'call_count'):
            signal_handler.call_count += 1
            if signal_handler.call_count > 1:
                logger.info("🛑 Second signal received, forcing exit")
                sys.exit(1)
        else:
            signal_handler.call_count = 1

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main(args) -> None:
    """Main function to orchestrate waypoint navigation."""
    nav_manager = None
    try:
        # Setup clients
        filter_client, controller_client = await setup_clients(args.filter_config, args.controller_config)

        # Initialize motion planner
        logger.info("🗺️  Initializing motion planner...")
        motion_planner = MotionPlanner(
            client=filter_client,
            tool_config_path=args.tool_config_path,
            waypoints_path=args.waypoints_path,
            last_row_waypoint_index=args.last_row_waypoint_index,
            turn_direction=args.turn_direction,
            row_spacing=args.row_spacing,
            headland_buffer=args.headland_buffer,
        )

        # Create nav_manager
        nav_manager = NavigationManager(
            filter_client=filter_client,
            controller_client=controller_client,
            motion_planner=motion_planner,
            no_stop=args.no_stop,
        )

        setup_signal_handlers(nav_manager=nav_manager)
        # Store the main task reference for cancellation
        nav_manager.main_task = asyncio.current_task()

        # Run navigation
        await nav_manager.run_navigation()

    except asyncio.CancelledError:
        logger.info("🛑 Main task cancelled")
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt in main")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")

    finally:
        # Save navigation progress to JSON file
        progress_path = Path("./visualization/navigation_progress.json")
        try:
            # Convert protobuf Track objects to simple [x, y, heading] format
            serializable_progress = {}
            for segment_name, track in nav_manager.navigation_progress.items():
                x: list[float] = []
                y: list[float] = []
                heading: list[float] = []

                track_waypoints = [Pose3F64.from_proto(pose) for pose in track.waypoints]
                for pose in track_waypoints:
                    x.append(pose.a_from_b.translation[0])
                    y.append(pose.a_from_b.translation[1])
                    heading.append(pose.a_from_b.rotation.log()[-1])

                serializable_progress[segment_name] = {
                    'waypoints_count': len(track.waypoints),
                    'x': x,
                    'y': y,
                    'heading': heading,
                }

            with open(progress_path, "w") as f:
                json.dump(serializable_progress, f, indent=2)
            logger.info(f"✅ Navigation progress saved to {progress_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save navigation progress: {e}")

        positions_path = Path("./visualization/robot_positions.json")
        try:
            with open(positions_path, "w") as f:
                json.dump(nav_manager.robot_positions, f, indent=2)
            logger.info(f"✅ Robot positions saved to {positions_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save robot positions: {e}")

        if not nav_manager.shutdown_requested:
            await nav_manager._cleanup()

        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Waypoint navigation using MotionPlanner and track_follower service"
    )

    # Required arguments
    parser.add_argument("--filter-config", type=Path, required=True, help="Path to filter service config JSON file")
    parser.add_argument("--waypoints-path", type=Path, required=True, help="Path to waypoints JSON file (Track format)")
    parser.add_argument("--tool-config-path", type=Path, required=True, help="Path to tool configuration JSON file")
    parser.add_argument(
        "--controller-config", type=Path, required=True, help="Path to track_follower service config JSON file"
    )

    # MotionPlanner configuration
    parser.add_argument(
        "--last-row-waypoint-index",
        type=int,
        default=6,
        help="Index of the last waypoint in the current row (default: 6)",
    )
    parser.add_argument(
        "--turn-direction",
        choices=["left", "right"],
        default="left",
        help="Direction to turn at row ends (default: left)",
    )
    parser.add_argument("--row-spacing", type=float, default=6.0, help="Spacing between rows in meters (default: 3.0)")
    parser.add_argument(
        "--headland-buffer",
        type=float,
        default=2.0,
        help="Buffer distance for headland maneuvers in meters (default: 2.0)",
    )
    parser.add_argument("--no-stop", action="store_true", help="Disable stopping at each waypoint")

    args = parser.parse_args()

    # Validate file paths
    for path_arg in [args.filter_config, args.waypoints_path, args.controller_config]:
        if not path_arg.exists():
            logger.error(f"❌ File not found: {path_arg}")
            sys.exit(1)

    # Run the main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("\n🛑 Final keyboard interrupt")
    except Exception as e:
        logger.error(f"💥 Unhandled exception: {e}")
    finally:
        logger.info("👋 Script terminated")
        sys.exit(0)
