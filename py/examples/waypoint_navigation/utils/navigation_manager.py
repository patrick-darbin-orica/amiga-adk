# navigation_manager.py
from __future__ import annotations
import asyncio
import logging
import numpy as np
from typing import Optional, TYPE_CHECKING, Dict, List

from farm_ng.core.event_client import EventClient
from farm_ng.track.track_pb2 import (
    RobotStatus,
    Track,
    TrackFollowerState,
    TrackFollowRequest,
    TrackStatusEnum,
)
from google.protobuf.empty_pb2 import Empty
from utils.actuator import BaseActuator, NullActuator
from utils.canbus import move_robot_forward, trigger_dipbob

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from motion_planner import MotionPlanner


class NavigationManager:
    """Orchestrates waypoint navigation using MotionPlanner and track_follower service.
       Optionally pulses an H-bridge (linear actuator) in forward after each completed segment.
    """

    def __init__(
        self,
        filter_client: EventClient,
        controller_client: EventClient,
        motion_planner: MotionPlanner,
        no_stop: bool = False,
        actuator: BaseActuator | None = None,
        # Actuator / CAN options
        canbus_client: Optional[EventClient] = None,
        actuator_enabled: bool = True,  # TODO: Remove
        actuator_id: int = 0,
        actuator_open_seconds: float = 1.5,  # TODO: Remove
        actuator_close_seconds: float = 1.5,
        actuator_rate_hz: float = 10.0,
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
        self.actuator = actuator or NullActuator()

        # Actuator / CAN config
        self.canbus_client = canbus_client
        self.actuator_enabled = actuator_enabled and (
            self.canbus_client is not None)
        self.actuator_id = actuator_id
        self.actuator_open_seconds = max(0.0, actuator_open_seconds)
        self.actuator_close_seconds = max(0.0, actuator_close_seconds)
        self.actuator_rate_hz = max(0.1, actuator_rate_hz)

        if actuator_enabled and self.canbus_client is None:
            logger.warning(
                "Actuator was enabled but no CAN bus client provided; disabling actuator pulses.")
            self.actuator_enabled = False

    def record_robot_position(self, segment_name: str) -> None:
        """Record robot position before starting a track segment.

        Args:
            segment_name: Name of the track segment about to be executed
        """
        current_pose_obj = self.motion_planner.current_pose
        if current_pose_obj is not None:
            try:
                translation_array = np.asarray(
                    current_pose_obj.a_from_b.translation)
                x = float(translation_array[0])
                y = float(translation_array[1])
                heading = float(current_pose_obj.a_from_b.rotation.log()[-1])

                position_record = {"segment_name": segment_name,
                                   "x": x, "y": y, "heading": heading}

                self.robot_positions.append(position_record)
                logger.info(
                    f"📍 Recorded robot position for segment '{segment_name}': "
                    f"({x:.2f}, {y:.2f}, {np.degrees(heading):.1f}°)"
                )
            except Exception as e:
                logger.error(f"❌ Failed to record robot position: {e}")

    async def set_track(self, track: Track) -> None:
        """Set the track for the track_follower to follow."""
        logger.info(
            f"📤 Setting track with {len(track.waypoints)} waypoints...")
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
        """Process incoming track follower state messages."""
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
                            failure_modes.append(f"UNKNOWN({mode})")
                            logger.error(
                                f"❌ Error getting failure mode name: {e}")

                    logger.info(
                        f"Robot not controllable. Failure modes: {failure_modes}")
                except Exception as e:
                    logger.error(
                        f"Robot not controllable. Failed to get failure modes: {e}")
            self.track_failed_event.set()

        # Log cross-track error if available
        if (
            hasattr(state, "progress")
            and state.progress
            and hasattr(state.progress, "cross_track_error")
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

        if self.monitor_task and not self.monitor_task.done():
            logger.info("🛑 Cancelling monitor task...")
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        try:
            await self.motion_planner._shutdown()
        except Exception as e:
            logger.error(f"❌ Error shutting down motion planner: {e}")

        logger.info("✅ Cleanup completed")

    def get_user_choice(self) -> str:
        """Get user input for navigation choice."""
        if self.no_stop or "waypoint" not in self.curr_segment_name:
            logger.info(
                "🚀 Either no stop mode enabled or going to the next row, automatically continuing to next waypoint"
            )
            return "continue"

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

                if choice in ["1", "c", "continue"]:
                    print("➡️  Continuing to next waypoint...")
                    return "continue"
                elif choice in ["2", "r", "redo"]:
                    print("🔄 Redoing current segment...")
                    return "redo"
                elif choice in ["q", "quit", "exit"]:
                    print("🛑 Quitting navigation...")
                    return "quit"
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or q.")

            except (EOFError, KeyboardInterrupt):
                print("\n🛑 Navigation interrupted by user")
                return "quit"

    async def wait_for_track_completion(self, timeout: float = 60.0) -> bool:
        """Wait for track to complete or fail."""
        logger.info(f"⏳ Waiting for track completion (timeout: {timeout}s)...")

        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self.track_complete_event.wait()),
                    asyncio.create_task(self.track_failed_event.wait()),
                ],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if not done:
                logger.warning("⏰ Timeout waiting for track completion")
                return False

            if self.track_complete_event.is_set():
                return True
            elif self.track_failed_event.is_set():
                return False

        except Exception as e:
            logger.error(f"❌ Error waiting for track completion: {e}")
            return False

        return False

    async def execute_single_track(self, track: Track, timeout: float = 30.0) -> bool:
        """Execute a single track segment and wait for completion."""
        self.track_complete_event.clear()
        self.track_failed_event.clear()

        try:
            await self.set_track(track)
            await asyncio.sleep(1.0)  # ensure track is set
            # TODO: Add IMUjiggle function
            await self.start_following()

            success = await self.wait_for_track_completion(timeout)

            if success:
                logger.info("✅ Track segment completed successfully")
                
                await asyncio.sleep(2.0)
                
                if self.actuator_enabled:
                    await trigger_dipbob("can0")
                    logger.info("Deploying dipbob")
                    await asyncio.sleep(5.0) # TODO: Swap for awaiting measurement
                    await self.actuator.pulse_sequence(
                        open_seconds=self.actuator_open_seconds,
                        close_seconds=self.actuator_close_seconds,
                        rate_hz=self.actuator_rate_hz,
                        settle_before=3.0,     # your current pre‑pulse wait
                        settle_between=1.0
                    )
            else:
                logger.warning("❌ Track segment failed or timed out")

            return success

        except Exception as e:
            logger.error(f"❌ Error executing track: {e}")
            return False

    async def run_navigation(self) -> None:
        """Run the complete waypoint navigation sequence."""
        logger.info("🚁 Starting waypoint navigation...")
        self.monitor_task = asyncio.create_task(self.monitor_track_state())

        try:
            segment_count = 0

            while not self.shutdown_requested:
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping navigation")
                    break

                user_choice: str = self.get_user_choice()

                if user_choice == "quit":
                    logger.info("🛑 User requested quit, stopping navigation")
                    self.shutdown_requested = True
                    break

                if user_choice == "redo":
                    logger.info(
                        "🔄 Redoing last segment with recalculated path...")
                    (track_segment, segment_name) = await self.motion_planner.redo_last_segment()
                else:
                    (track_segment, segment_name) = await self.motion_planner.next_track_segment()

                logger.info(f"\n--- Segment {segment_count + 1} ---")

                if track_segment is None:
                    logger.info(
                        "🏁 No more track segments. Navigation complete!")
                    self.record_robot_position("Final waypoint")
                    break

                self.record_robot_position(segment_name)
                logger.info(
                    f"Got track segment '{segment_name}' with {len(track_segment.waypoints)} waypoints"
                )
                self.curr_segment_name = segment_name
                self.navigation_progress[segment_name] = track_segment

                segment_count += 1
                logger.info(
                    f"📍 Executing track segment {segment_count} with {len(track_segment.waypoints)} waypoints"
                )

                success = await self.execute_single_track(track_segment)
                failed_attempts: int = 0

                while not success:
                    if self.shutdown_requested:
                        break
                    logger.warning(
                        f"💥 Failed to execute segment {segment_count}. Stopping navigation.")
                    failed_attempts += 1
                    if segment_count == 1 and failed_attempts > 5:
                        await move_robot_forward(time_goal=1.5)
                        logger.info(
                            f"Moving robot forward | Failed attempts: {failed_attempts}")
                        failed_attempts = 0
                    track_segment, segment_name = await self.motion_planner.redo_last_segment()
                    success = await self.execute_single_track(track_segment)

            logger.info(
                f"🎯 Navigation completed after {segment_count} segments")

        except asyncio.CancelledError:
            logger.info("🛑 Navigation task cancelled")
            raise
        except KeyboardInterrupt:
            logger.info("\n🛑 Navigation interrupted by user")
        except Exception as e:
            logger.error(f"💥 Navigation failed with error: {e}")
        finally:
            await self._cleanup()
