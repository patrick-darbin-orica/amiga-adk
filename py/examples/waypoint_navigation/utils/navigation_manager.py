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
        actuator_open_seconds: float = 0.2,#6.5, 
        actuator_close_seconds: float = 0.3,#7,
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
        self.vision_active = False
        self._controller_lock = getattr(self, "_controller_lock", asyncio.Lock())


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

    async def _cancel_following(self):
        async with self._controller_lock:
            try:
                await self.controller_client.request_reply("/cancel", Empty())
            except Exception:
                # ok if already idle
                pass

    async def _set_track_locked(self, track):
        req = TrackFollowRequest(track=track)
        async with self._controller_lock:
            await self.controller_client.request_reply("/set_track", req)

    async def _start_following_locked(self):
        async with self._controller_lock:
            await self.controller_client.request_reply("/start", Empty())

    async def _pause_following_locked(self):
        async with self._controller_lock:
            await self.controller_client.request_reply("/pause", Empty())
        
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
                    f"Recorded robot position for segment '{segment_name}': "
                    f"({x:.2f}, {y:.2f}, {np.degrees(heading):.1f}°)"
                )
            except Exception as e:
                logger.error(f"FAIL: Record robot position: {e}")

    async def set_track(self, track: Track) -> None:
        """Set the track for the track_follower to follow."""
        logger.info(
            f"Setting track with {len(track.waypoints)} waypoints...")
        try:
            # await self.controller_client.request_reply("/set_track", TrackFollowRequest(track=track))
            if getattr(self, "vision_active", False):
                # let vision finish its cancel→set→start
                await asyncio.sleep(0.05)
            await self._set_track_locked(track)
            logger.info("SUCCESS: Track set")
        except Exception as e:
            logger.error(f"FAIL: Track not set {e}")
            raise
        
    async def replace_track_and_start(self, track) -> None:
        async with self._controller_lock:
            # 0) ensure robot is controllable / auto-mode on (optional but smart)
            st = await self._get_follower_state()
            if not st.robot_status.controllable:
                raise RuntimeError(f"Robot not controllable. Failure modes: {[m.name for m in st.robot_status.failure_modes]}")

            # 1) cancel current (ignore errors if idle)
            try:
                # await self.controller_client.request_reply("/cancel", Empty())
                await self._cancel_following()
            except Exception:
                pass

            # 2) wait until not FOLLOWING anymore
            await self._wait_until(lambda s: s.status != TrackFollowerState.TRACK_FOLLOWING, timeout=3.0)

            # 3) set the new track (proto, not bytes)
            req = TrackFollowRequest(track=track)
            await self.controller_client.request_reply("/set_track", req)

            # 4) wait until LOADED, then start
            await self._wait_until(lambda s: s.status == TrackFollowerState.TRACK_LOADED, timeout=2.0)
            await self.controller_client.request_reply("/start", Empty())

            # 5) confirm it actually started
            await self._wait_until(lambda s: s.status == TrackFollowerState.TRACK_FOLLOWING, timeout=2.0)

    async def _get_follower_state(self) -> TrackFollowerState:
        return await self.controller_client.request_reply("/get_state", Empty(), decode=True)

    async def _wait_until(self, pred, timeout: float):
        import time
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            st = await self._get_follower_state()
            if pred(st):
                return st
            await asyncio.sleep(0.05)
        raise TimeoutError("wait condition not met")

    async def start_following(self) -> None:
        """Start following the currently set track."""
        logger.info("Starting track following...")
        try:
            # await self.controller_client.request_reply("/start", Empty())
            if getattr(self, "vision_active", False):
                # let vision finish its cancel→set→start
                await asyncio.sleep(0.05)
            await self._start_following_locked()
            logger.info("START: Track following")
        except Exception as e:
            logger.error(f"FAIL: track following not started: {e}")
            raise
        
    async def monitor_track_state(self) -> None:
        """Monitor the track_follower state and set events based on status."""
        logger.info("Starting track state monitoring...")

        try:
            config = self.controller_client.config
            subscription = config.subscriptions[0] if config.subscriptions else "/state"

            async for event, message in self.controller_client.subscribe(subscription, decode=True):
                if self.shutdown_requested:
                    logger.info("SHUTDOWN: Monitor task received shutdown signal")
                    break

                if isinstance(message, TrackFollowerState):
                    await self._process_track_state(message)

        except asyncio.CancelledError:
            logger.info("STOP: Monitor task cancelled")
            raise
        except Exception as e:
            logger.error(f"ERROR: Monitoring track state: {e}")
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
                logger.info(f"Track status changed: {status_name}")
            except Exception as e:
                logger.error(f"ERROR: getting status name: {e}")

        # Check for completion or failure
        if track_status == TrackStatusEnum.TRACK_COMPLETE:
            logger.info("SUCCESS: Track completed")
            self.track_complete_event.set()

        elif track_status in [
            TrackStatusEnum.TRACK_FAILED,
            TrackStatusEnum.TRACK_ABORTED,
            TrackStatusEnum.TRACK_CANCELLED,
        ]:
            try:
                status_name = TrackStatusEnum.Name(track_status)
                logger.info(f"ERROR: Track failed with status: {status_name}")
            except Exception as e:
                logger.error(f"ERROR: getting status name: {e}")

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
                                f"ERROR: getting failure mode name: {e}")

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
                    f"ERROR: Cross-track: {error.total_distance:.2f}m "
                    f"(lateral: {error.lateral_distance:.2f}m, "
                    f"longitudinal: {error.longitudinal_distance:.2f}m)"
                )

    async def _cleanup(self):
        """Clean up resources and cancel tasks."""
        logger.info("Starting cleanup...")

        self.shutdown_requested = True

        if self.monitor_task and not self.monitor_task.done():
            logger.info("Cancelling monitor task...")
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        try:
            await self.motion_planner._shutdown()
        except Exception as e:
            logger.error(f"ERROR when shutting down motion planner: {e}")

        logger.info("Cleanup completed")

    async def _hold_if_vision_active(self) -> None:
        """Pause row/waypoint progression while a vision override is running."""
        while self.vision_active and not self.shutdown_requested:
            await asyncio.sleep(0.05)

    def get_user_choice(self) -> str:
        """Get user input for navigation choice."""
        if self.no_stop or "waypoint" not in self.curr_segment_name:
            logger.info(
                "Either no stop mode enabled or going to the next row, automatically continuing to next waypoint"
            )
            return "continue"

        print("\n" + "=" * 50)
        print("NAVIGATION CHOICE")
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
                    print("Continuing to next waypoint...")
                    return "continue"
                elif choice in ["2", "r", "redo"]:
                    print("Redoing current segment...")
                    return "redo"
                elif choice in ["q", "quit", "exit"]:
                    print("Quitting navigation...")
                    return "quit"
                else:
                    print("Invalid choice. Please enter 1, 2, or q.")

            except (EOFError, KeyboardInterrupt):
                print("\nNavigation interrupted by user")
                return "quit"

    async def wait_for_track_completion(self, timeout: float = 60.0) -> bool:
        """Wait for track to complete or fail."""
        logger.info(f"Waiting for track completion (timeout: {timeout}s)...")

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
                logger.warning("Timeout waiting for track completion")
                return False

            if self.track_complete_event.is_set():
                return True
            elif self.track_failed_event.is_set():
                return False

        except Exception as e:
            logger.error(f"ERROR: waiting for track completion: {e}")
            return False

        return False

    async def execute_single_track(self, track: Track, timeout: float = 30.0, *, do_post_actions: bool = True) -> bool:
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
                logger.info("SUCCESS: Track segment completed")
                if do_post_actions and self.actuator_enabled:
                    # 1) wait briefly
                    await asyncio.sleep(2.0)

                    # 2) Deploy plumbob (tool already over hole)
                    await trigger_dipbob("can0")
                    logger.info("Deploying dipbob")
                    await asyncio.sleep(3.0)  # TODO: swap for measurement await

                    # 3) Move forward so robot origin is over the hole
                    origin_track = await self.motion_planner.create_tool_to_origin_segment()
                    ok2 = await self.execute_single_track(origin_track, timeout=15.0, do_post_actions=False)
                    if not ok2:
                        logger.warning("tool→origin micro-segment failed; skipping chute pulse")
                        return success  # don't open chute if failed

                    # 4) Open/close chute
                    await self.actuator.pulse_sequence(
                        open_seconds=self.actuator_open_seconds,
                        close_seconds=self.actuator_close_seconds,
                        rate_hz=self.actuator_rate_hz,
                        settle_before=3.0,
                        settle_between=0.0,
                        wait_for_enter_between=True,
                        enter_prompt="Hole measured. Press ENTER to close the chute...",
                        enter_timeout=30.0,      # optional: add a safety timeout if you want
                    )
            else:
                logger.warning("ERROR: Track segment failed or timed out")

            return success

        except Exception as e:
            logger.error(f"ERROR: executing track: {e}")
            return False

    async def run_navigation(self) -> None:
        """Run the complete waypoint navigation sequence."""
        logger.info("Starting waypoint navigation...")
        self.monitor_task = asyncio.create_task(self.monitor_track_state())

        try:
            segment_count = 0

            while not self.shutdown_requested:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping navigation")
                    break
                
                await self._hold_if_vision_active()
                user_choice: str = self.get_user_choice()

                if user_choice == "quit":
                    logger.info("User requested quit, stopping navigation")
                    self.shutdown_requested = True
                    break
                
                # Respect any vision override before (re)planning
                await self._hold_if_vision_active()
                if user_choice == "redo":
                    logger.info(
                        "Redoing last segment with recalculated path...")
                    (track_segment, segment_name) = await self.motion_planner.redo_last_segment()
                else:
                    (track_segment, segment_name) = await self.motion_planner.next_track_segment()

                logger.info(f"\n--- Segment {segment_count + 1} ---")

                if track_segment is None:
                    logger.info(
                        "No more track segments. Navigation complete!")
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
                    f"Executing track segment {segment_count} with {len(track_segment.waypoints)} waypoints"
                )

                # Don’t launch execution while vision overrides are active
                await self._hold_if_vision_active()
                success = await self.execute_single_track(track_segment)
                failed_attempts: int = 0

                while not success:
                    if self.shutdown_requested:
                        break
                    logger.warning(
                        f"Failed to execute segment {segment_count}. Stopping navigation.")
                    failed_attempts += 1
                    if segment_count == 1 and failed_attempts > 5:
                        await move_robot_forward(time_goal=1.5)
                        logger.info(
                            f"Moving robot forward | Failed attempts: {failed_attempts}")
                        failed_attempts = 0
                    track_segment, segment_name = await self.motion_planner.redo_last_segment()
                    success = await self.execute_single_track(track_segment)

            logger.info(
                f"Navigation completed after {segment_count} segments")

        except asyncio.CancelledError:
            logger.info("Navigation task cancelled")
            raise
        except KeyboardInterrupt:
            logger.info("\nNavigation interrupted by user")
        except Exception as e:
            logger.error(f"Navigation failed with error: {e}")
        finally:
            await self._cleanup()
