# navigation_manager.py
from __future__ import annotations
import asyncio
import logging
import numpy as np
from pathlib import Path
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
from utils.canbus import trigger_dipbob, imu_wiggle
from utils.navigation_state import set_navigation_state
from utils.hole_alignment import align_with_oak0

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
        actuator_open_seconds: float = 6.5,
        actuator_close_seconds: float = 7,
        actuator_rate_hz: float = 10.0,
        # Hole alignment options
        hole_alignment_enabled: bool = True,
        hole_alignment_model_path: Optional[Path] = None,
        hole_alignment_tolerance_px: int = 40,
        hole_alignment_move_gain: float = 0.001,
        hole_alignment_derivative_gain: float = 0.002,
        hole_alignment_max_velocity: float = 0.15,
        hole_alignment_timeout: float = 30.0,
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
        self.actuator_deploying = False
        self.track_executing = False  # Track if robot is actively executing a track segment
        self.waiting_for_collar_detection = False  # Track if robot is stopped at approach position waiting for vision
        self._controller_lock = getattr(self, "_controller_lock", asyncio.Lock())
        self.last_failure_modes: List[str] = []  # Store last detected failure modes


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

        # Hole alignment configuration
        self.hole_alignment_enabled = hole_alignment_enabled and (self.canbus_client is not None)
        self.hole_alignment_model_path = hole_alignment_model_path or Path(__file__).parent.parent / "detection" / "best.engine"
        self.hole_alignment_tolerance_px = hole_alignment_tolerance_px
        self.hole_alignment_move_gain = hole_alignment_move_gain
        self.hole_alignment_derivative_gain = hole_alignment_derivative_gain
        self.hole_alignment_max_velocity = hole_alignment_max_velocity
        self.hole_alignment_timeout = hole_alignment_timeout

        if hole_alignment_enabled and self.canbus_client is None:
            logger.warning(
                "Hole alignment was enabled but no CAN bus client provided; disabling hole alignment.")
            self.hole_alignment_enabled = False

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
            await self._set_track_locked(track)
            # logger.info("SUCCESS: Track set")
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
        # logger.info("Starting track following...")
        try:
            await self._start_following_locked()
            # logger.info("START: Track following")
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

        # Log status changes (only for aborted/cancelled)
        if prev_status != track_status:
            try:
                status_name = TrackStatusEnum.Name(track_status)
                # Only log if status is TRACK_ABORTED or TRACK_CANCELLED
                if track_status in [TrackStatusEnum.TRACK_ABORTED, TrackStatusEnum.TRACK_CANCELLED]:
                    logger.info(f"Track status changed: {status_name}")
                # Log ALL status changes for debugging
                logger.info(f"[TRACK DEBUG] Status change: {TrackStatusEnum.Name(prev_status) if prev_status is not None else 'None'} -> {status_name}")
                # Update Flask GUI state (always update GUI regardless of status)
                set_navigation_state(track_status=status_name)
            except Exception as e:
                logger.error(f"ERROR: getting status name: {e}")

        # Check for completion or failure
        if track_status == TrackStatusEnum.TRACK_COMPLETE:
            logger.info("[TRACK DEBUG] TRACK_COMPLETE received, setting track_complete_event")
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
                    # Store failure modes for later processing
                    self.last_failure_modes = failure_modes
                except Exception as e:
                    logger.error(
                        f"Robot not controllable. Failed to get failure modes: {e}")
                    self.last_failure_modes = []
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

    async def _wait_for_cone_or_skip(self, vision_timeout: float = 10.0) -> bool:
        """
        Wait for vision to detect a cone in the search zone if vision is enabled.
        If vision is NOT enabled, proceed immediately to CSV waypoint.
        If vision IS enabled, wait up to vision_timeout seconds for cone detection.
        Returns True to proceed, False if shutdown requested.

        Args:
            vision_timeout: Maximum time to wait for collar detection (seconds, default 10.0)
        """
        from pathlib import Path

        # Check if vision is running by looking for .vision_running flag file
        vision_flag_file = Path(__file__).parent.parent / ".vision_running"
        vision_enabled = vision_flag_file.exists()

        if not vision_enabled:
            logger.info("[VISION] Vision not enabled (.vision_running flag not found), proceeding to CSV waypoint")
            return True  # Vision not enabled, proceed to CSV waypoint immediately

        # Vision is enabled - only reset cone detection flag if not already detected (avoid race condition)
        if not getattr(self, "cone_detected_for_current_wp", False):
            self.cone_detected_for_current_wp = False
        self.current_waypoint_start_time = asyncio.get_event_loop().time()

        logger.info(f"[VISION] Vision enabled - waiting up to {vision_timeout}s for collar detection...")

        # Wait for cone detection with timeout
        start_time = asyncio.get_event_loop().time()
        while not getattr(self, "cone_detected_for_current_wp", False):
            if self.shutdown_requested:
                return False

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= vision_timeout:
                logger.warning(f"[VISION] Timeout after {vision_timeout}s - no collar detected, proceeding to CSV waypoint")
                return True

            await asyncio.sleep(0.1)

        logger.info("[VISION] Collar detected in search zone - proceeding to refined position")
        return True

    def get_user_choice(self) -> str:
        """Get user input for navigation choice (DEBUG: always continue)."""
        logger.info("DEBUG: Auto-selecting 'continue' (choice 1).")
        return "continue"

    # def get_user_choice(self) -> str:
    #     """Get user input for navigation choice."""
    #     if self.no_stop or "waypoint" not in self.curr_segment_name:
    #         logger.info(
    #             "Either no stop mode enabled or going to the next row, automatically continuing to next waypoint"
    #         )
    #         return "continue"

    #     print("\n" + "=" * 50)
    #     print("NAVIGATION CHOICE")
    #     print("=" * 50)
    #     print("What would you like to do next?")
    #     print("  1. Continue to the next waypoint")
    #     print("  2. Redo the current segment")
    #     print("  q. Quit navigation")
    #     print("-" * 50)

    #     while True:
    #         try:
    #             choice = input("Enter your choice (1/2/q): ").strip().lower()

    #             if choice in ["1", "c", "continue"]:
    #                 print("Continuing to next waypoint...")
    #                 return "continue"
    #             elif choice in ["2", "r", "redo"]:
    #                 print("Redoing current segment...")
    #                 return "redo"
    #             elif choice in ["q", "quit", "exit"]:
    #                 print("Quitting navigation...")
    #                 return "quit"
    #             else:
    #                 print("Invalid choice. Please enter 1, 2, or q.")

    #         except (EOFError, KeyboardInterrupt):
    #             print("\nNavigation interrupted by user")
    #             return "quit"

    async def wait_for_track_completion(self, timeout: float = 60.0) -> bool:
        """Wait for track to complete or fail."""
        logger.info(f"Waiting for track completion (timeout: {timeout}s)...")
        logger.info(f"[TRACK DEBUG] Current track status before wait: {TrackStatusEnum.Name(self.current_track_status) if self.current_track_status is not None else 'None'}")
        logger.info(f"[TRACK DEBUG] Monitor task running: {self.monitor_task is not None and not self.monitor_task.done() if self.monitor_task else 'No monitor task'}")

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
                logger.warning(f"[TRACK DEBUG] Final track status after timeout: {TrackStatusEnum.Name(self.current_track_status) if self.current_track_status is not None else 'None'}")
                logger.warning(f"[TRACK DEBUG] Monitor task still alive: {not self.monitor_task.done() if self.monitor_task else 'No monitor task'}")
                return False

            if self.track_complete_event.is_set():
                logger.info("[TRACK DEBUG] Track completed successfully")
                return True
            elif self.track_failed_event.is_set():
                logger.warning("[TRACK DEBUG] Track failed")
                return False

        except Exception as e:
            logger.error(f"ERROR: waiting for track completion: {e}")
            logger.error(f"[TRACK DEBUG] Exception traceback:", exc_info=True)
            return False

        return False

    async def execute_single_track(self, track: Track, timeout: float = 30.0, *, do_post_actions: bool = True, max_filter_retries: int = 3, max_canbus_retries: int = 2) -> bool:
        """Execute a single track segment and wait for completion.

        If FILTER_DIVERGED is detected, will attempt to wiggle the robot and retry.
        If CANBUS_TIMEOUT is detected, will wait and retry.

        Args:
            track: Track segment to execute
            timeout: Timeout in seconds for track completion
            do_post_actions: Whether to perform actuator deployment after completion
            max_filter_retries: Maximum number of wiggle retries on FILTER_DIVERGED (default: 3)
            max_canbus_retries: Maximum number of retries on CANBUS_TIMEOUT (default: 2)
        """
        filter_retry_count = 0
        canbus_retry_count = 0

        while filter_retry_count <= max_filter_retries or canbus_retry_count <= max_canbus_retries:
            self.track_complete_event.clear()
            self.track_failed_event.clear()
            self.last_failure_modes = []  # Clear previous failure modes

            try:
                # Set flag to prevent vision from overriding waypoints during execution
                self.track_executing = True

                await self.set_track(track)
                await asyncio.sleep(1.0)  # ensure track is set
                await self.start_following()

                success = await self.wait_for_track_completion(timeout)

                # Check if failure was due to FILTER_DIVERGED
                if not success and "FILTER_DIVERGED" in self.last_failure_modes:
                    filter_retry_count += 1

                    if filter_retry_count <= max_filter_retries:
                        logger.warning(f"FILTER_DIVERGED detected. Attempting wiggle recovery (attempt {filter_retry_count}/{max_filter_retries})...")

                        # Cancel current track
                        await self._cancel_following()

                        # Perform wiggle if canbus client is available
                        if self.canbus_client is not None:
                            wiggle_success = await imu_wiggle(
                                canbus_client=self.canbus_client,
                                filter_client=self.filter_client,
                                duration_seconds=3.0,
                                angular_velocity=0.3,
                                check_convergence=True,
                                max_attempts=1  # One wiggle attempt per retry
                            )

                            if wiggle_success:
                                logger.info("✓ Filter converged after wiggle. Retrying track segment...")
                                continue  # Retry the track
                            else:
                                logger.warning("Filter still diverged after wiggle.")
                        else:
                            logger.warning("Cannot wiggle: no canbus client available")
                    else:
                        logger.error(f"Max filter retries ({max_filter_retries}) exceeded. Giving up on this segment.")
                        return False

                # Check if failure was due to CANBUS_TIMEOUT
                elif not success and "CANBUS_TIMEOUT" in self.last_failure_modes:
                    canbus_retry_count += 1

                    if canbus_retry_count <= max_canbus_retries:
                        # Wait for canbus service to recover before retrying
                        recovery_delay = 3.0  # Give canbus service time to clear backlog
                        # logger.warning(f"CANBUS_TIMEOUT detected. Waiting {recovery_delay}s for service recovery (attempt {canbus_retry_count}/{max_canbus_retries})...")

                        # Cancel current track
                        await self._cancel_following()
                        await asyncio.sleep(recovery_delay)

                        logger.info("Retrying track segment after canbus recovery delay...")
                        continue  # Retry the track
                    else:
                        logger.error(f"Max canbus retries ({max_canbus_retries}) exceeded. Giving up on this segment.")
                        return False
                else:
                    # Success or non-filter/canbus failure - break out of retry loop
                    break

            except Exception as e:
                logger.error(f"Exception during track execution: {e}")
                self.track_executing = False
                raise

        success = not self.track_failed_event.is_set()

        # Clear flag after track execution completes (before deployment)
        self.track_executing = False

        if success:
            # logger.info("SUCCESS: Track segment completed")

            # Add recovery delay to allow CAN bus to settle before next segment
            # This helps prevent CANBUS_TIMEOUT failures in track follower
            recovery_delay = 0.5  # 500ms recovery time
            logger.debug(f"CAN bus recovery delay: {recovery_delay}s")
            await asyncio.sleep(recovery_delay)

            if do_post_actions and self.actuator_enabled:
                # Set flag to disable vision during deployment
                self.actuator_deploying = True

                try:
                    # 1) Wait briefly after parking
                    logger.info("[DEPLOY DEBUG] Starting post-actions, waiting 2s...")
                    await asyncio.sleep(2.0)

                    # 2) Perform hole alignment with oak0 (downward-facing camera)
                    if self.hole_alignment_enabled:
                        logger.info("[HOLE ALIGN] Starting fine alignment using oak0 camera...")
                        alignment_success = await align_with_oak0(
                            canbus_client=self.canbus_client,
                            model_path=self.hole_alignment_model_path,
                            tolerance_px=self.hole_alignment_tolerance_px,
                            move_gain=self.hole_alignment_move_gain,
                            derivative_gain=self.hole_alignment_derivative_gain,
                            max_velocity=self.hole_alignment_max_velocity,
                            timeout_seconds=self.hole_alignment_timeout,
                        )

                        if alignment_success:
                            logger.info("[HOLE ALIGN] ✓ Hole alignment completed successfully")
                        else:
                            logger.warning("[HOLE ALIGN] ⚠ Hole alignment failed or timed out, proceeding anyway...")
                    else:
                        logger.info("[HOLE ALIGN] Hole alignment disabled, skipping...")

                    # 3) Deploy plumbob (tool should now be perfectly aligned over hole)
                    logger.info("[DEPLOY DEBUG] Calling trigger_dipbob...")
                    try:
                        await trigger_dipbob("can0", timeout=5.0)
                        logger.info("[DEPLOY DEBUG] Dipbob triggered successfully")
                        await asyncio.sleep(7.0)  # TODO: swap for measurement await
                    except asyncio.TimeoutError:
                        logger.warning("[DEPLOY DEBUG] Dipbob timeout - device may be unplugged, continuing anyway...")
                    except Exception as e:
                        logger.warning(f"[DEPLOY DEBUG] Dipbob error: {e}, continuing anyway...")

                    # 4) Move forward so robot origin is over the hole
                    origin_track = await self.motion_planner.create_tool_to_origin_segment()
                    ok2 = await self.execute_single_track(origin_track, timeout=15.0, do_post_actions=False)
                    if not ok2:
                        logger.warning("tool→origin micro-segment failed; skipping chute pulse")
                        return success  # don't open chute if failed

                    # 5) Open/close chute
                    await self.actuator.pulse_sequence(
                        open_seconds=self.actuator_open_seconds,
                        close_seconds=self.actuator_close_seconds,
                        rate_hz=self.actuator_rate_hz,
                        settle_before=3.0,
                        settle_between=2.0,
                        wait_for_enter_between=False,
                        enter_prompt="Hole measured. Press ENTER to close the chute...",
                        enter_timeout=30.0,      # safety timeout
                    )

                    # Wait for CAN bus to settle after actuator deployment
                    await asyncio.sleep(1.0)
                finally:
                    # Clear flag when deployment is complete
                    self.actuator_deploying = False

            # Update Flask GUI state AFTER deployment completes (if this was a waypoint segment)
            # NOTE: The waypoint index was already incremented in motion_planner when the track was created
            # We just need to update the GUI to reflect the completed waypoint
            if do_post_actions:
                # motion_planner.current_waypoint_index is now pointing to the NEXT target
                # So the waypoint we just completed is current_waypoint_index - 1
                from utils.navigation_state import mark_waypoint_complete
                completed_wp_idx = self.motion_planner.current_waypoint_index - 1
                mark_waypoint_complete(completed_wp_idx)
                set_navigation_state(current_waypoint_index=self.motion_planner.current_waypoint_index)
                # logger.info(f"[GUI] Updated Flask state: completed waypoint {completed_wp_idx}, now targeting {self.motion_planner.current_waypoint_index}/{len(self.motion_planner.waypoints)}")
        else:
            logger.warning("ERROR: Track segment failed or timed out")

        return success

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

                user_choice: str = self.get_user_choice()

                if user_choice == "quit":
                    logger.info("User requested quit, stopping navigation")
                    self.shutdown_requested = True
                    break

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

                segment_count += 1
                logger.info(
                    f"Executing track segment {segment_count} with {len(track_segment.waypoints)} waypoints"
                )

                # Determine if this is a waypoint segment (should deploy) vs turn/maneuver segment (no deployment)
                is_waypoint_segment = "waypoint" in segment_name.lower() and "row_end" not in segment_name.lower()
                is_approach_segment = "approach" in segment_name.lower()

                # Save track to navigation progress
                self.navigation_progress[segment_name] = track_segment

                # Execute approach segment (stops 2m before search zone)
                if is_approach_segment:
                    logger.info("[APPROACH] Executing approach segment (stopping 2m before search zone)")
                    success = await self.execute_single_track(track_segment, do_post_actions=False)

                    if not success:
                        logger.warning("Approach segment failed, retrying...")
                        # Decrement waypoint index so redo_last_segment() retries the same waypoint
                        self.motion_planner.current_waypoint_index -= 1
                        continue

                    logger.info("[APPROACH] Reached approach waypoint, waiting for collar detection...")

                    # Set flag to allow vision to override waypoint position
                    self.waiting_for_collar_detection = True

                    # Wait for vision to detect collar at approach position
                    should_proceed = await self._wait_for_cone_or_skip()

                    # Clear flag after vision detection or skip
                    self.waiting_for_collar_detection = False

                    if not should_proceed:
                        # Shutdown was requested during wait
                        break

                    # Rebuild track from approach position to actual collar location
                    if getattr(self, "cone_detected_for_current_wp", False):
                        logger.info("[VISION] Collar detected, creating final approach track to collar")
                        final_approach_track = await self.motion_planner.create_approach_to_waypoint_segment()
                    else:
                        logger.info("[VISION] No collar detected, proceeding to CSV waypoint")
                        final_approach_track = await self.motion_planner.create_approach_to_waypoint_segment()

                    # Reset flag for next waypoint
                    self.cone_detected_for_current_wp = False

                    # Execute final approach with deployment
                    final_segment_name = f"final_approach_{segment_name}"
                    self.navigation_progress[final_segment_name] = final_approach_track
                    logger.info("[APPROACH] Executing final approach to collar/waypoint")
                    success = await self.execute_single_track(final_approach_track, do_post_actions=True)

                elif is_waypoint_segment:
                    # Direct waypoint approach (robot starts close to waypoint, skips approach segment)
                    # This should use the same two-stage logic as approach segments to ensure deployment
                    # happens at the correct location after vision refinement

                    logger.info("[DIRECT APPROACH] Executing initial segment (no deployment yet)")
                    success = await self.execute_single_track(track_segment, do_post_actions=False)

                    if not success:
                        logger.warning("Direct approach segment failed, retrying...")
                        continue

                    logger.info("[DIRECT APPROACH] Reached waypoint vicinity, waiting for collar detection...")

                    # Set flag to allow vision to override waypoint position
                    self.waiting_for_collar_detection = True

                    # Wait for vision to detect collar
                    should_proceed = await self._wait_for_cone_or_skip()

                    # Clear flag after vision detection or skip
                    self.waiting_for_collar_detection = False

                    if not should_proceed:
                        # Shutdown was requested during wait
                        break

                    # Rebuild track from current position to actual collar location
                    if getattr(self, "cone_detected_for_current_wp", False):
                        logger.info("[VISION] Collar detected, creating final approach track to collar")
                        final_approach_track = await self.motion_planner.create_approach_to_waypoint_segment()
                    else:
                        logger.info("[VISION] No collar detected, proceeding to CSV waypoint")
                        final_approach_track = await self.motion_planner.create_approach_to_waypoint_segment()

                    # Reset flag for next waypoint
                    self.cone_detected_for_current_wp = False

                    # Execute final approach with deployment
                    final_segment_name = f"final_direct_{segment_name}"
                    self.navigation_progress[final_segment_name] = final_approach_track
                    logger.info("[DIRECT APPROACH] Executing final approach to collar/waypoint with deployment")
                    success = await self.execute_single_track(final_approach_track, do_post_actions=True)
                else:
                    # Turn/maneuver segments - no deployment
                    # Calculate dynamic timeout based on number of waypoints (assume ~1.5s per waypoint)
                    num_waypoints = len(track_segment.waypoints)
                    dynamic_timeout = max(30.0, num_waypoints * 1.5 + 10.0)  # At least 30s, or 1.5s/wp + 10s buffer
                    logger.info(f"Using dynamic timeout of {dynamic_timeout:.1f}s for {num_waypoints} waypoints")
                    success = await self.execute_single_track(track_segment, timeout=dynamic_timeout, do_post_actions=False)

                failed_attempts: int = 0

                while not success:
                    if self.shutdown_requested:
                        break
                    logger.warning(
                        f"Failed to execute segment {segment_count}. Stopping navigation.")
                    failed_attempts += 1

                    # Add exponential backoff delay after failures to allow CAN bus recovery
                    # This is especially important for CANBUS_TIMEOUT failures
                    backoff_delay = min(2.0 ** failed_attempts, 10.0)  # Cap at 10 seconds
                    logger.info(f"Waiting {backoff_delay:.1f}s before retry (attempt {failed_attempts})...")
                    await asyncio.sleep(backoff_delay)

                    if segment_count == 1 and failed_attempts > 5:
                        # await move_robot_forward(time_goal=1.5) #TODO: implement
                        logger.info(
                            f"Moving robot forward | Failed attempts: {failed_attempts}")
                        failed_attempts = 0
                    track_segment, segment_name = await self.motion_planner.redo_last_segment()
                    # Preserve segment type: no deployment for row-end or approach segments
                    is_waypoint_segment = "waypoint" in segment_name.lower() and "row_end" not in segment_name.lower()
                    is_approach_segment = "approach" in segment_name.lower()
                    should_deploy = is_waypoint_segment and not is_approach_segment

                    # Calculate dynamic timeout for maneuver segments
                    if not is_waypoint_segment and not is_approach_segment:
                        num_waypoints = len(track_segment.waypoints)
                        dynamic_timeout = max(30.0, num_waypoints * 1.5 + 10.0)
                        success = await self.execute_single_track(track_segment, timeout=dynamic_timeout, do_post_actions=should_deploy)
                    else:
                        success = await self.execute_single_track(track_segment, do_post_actions=should_deploy)

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
