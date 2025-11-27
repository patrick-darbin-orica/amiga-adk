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
import json
import logging
from enum import Enum
from math import radians, cos, sin, hypot
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple
import pandas as pd

import numpy as np
from farm_ng.core.event_client import EventClient
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng.track.track_pb2 import Track
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Pose3F64
from farm_ng_core_pybind import Rotation3F64
from google.protobuf.empty_pb2 import Empty
from track_planner import TrackBuilder
from utils.navigation_state import set_navigation_state
from utils.pose_cache import set_latest_pose


def _poses_from_csv(csv_path: Path, last_row_waypoint_index: int | None = None) -> dict[int, Pose3F64]:
    """
    Load ENU waypoints from CSV with columns:
      - dx (Easting, meters), dy (Northing, meters)
      - optional: yaw_deg (heading along row, degrees). If omitted, we'll infer from neighbors.

    Args:
        csv_path: Path to the CSV file
        last_row_waypoint_index: Index of the last waypoint in the first row (1-indexed).
                                 This waypoint will use backward difference for heading inference
                                 to align with approach direction, not exit direction.

    Returns a dict of Pose3F64 representing *world_from_hole* in NWU, 1-indexed,
    matching what the JSON Track loader produced.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if not {"dx", "dy"} <= set(df.columns):
        raise RuntimeError(f"{csv_path} must contain columns 'dx' and 'dy'.")

    # ENU -> NWU: north = dy, west = -dx
    north = df["dy"].astype(float).to_numpy()
    west = (-df["dx"].astype(float)).to_numpy() # may require a negative sign

    # Yaw: prefer yaw_deg column; else infer from consecutive points (path tangent).
    if "yaw_deg" in df.columns:
        yaw = np.deg2rad(df["yaw_deg"].astype(float).to_numpy())
    else:
        # Infer yaw at each point using forward difference, last uses backward difference.
        dx_n = np.zeros_like(north)
        dy_w = np.zeros_like(west)
        if len(north) > 1:
            dx_n[:-1] = north[1:] - north[:-1]
            dy_w[:-1] = west[1:] - west[:-1]
            dx_n[-1] = north[-1] - north[-2]
            dy_w[-1] = west[-1] - west[-2]

            # Special handling for last_row_waypoint: use backward difference (approach direction)
            # This ensures the waypoint is oriented toward the direction the robot is traveling
            # when it arrives, not where it's going next (which may involve row-end maneuvers)
            if last_row_waypoint_index is not None and 1 <= last_row_waypoint_index <= len(north):
                # The poses dict uses enumerate(start=1), so CSV row with ID=N becomes waypoint N+1
                # If last_row_waypoint_index=3, that's CSV row ID=2 (3rd row of data), at array index 2
                idx = last_row_waypoint_index - 1  # Convert from 1-indexed waypoint to 0-indexed array
                if idx > 0:  # Can only use backward difference if not the first waypoint
                    dx_n[idx] = north[idx] - north[idx - 1]
                    dy_w[idx] = west[idx] - west[idx - 1]
                    logger.info(f"Using backward difference for waypoint {last_row_waypoint_index} (array index {idx}) heading (last row waypoint)")

        # In NWU, yaw is atan2(Y_west, X_north)
        yaw = np.arctan2(dy_w, dx_n)

    poses: dict[int, Pose3F64] = {}
    zero_tangent = np.zeros((6, 1), dtype=np.float64)
    for i, (n, w, th) in enumerate(zip(north, west, yaw), start=1):
        iso = Isometry3F64(
            np.array([n, w, 0.0], dtype=np.float64), Rotation3F64.Rz(float(th)))
        poses[i] = Pose3F64(a_from_b=iso, frame_a="world",
                            frame_b="hole", tangent_of_b_in_a=zero_tangent)
    return poses

def _offset_towards(start_xy, target_xy, offset_m):
    """
    Returns a point lying on the segment [start -> target] but 'offset_m' short of 'target'.
    If dist(start, target) <= offset_m, returns 'start' (i.e., don't move past/through the target).
    """
    sx, sy = start_xy
    tx, ty = target_xy
    dx, dy = tx - sx, ty - sy
    dist = hypot(dx, dy)
    if dist <= 1e-6:
        return (sx, sy)
    # If we are closer than the offset already, just stay put.
    if dist <= offset_m:
        return (sx, sy)
    scale = (dist - offset_m) / dist
    return (sx + dx * scale, sy + dy * scale)

class FirstManeuver(Enum):
    """Enum to represent the first maneuver type."""

    AB = "ab_segment"
    TURN_THEN_AB = "turn_then_ab_segment"
    LATERAL_CORRECTION = "lateral_correction_segment"
    REPOSITIONING = "repositioning_segment"


logger = logging.getLogger("Motion Planner")


async def get_current_pose(client: EventClient | None = None, timeout: float = 5.0) -> Optional[Pose3F64]:
    """Get the current pose for the track.

    Args:
        client: A EventClient for the required service (filter)
    Returns:
        The current pose (Pose3F64) if available, otherwise None.
    """

    if client is not None:
        try:
            # Get the current state of the filter
            state: FilterState = await asyncio.wait_for(
                client.request_reply("/get_state", Empty(), decode=True), timeout=timeout
            )

            # Update pose cache with filter state for Flask GUI
            pose = Pose3F64.from_proto(state.pose)
            x = float(pose.a_from_b.translation[0])
            y = float(pose.a_from_b.translation[1])
            yaw = float(pose.a_from_b.rotation.log()[-1])
            converged = bool(getattr(state, "has_converged", False))
            set_latest_pose(x, y, yaw, converged)

            return pose
        except asyncio.TimeoutError:
            logger.info(
                "Timeout while getting filter state. Using default start pose.")
        except Exception as e:
            logger.error(
                f"Error getting filter state: {e}. Using default start pose.")

    return None


class MotionPlanner:
    """A class to handle motion planning for the Amiga."""

    def __init__(
        self,
        client: EventClient,
        waypoints_path: Path | str,
        tool_config_path: Path | str,
        last_row_waypoint_index: int,
        turn_direction: str,
        row_spacing: float,
        headland_buffer: float,
        return_to_start: bool = False,
    ):
        self.client = client
        self.waypoints: Dict[int, Pose3F64] = {}
        self.original_csv_waypoints: Dict[int, Pose3F64] = {}  # Store original CSV waypoints for vision search zones
        self.last_row_waypoint_index = last_row_waypoint_index
        self.row_spacing = row_spacing
        self.headland_buffer = headland_buffer
        self.return_to_start = return_to_start
        self.current_waypoint_index = 0
        self.current_pose: Optional[Pose3F64] = None
        self.pose_query_task: asyncio.Task | None = None
        self.should_poll: bool = True
        # Track if we have finished all row end maneuvers (2 segments: headland buffer + π turn)
        self.row_end_segment_index: int = 1
        if turn_direction not in ["left", "right"]:
            raise ValueError("turn_direction must be either 'left' or 'right'")
        self.turn_angle_sign: float = 1.0 if turn_direction == "left" else -1.0

        # Cached turn waypoints for consistent retry behavior (only segment 1 is straight)
        self._turn_waypoint_1: Optional[Pose3F64] = None

        if not isinstance(waypoints_path, Path):
            waypoints_path = Path(waypoints_path)
        try:
            # Load waypoints either from Track JSON or CSV
            if waypoints_path.suffix.lower() == ".csv":
                waypoints_dict = _poses_from_csv(waypoints_path, last_row_waypoint_index=last_row_waypoint_index)
            else:
                track: Track = proto_from_json_file(waypoints_path, Track())
                waypoints_dict = {i: Pose3F64.from_proto(
                    p) for i, p in enumerate(track.waypoints, 1)}

        except Exception as e:
            raise RuntimeError(
                f"Failed to load waypoints from {waypoints_path}: {e}")

        # Load tool offsets
        self.tool_offset = self._load_tool_offset(tool_config_path)

        # Store UNTRANSFORMED waypoints for vision search zone validation
        # (Cones are placed at surveyed waypoint locations, NOT at robot target positions)
        self.original_csv_waypoints = waypoints_dict.copy()

        # Transform hole coordinates to robot coordinates
        self.waypoints = self._transform_holes_to_robot_poses(waypoints_dict.copy())

        # Initialize navigation state for Flask GUI
        set_navigation_state(
            total_waypoints=len(self.waypoints),
            current_waypoint_index=0,
            navigation_running=True
        )

        self.pose_query_task = asyncio.create_task(self._update_current_pose())

    def _load_tool_offset(self, tool_offsets_path: Path) -> Pose3F64:
        """Load tool offset from JSON file, but flip so planner aligns robot origin on waypoint first."""
        with open(tool_offsets_path, 'r') as f:
            offset_data = json.load(f)

        translation = offset_data["translation"]
        logger.info(f"Loaded tool offset from {tool_offsets_path}: x={translation['x']:.3f}m, y={translation['y']:.3f}m, z={translation['z']:.3f}m")

        # Define tool_from_robot instead of robot_from_tool
        robot_from_tool = Pose3F64(
            a_from_b=Isometry3F64(
                translation=[translation["x"], translation["y"], translation["z"]], rotation=Rotation3F64()
            ),
            frame_a="robot",
            frame_b="tool",
        )
        return robot_from_tool

    def _transform_holes_to_robot_poses(self, hole_poses: Dict[int, Pose3F64]) -> Dict[int, Pose3F64]:
        """Transform hole coordinates to robot center coordinates."""
        robot_poses = {}

        for idx, hole_pose in hole_poses.items():
            # The loaded pose represents world_from_hole, but it came in as world_from_robot
            # We need to fix the frame assignment first
            world_from_hole = Pose3F64(
                a_from_b=hole_pose.a_from_b,  # Same transform
                frame_a="world",
                frame_b="hole",  # Change frame_b to "hole"
                tangent_of_b_in_a=hole_pose.tangent_of_b_in_a,
            )

            # Now calculate where robot should be
            # world_from_robot = world_from_hole * hole_from_robot
            hole_from_robot = self.tool_offset.inverse()
            hole_from_robot.frame_a = "hole"  # Make sure frames match
            hole_from_robot.frame_b = "robot"

            world_from_robot = world_from_hole * hole_from_robot
            robot_poses[idx] = world_from_robot

        return robot_poses

    async def _update_current_pose(self):
        """Update the current pose from the filter."""
        if self.client is None:
            raise RuntimeError("EventClient cannot be None")

        while self.should_poll:
            try:
                maybe_current_pose = await get_current_pose(self.client)
                if maybe_current_pose is not None:
                    self.current_pose = maybe_current_pose
                else:
                    logger.warning(
                        "Current pose is None, ensure your filter is running.")
            except Exception as e:
                logger.error(f"Error updating current pose: {e}")
                return None

            await asyncio.sleep(0.1)  # Poll at 10 Hz

    async def _get_current_pose(self) -> Pose3F64:
        """Get the current pose of the Amiga.

        NOTE: This will block until the pose is available.
        Returns:
            The current pose (Pose3F64)
        """
        current_pose = None
        while current_pose is None:
            current_pose = self.current_pose  # should be updated by the background task
            await asyncio.sleep(0.5)  # Wait for the pose to be updated

        return current_pose

    async def create_tool_to_origin_segment(self) -> Track:
        """Micro-move after dipper deployment. Advances 0.20m forward to position chute."""
        # Fixed advance distance after dipper deployment (not the full tool offset)
        advance_m = 0.25
        current = await self._get_current_pose()
        track_builder = TrackBuilder(start=current)
        track_builder.create_straight_segment(next_frame_b="tool_to_origin", distance=advance_m, spacing=0.05)
        logger.info(f"Creating post-dipper segment: advancing {advance_m:.3f}m")
        return track_builder.track

    async def override_next_waypoint_world_xy(self, X_w: float, Y_w: float, yaw_rad: float | None = None) -> int:
        """
        Replace the current target waypoint with a world pose at (X_w, Y_w). Heading defaults to current robot yaw.
        Returns the waypoint index that was modified.

        NOTE: (X_w, Y_w) represents the HOLE/COLLAR position. This method applies the tool offset
        transformation to calculate where the robot center should be, just like CSV waypoint loading does.
        """
        # Determine which waypoint to replace (the current target)
        idx = max(1, self.current_waypoint_index)

        # Use current heading if none provided
        if yaw_rad is None:
            pose_now = await self._get_current_pose()
            yaw_rad = float(pose_now.a_from_b.rotation.log()[-1])

        # Build world_from_hole pose (vision detects collar/hole position)
        iso_hole = Isometry3F64([float(X_w), float(Y_w), 0.0], Rotation3F64.Rz(float(yaw_rad)))
        world_from_hole = Pose3F64(iso_hole, frame_a="world", frame_b="hole")

        # Transform hole position to robot position using tool offset
        # world_from_robot = world_from_hole * hole_from_robot
        hole_from_robot = self.tool_offset.inverse()
        hole_from_robot.frame_a = "hole"
        hole_from_robot.frame_b = "robot"

        world_from_robot = world_from_hole * hole_from_robot

        logger.info(f"[VISION] Override waypoint {idx}: hole @ ({X_w:.3f}, {Y_w:.3f}) → robot @ ({world_from_robot.a_from_b.translation[0]:.3f}, {world_from_robot.a_from_b.translation[1]:.3f}) with tool offset {self.tool_offset.a_from_b.translation[0]:.3f}m")

        self.waypoints[idx] = world_from_robot
        return idx

    def _angle_difference(self, from_angle: float, to_angle: float) -> float:
        """Calculate the shortest angular difference between two angles."""
        diff = to_angle - from_angle
        # Wrap to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    async def _analyze_approach_scenario(self) -> dict:
        """Analyze the current robot state relative to the first goal."""

        current_pose = await self._get_current_pose()
        goal_pose = self.waypoints.get(1)

        if goal_pose is None:
            raise RuntimeError(
                "First waypoint (index 1) not found in waypoints")

        # Transform goal to robot frame to get relative position
        robot_from_goal = current_pose.inverse() * goal_pose
        goal_in_robot_frame = robot_from_goal.log()

        delta_x = goal_in_robot_frame[0]  # Forward/backward (North)
        delta_y = goal_in_robot_frame[1]  # Left/right (West)
        delta_heading = goal_in_robot_frame[-1]  # Yaw difference

        # Calculate bearing angle - how far off from "straight behind" we are
        bearing_angle = abs(np.arctan2(abs(delta_y), abs(
            delta_x))) if delta_x != 0 else np.pi / 2

        return {
            'delta_x': delta_x,
            'delta_y': delta_y,
            'delta_heading': delta_heading,
            'bearing_angle': bearing_angle,
            'bearing_degrees': np.degrees(bearing_angle),
            'longitudinal_distance': abs(delta_x),
            'lateral_distance': abs(delta_y),
            'heading_error': abs(delta_heading),
            'is_behind_goal': delta_x > 0,
        }

    async def _determine_first_maneuver(self) -> FirstManeuver:
        """Determine first maneuver strategy based on bearing and heading."""

        analysis = await self._analyze_approach_scenario()

        # Thresholds
        # 20 degrees | relatively small delta y compared to delta x
        BEARING_THRESHOLD = np.radians(20)
        # 10 degrees | relatively small heading error
        HEADING_THRESHOLD = np.radians(10)
        MIN_LONGITUDINAL_DISTANCE = (
            1.8  # 2 meters | ensure we're at least 2 m behind the goal to ensure a smooth arrival
        )

        bearing = analysis['bearing_angle']
        heading_error = analysis['heading_error']
        is_behind = analysis['is_behind_goal']
        longitudinal = analysis['longitudinal_distance']

        # First check if the robot needs to be repositioned
        if not is_behind and longitudinal < MIN_LONGITUDINAL_DISTANCE:
            # If we're not behind the goal and too close, we need to reposition
            return FirstManeuver.REPOSITIONING

        # Good bearing (roughly behind the goal)
        if bearing < BEARING_THRESHOLD:
            # Good heading --> Go straight to the next waypoint
            if heading_error <= HEADING_THRESHOLD:
                return FirstManeuver.AB
            else:  # Heading is bad, let's align the robot first and then send it
                return FirstManeuver.TURN_THEN_AB
        # Bad bearing (too much lateral offset)
        else:
            return FirstManeuver.LATERAL_CORRECTION

    async def build_track_to_robot_relative_goal(
        self, x_fwd_m: float, y_left_m: float, standoff_m: float = 0.5, spacing: float = 0.1
        ):
        """Convert (x_fwd,y_left) in robot frame into a world pose and build a short AB track."""
        current_pose = await self._get_current_pose()  # uses your running filter task 
        yaw = current_pose.a_from_b.rotation.log()[-1]

        # standoff along the ray
        dist = hypot(x_fwd_m, y_left_m)
        if dist > standoff_m:
            k = (dist - standoff_m) / dist
            x_fwd_m *= k; y_left_m *= k
        else:
            x_fwd_m *= 0.9; y_left_m *= 0.9

        c, s = cos(yaw), sin(yaw)
        dx_w =  x_fwd_m*c - y_left_m*s
        dy_w =  x_fwd_m*s + y_left_m*c

        goal_t = current_pose.a_from_b.translation.copy()
        goal_t[0] += dx_w; goal_t[1] += dy_w

        # keep heading same (bearing‑agnostic arrival)
        goal = Pose3F64(Isometry3F64(goal_t, Rotation3F64.Rz(yaw)), frame_a="world", frame_b="vision_goal")
        
        tb = TrackBuilder(start=current_pose)
        tb.create_ab_segment(next_frame_b="vision_goal", final_pose=goal, spacing=spacing)  # 
        return tb.track, goal
    
    async def _create_lateral_correction(self) -> Track:
        "Drive robot perpendicular to correct lateral offset, then approach goal."

        analysis = await self._analyze_approach_scenario()

        goal_pose = self.waypoints.get(1)

        if goal_pose is None:
            raise RuntimeError(
                "First waypoint (index 1) not found in waypoints")

        # Get current and goal headings in world frame
        current_pose = await self._get_current_pose()
        current_heading = current_pose.a_from_b.rotation.log()[-1]
        goal_heading = goal_pose.a_from_b.rotation.log()[-1]

        # Calculate perpendicular direction to the goal heading
        # If goal is pointing North (0°), perpendicular could be East (90°) or West (-90°)
        # We choose based on which side the goal is on
        # Which side is goal on?
        goal_direction_sign = 1 if analysis['delta_y'] > 0 else -1
        perpendicular_heading = goal_heading + \
            (np.pi / 2) * goal_direction_sign

        turn_to_perpendicular = self._angle_difference(
            current_heading, perpendicular_heading)

        # Create the track
        track_builder = TrackBuilder(start=current_pose)

        # Step 1: Turn to face perpendicular to goal
        track_builder.create_turn_segment(
            next_frame_b="facing_goal_laterally", angle=turn_to_perpendicular, spacing=0.05
        )

        # Step 2: Drive towards the goal until we're close laterally
        lateral_correction_distance = analysis['lateral_distance']
        track_builder.create_straight_segment(
            next_frame_b="laterally_aligned", distance=lateral_correction_distance, spacing=0.1
        )

        # Step 3: Turn to align with the goal heading
        turn_to_goal_heading = self._angle_difference(
            perpendicular_heading, goal_heading)
        track_builder.create_turn_segment(
            next_frame_b="aligned_to_goal_heading", angle=turn_to_goal_heading, spacing=0.05
        )

        # Step 4: Drive straight to goal - check if we should use approach offset
        # Calculate distance from laterally aligned position to goal
        longitudinal_distance = analysis['longitudinal_distance']

        APPROACH_OFFSET = 1.5
        if longitudinal_distance > APPROACH_OFFSET + 0.5:  # Add 0.5m buffer
            # Robot is far - use approach waypoint
            current_x = current_pose.a_from_b.translation[0]
            current_y = current_pose.a_from_b.translation[1]
            goal_x = goal_pose.a_from_b.translation[0]
            goal_y = goal_pose.a_from_b.translation[1]

            approach_x, approach_y = _offset_towards(
                (current_x, current_y),
                (goal_x, goal_y),
                APPROACH_OFFSET
            )
            approach_iso = Isometry3F64(
                np.array([approach_x, approach_y, 0.0], dtype=np.float64),
                goal_pose.a_from_b.rotation
            )
            approach_pose = Pose3F64(
                a_from_b=approach_iso,
                frame_a="world",
                frame_b="approach_1"
            )
            track_builder.create_ab_segment(
                next_frame_b="approach_1", final_pose=approach_pose, spacing=0.5)
        else:
            # Robot is close - go directly to waypoint
            track_builder.create_ab_segment(
                next_frame_b="waypoint_1", final_pose=goal_pose, spacing=0.5)

        self.current_waypoint_index += 1

        return track_builder.track

    async def _create_turn_and_ab(self) -> Track:
        """Create a track consisting of a turn in place and an AB segment."""
        # First calculate how much we need to turn to align to the goal
        current_pose = await self._get_current_pose()
        goal_pose = self.waypoints.get(1)

        if goal_pose is None:
            raise RuntimeError(
                "First waypoint (index 1) not found in waypoints")

        turn_angle = self._angle_difference(
            current_pose.a_from_b.rotation.log(
            )[-1], goal_pose.a_from_b.rotation.log()[-1]
        )
        track_builder = TrackBuilder(start=current_pose)
        track_builder.create_turn_segment(
            next_frame_b="aligned_to_goal", angle=turn_angle, spacing=0.05)

        # Check distance to determine if we should use approach offset
        current_x = current_pose.a_from_b.translation[0]
        current_y = current_pose.a_from_b.translation[1]
        goal_x = goal_pose.a_from_b.translation[0]
        goal_y = goal_pose.a_from_b.translation[1]
        distance_to_waypoint = hypot(goal_x - current_x, goal_y - current_y)

        APPROACH_OFFSET = 1.2
        if distance_to_waypoint > APPROACH_OFFSET + 0.5:  # Add 0.5m buffer
            # Robot is far - use approach waypoint
            approach_x, approach_y = _offset_towards(
                (current_x, current_y),
                (goal_x, goal_y),
                APPROACH_OFFSET
            )
            approach_iso = Isometry3F64(
                np.array([approach_x, approach_y, 0.0], dtype=np.float64),
                goal_pose.a_from_b.rotation  # Use goal heading after turn
            )
            approach_pose = Pose3F64(
                a_from_b=approach_iso,
                frame_a="world",
                frame_b="approach_1"
            )
            track_builder.create_ab_segment(
                next_frame_b="approach_1", final_pose=approach_pose, spacing=0.5)
        else:
            # Robot is close - go directly to waypoint
            track_builder.create_ab_segment(
                next_frame_b="waypoint_1", final_pose=goal_pose, spacing=0.5)

        self.current_waypoint_index += 1

        return track_builder.track

    async def _create_ab_segment_to_next_waypoint(self, approach_offset_m: float = 0.0) -> Track:
        """Create an AB segment to the next waypoint, optionally with approach offset.

        Args:
            approach_offset_m: Distance to stop before the waypoint (0.0 = go to waypoint directly)

        Returns:
            The track segment to the next waypoint (Track)
        """
        # 1. Ensure we have the current pose
        current_pose = await self._get_current_pose()

        # 2. Create the track (AB) segment to the next waypoint
        track_builder = TrackBuilder(start=current_pose)
        self.current_waypoint_index += 1

        target_pose = self.waypoints[self.current_waypoint_index]

        # If approach_offset_m is specified, create intermediate approach waypoint
        if approach_offset_m > 0.0:
            # Calculate position 2m before the target waypoint
            current_x = current_pose.a_from_b.translation[0]
            current_y = current_pose.a_from_b.translation[1]
            target_x = target_pose.a_from_b.translation[0]
            target_y = target_pose.a_from_b.translation[1]

            approach_x, approach_y = _offset_towards(
                (current_x, current_y),
                (target_x, target_y),
                approach_offset_m
            )

            # Create approach pose with current heading (don't force rotation)
            # This prevents premature turns when waypoints have opposite directions
            current_heading = current_pose.a_from_b.rotation
            approach_iso = Isometry3F64(
                np.array([approach_x, approach_y, 0.0], dtype=np.float64),
                current_heading
            )
            approach_pose = Pose3F64(
                a_from_b=approach_iso,
                frame_a="world",
                frame_b=f"approach_{self.current_waypoint_index}"
            )

            # Build track to approach waypoint (not full waypoint yet)
            track_builder.create_ab_segment(
                next_frame_b=f"approach_{self.current_waypoint_index}",
                final_pose=approach_pose,
                spacing=0.5,
            )
        else:
            # Standard behavior - go directly to waypoint
            track_builder.create_ab_segment(
                next_frame_b=f"waypoint_{self.current_waypoint_index}",
                final_pose=target_pose,
                spacing=0.5,
            )

        return track_builder.track

    async def wait_for_vision_buffer(self, waypoint_idx: int, timeout_s: float = 0.1, check_interval_s: float = 0.2) -> bool:
        """Wait for vision buffer to populate with sufficient detections for planning.

        This look-ahead pause allows the vision system to detect the next collar
        before we plan the approach direction.

        Args:
            waypoint_idx: The waypoint index to wait for detections
            timeout_s: Maximum time to wait (default 3.0 seconds)
            check_interval_s: How often to check buffer (default 0.2 seconds)

        Returns:
            True if sufficient detections were collected, False if timeout
        """
        MIN_CONFIDENCE = 0.7
        MIN_DETECTIONS = 1

        logger.info(f"[VISION LOOKAHEAD] Waiting up to {timeout_s}s for collar detections for waypoint {waypoint_idx}...")

        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed >= timeout_s:
                logger.info(f"[VISION LOOKAHEAD] Timeout after {timeout_s}s - proceeding with CSV waypoint")
                return False

            # Check if we have sufficient detections
            if hasattr(self, "vision_detection_buffer"):
                detections = self.vision_detection_buffer.get(waypoint_idx, [])
                high_conf_detections = [
                    det for det in detections
                    if det[3] >= MIN_CONFIDENCE
                ]

                if len(high_conf_detections) >= MIN_DETECTIONS:
                    logger.info(f"[VISION LOOKAHEAD] Got {len(high_conf_detections)} detections after {elapsed:.1f}s")
                    return True

            # Wait before checking again
            await asyncio.sleep(check_interval_s)

    def get_best_collar_estimate_from_vision(self, waypoint_idx: int) -> Optional[Tuple[float, float]]:
        """Get the best collar position estimate from vision detection buffer.

        This is called BEFORE creating the approach segment to aim directly at the detected collar.
        Uses median filtering of recent detections to reduce noise.

        Args:
            waypoint_idx: The waypoint index to get collar estimate for

        Returns:
            Tuple of (x_world, y_world) if reliable detection exists, None otherwise
        """
        if not hasattr(self, "vision_detection_buffer"):
            return None

        detections = self.vision_detection_buffer.get(waypoint_idx, [])

        if len(detections) == 0:
            logger.info(f"[VISION PLANNING] No buffered detections for waypoint {waypoint_idx}")
            return None

        # Filter for high-confidence detections
        MIN_CONFIDENCE = 0.7
        MIN_DETECTIONS = 2  # Require at least 2 detections for reliability

        high_conf_detections = [
            det for det in detections
            if det[3] >= MIN_CONFIDENCE  # det[3] is confidence
        ]

        if len(high_conf_detections) < MIN_DETECTIONS:
            logger.info(f"[VISION PLANNING] Insufficient high-confidence detections for waypoint {waypoint_idx}: "
                       f"{len(high_conf_detections)}/{len(detections)} (need {MIN_DETECTIONS})")
            return None

        # Use median position of recent high-confidence detections (robust to outliers)
        x_positions = [det[1] for det in high_conf_detections]  # det[1] is x_world
        y_positions = [det[2] for det in high_conf_detections]  # det[2] is y_world

        median_x = float(np.median(x_positions))
        median_y = float(np.median(y_positions))

        logger.info(f"[VISION PLANNING] Using median collar estimate for waypoint {waypoint_idx}: "
                   f"({median_x:.2f}, {median_y:.2f}) from {len(high_conf_detections)} detections")

        return (median_x, median_y)

    async def create_approach_to_waypoint_segment(self) -> Track:
        """Create segment from current approach position to the actual waypoint.

        This is called after stopping at the approach waypoint and detecting the collar.
        Returns a track from current position to the waypoint (which may have been overridden by vision).
        """
        current_pose = await self._get_current_pose()
        target_pose = self.waypoints[self.current_waypoint_index]

        track_builder = TrackBuilder(start=current_pose)
        track_builder.create_ab_segment(
            next_frame_b=f"waypoint_{self.current_waypoint_index}",
            final_pose=target_pose,
            spacing=0.5,
        )
        return track_builder.track

    async def _row_end_maneuver(self, index: int) -> Track:
        """Create a row end maneuver segment based on the index.

        Two-segment row-end maneuver using π turn:
        1. Drive forward into headland buffer
        2. π (180°) turn to next row

        Args:
            index: The index of the row end maneuver (1 to 2)
                1: Drive forward into headland buffer
                2: π turn to next row (replaces the old 3-segment turn sequence)
        Returns:
            The track segment for the row end maneuver (Track)
        """
        if index < 1 or index > 2:
            raise ValueError("index must be between 1 and 2")

        track_segment: Track
        next_frame_b = f"row_end_{index}"

        if index == 1:
            # Segment 1: Drive forward into headland buffer
            # Cache target waypoint on first attempt, reuse on retry
            if not hasattr(self, '_turn_waypoint_1') or self._turn_waypoint_1 is None:
                current_pose = await self._get_current_pose()
                self._turn_waypoint_1 = self._compute_waypoint_ahead(
                    current_pose, distance=self.headland_buffer
                )
                logger.info(f"[TURN] Created virtual waypoint 1 at {self.headland_buffer}m ahead")

            # Build track from current position to cached waypoint
            current_pose = await self._get_current_pose()
            track_builder = TrackBuilder(start=current_pose)
            track_builder.create_straight_segment(
                next_frame_b=next_frame_b,
                distance=self.headland_buffer,
                spacing=0.5
            )
            track_segment = track_builder.track

        else:  # index == 2
            # Segment 2: π turn to next row
            # This replaces the old 3-segment sequence (turn 90° → drive across → turn 90°)
            current_pose = await self._get_current_pose()
            track_builder = TrackBuilder(start=current_pose)
            track_builder.create_arc_segment(
                next_frame_b=next_frame_b,
                radius=self.row_spacing / 2,  # Turn radius fits exactly to next row
                angle=radians(180 * self.turn_angle_sign),  # π turn
                spacing=0.15
            )
            track_segment = track_builder.track
            logger.info(f"[TURN] Created π turn with radius {self.row_spacing / 2}m")

        return track_segment

    def _compute_waypoint_ahead(self, current_pose: Pose3F64, distance: float) -> Pose3F64:
        """Compute a virtual waypoint at specified distance ahead of current pose.

        Args:
            current_pose: Current robot pose
            distance: Distance ahead to place waypoint (meters)

        Returns:
            Virtual waypoint pose at specified distance ahead
        """
        # Get current heading from pose
        current_iso = current_pose.a_from_b

        # Create forward translation in robot frame
        forward_offset = Isometry3F64(
            translation=[distance, 0, 0],  # Forward is +X in robot frame
            rotation=Rotation3F64()  # No rotation
        )

        # Compute target pose: current_pose * forward_offset
        target_iso = current_iso * forward_offset

        # Create target waypoint with same frame names
        target_waypoint = Pose3F64(
            a_from_b=target_iso,
            frame_a=current_pose.frame_a,
            frame_b=f"{current_pose.frame_b}_virtual"
        )

        return target_waypoint

    def _clear_turn_waypoints(self):
        """Clear cached turn waypoints when starting new turn sequence."""
        self._turn_waypoint_1 = None
        # Note: _turn_waypoint_3 no longer exists with π turn implementation

    async def _shutdown(self):
        """Shutdown the motion planner."""
        if self.pose_query_task is not None:
            self.should_poll = False
            await self.pose_query_task
            self.pose_query_task = None

    async def redo_last_segment(self) -> Tuple[Optional[Track], Optional[str]]:
        """Redo the last segment.

        Returns:
            The last track segment (Track) and its name.
        """
        if self.current_waypoint_index == 0 or self.current_waypoint_index is None:
            logger.info("No previous segment to redo.")
            return (None, None)

        # Check if we're completing a row end maneuver
        if self.current_waypoint_index == self.last_row_waypoint_index:
            # We're in the row-end maneuver sequence
            if self.row_end_segment_index == 1:
                # We are about to switch to the next row, but we haven't started the row end maneuvers yet.
                # So we just reset our index and let the motion planner handle the next segment.
                self.current_waypoint_index -= 1
            else:
                # We are already in the row end maneuvers and we need to redo the last segment.
                # Decrement row_end_segment_index to re-execute the previous row-end maneuver
                self.row_end_segment_index -= 1
                logger.info(f"Redoing row end maneuver, decremented index to {self.row_end_segment_index}")
        else:  # We're not trying to switch rows, just redo the last AB segment
            self.current_waypoint_index -= 1

        return await self.next_track_segment()

    async def next_track_segment(self) -> Tuple[Optional[Track], Optional[str]]:
        """Get the next track segment to navigate to.

        Returns:
            The next track segment (Track)
        """
        if self.current_waypoint_index >= len(self.waypoints):
            # Check if return-to-start is enabled and we haven't performed the return yet
            if self.return_to_start and self.row_end_segment_index == 1:
                # Trigger return-to-start sequence by performing row-end maneuver
                logger.info("Reached last waypoint. Performing row-end sequence to return to start...")
                # We'll now execute row-end segments 1-2 (headland buffer + π turn)
                current_index = self.row_end_segment_index
                track_segment = await self._row_end_maneuver(current_index)
                self.row_end_segment_index += 1
                return (track_segment, f"return_to_start_{current_index}")
            elif self.return_to_start and self.row_end_segment_index > 1 and self.row_end_segment_index < 3:
                # Continue executing return-to-start row-end segments
                current_index = self.row_end_segment_index
                track_segment = await self._row_end_maneuver(current_index)
                self.row_end_segment_index += 1
                return (track_segment, f"return_to_start_{current_index}")
            else:
                # No return-to-start or already completed return sequence
                logger.info("No more waypoints to navigate to.")
                # Reset row_end_segment_index in case we want to restart
                self.row_end_segment_index = 1
                asyncio.create_task(self._shutdown())
                return (None, None)

        # Check if this is the very first maneuver:
        if self.current_waypoint_index == 0:
            seg_name: Optional[str] = "waypoint_0_to_1"
            track: Optional[Track] = None
            maneuver_type: FirstManeuver = await self._determine_first_maneuver()
            if maneuver_type == FirstManeuver.AB:
                # NEW: Wait for vision buffer to populate, then check for collar detection
                next_wp_idx = 1
                await self.wait_for_vision_buffer(next_wp_idx, timeout_s=0.5)

                collar_estimate = self.get_best_collar_estimate_from_vision(next_wp_idx)
                if collar_estimate is not None:
                    collar_x, collar_y = collar_estimate
                    # Override waypoint with detected collar position
                    await self.override_next_waypoint_world_xy(collar_x, collar_y, yaw_rad=None)
                    logger.info(f"[VISION PLANNING] Overriding waypoint {next_wp_idx} with buffered collar detection before approach")

                # Check distance to determine if we should use approach offset
                current_pose = await self._get_current_pose()
                goal_pose = self.waypoints.get(1)
                if goal_pose is not None:
                    current_x = current_pose.a_from_b.translation[0]
                    current_y = current_pose.a_from_b.translation[1]
                    goal_x = goal_pose.a_from_b.translation[0]
                    goal_y = goal_pose.a_from_b.translation[1]
                    distance_to_waypoint = hypot(goal_x - current_x, goal_y - current_y)

                    APPROACH_OFFSET = 1.2
                    if distance_to_waypoint > APPROACH_OFFSET + 0.5:  # Add 0.5m buffer
                        # Robot is far - use approach offset
                        track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=APPROACH_OFFSET)
                        seg_name = "approach_waypoint_0_to_1"
                    else:
                        # Robot is close - go directly
                        track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=0.0)
                else:
                    track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=0.0)
            elif maneuver_type == FirstManeuver.REPOSITIONING:
                logger.error("Robot is not behind goal. Reposition it first")
                seg_name = None
            elif maneuver_type == FirstManeuver.TURN_THEN_AB:
                track = await self._create_turn_and_ab()
            elif maneuver_type == FirstManeuver.LATERAL_CORRECTION:
                track = await self._create_lateral_correction()
            else:
                logger.error(f"Unknown maneuver type: {maneuver_type}")
                seg_name = None

            return (track, seg_name)

        # Check if we're switching to the next row or just moving to the next waypoint
        if self.current_waypoint_index != self.last_row_waypoint_index:
            # We're not transitioning to a new row, we will just create an AB segment to the next waypoint
            curr_index = self.current_waypoint_index

            # NEW: Wait for vision buffer to populate, then check for collar detection
            next_waypoint_idx = self.current_waypoint_index + 1
            await self.wait_for_vision_buffer(next_waypoint_idx, timeout_s=0.5)

            collar_estimate = self.get_best_collar_estimate_from_vision(next_waypoint_idx)
            if collar_estimate is not None:
                collar_x, collar_y = collar_estimate
                # Override waypoint with detected collar position
                await self.override_next_waypoint_world_xy(collar_x, collar_y, yaw_rad=None)
                logger.info(f"[VISION PLANNING] Overriding waypoint {next_waypoint_idx} with buffered collar detection before approach")

            # Check if robot is already close to the next waypoint
            current_pose = await self._get_current_pose()
            if next_waypoint_idx in self.waypoints:
                target_pose = self.waypoints[next_waypoint_idx]
                current_x = current_pose.a_from_b.translation[0]
                current_y = current_pose.a_from_b.translation[1]
                target_x = target_pose.a_from_b.translation[0]
                target_y = target_pose.a_from_b.translation[1]
                distance_to_waypoint = hypot(target_x - current_x, target_y - current_y)

                # Only use two-stage approach if robot is far enough away
                APPROACH_OFFSET = 1.2
                if distance_to_waypoint > APPROACH_OFFSET + 0.5:  # Add 0.5m buffer
                    # Robot is far - use two-stage approach
                    track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=APPROACH_OFFSET)
                    next_index = self.current_waypoint_index
                    seg_name = f"approach_waypoint_{curr_index}_to_{next_index}"
                else:
                    # Robot is already close - direct approach
                    track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=0.0)
                    next_index = self.current_waypoint_index
                    seg_name = f"waypoint_{curr_index}_to_{next_index}"
            else:
                # Fallback: use direct approach
                track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=0.0)
                next_index = self.current_waypoint_index
                seg_name = f"waypoint_{curr_index}_to_{next_index}"

            return (track, seg_name)

        # We're switching to the next row
        # 1. Ensure row_end_segment_index is valid (in case of previous failures)
        if self.row_end_segment_index < 1 or self.row_end_segment_index > 3:
            logger.warning(f"[DEBUG] row_end_segment_index={self.row_end_segment_index} is invalid, resetting to 1")
            self.row_end_segment_index = 1

        logger.info(f"[DEBUG] At row-end check: row_end_segment_index={self.row_end_segment_index}, current_waypoint_index={self.current_waypoint_index}")

        # 2. Check if we have finished all row end maneuvers (now only 2 segments with π turn)
        if self.row_end_segment_index >= 3:
            logger.info("Finished all row end maneuvers, moving to the next row.")
            # Clear cached turn waypoints for next turn sequence
            self._clear_turn_waypoints()
            # Reset row_end_segment_index for the next row-end sequence
            self.row_end_segment_index = 1
            curr_index = self.current_waypoint_index

            # NEW: Wait for vision buffer to populate, then check for collar detection
            next_waypoint_idx = self.current_waypoint_index + 1
            await self.wait_for_vision_buffer(next_waypoint_idx, timeout_s=0.5)

            collar_estimate = self.get_best_collar_estimate_from_vision(next_waypoint_idx)
            if collar_estimate is not None:
                collar_x, collar_y = collar_estimate
                # Override waypoint with detected collar position
                await self.override_next_waypoint_world_xy(collar_x, collar_y, yaw_rad=None)
                logger.info(f"[VISION PLANNING] Overriding waypoint {next_waypoint_idx} with buffered collar detection before approach")

            # Check if robot is already close to the first waypoint of next row
            current_pose = await self._get_current_pose()
            if next_waypoint_idx in self.waypoints:
                target_pose = self.waypoints[next_waypoint_idx]
                current_x = current_pose.a_from_b.translation[0]
                current_y = current_pose.a_from_b.translation[1]
                target_x = target_pose.a_from_b.translation[0]
                target_y = target_pose.a_from_b.translation[1]
                distance_to_waypoint = hypot(target_x - current_x, target_y - current_y)

                # Only use two-stage approach if robot is far enough away
                APPROACH_OFFSET = 1.2
                if distance_to_waypoint > APPROACH_OFFSET + 0.5:  # Add 0.5m buffer
                    # Robot is far - use two-stage approach
                    track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=APPROACH_OFFSET)
                    next_index = self.current_waypoint_index
                    seg_name = f"approach_waypoint_{curr_index}_to_{next_index}"
                else:
                    # Robot is already close - direct approach
                    track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=0.0)
                    next_index = self.current_waypoint_index
                    seg_name = f"waypoint_{curr_index}_to_{next_index}"
            else:
                # Fallback: use direct approach
                track = await self._create_ab_segment_to_next_waypoint(approach_offset_m=0.0)
                next_index = self.current_waypoint_index
                seg_name = f"waypoint_{curr_index}_to_{next_index}"

            return (track, seg_name)
        else:
            # We need to return a segment from the row end maneuver
            current_index = self.row_end_segment_index
            track_segment = await self._row_end_maneuver(current_index)
            self.row_end_segment_index += 1
            return (track_segment, f"row_end_{current_index}")
