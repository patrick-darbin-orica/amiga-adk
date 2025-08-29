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


def _poses_from_csv(csv_path: Path) -> dict[int, Pose3F64]:
    """
    Load ENU waypoints from CSV with columns:
      - dx (Easting, meters), dy (Northing, meters)
      - optional: yaw_deg (heading along row, degrees). If omitted, we'll infer from neighbors.

    Returns a dict of Pose3F64 representing *world_from_hole* in NWU, 1-indexed,
    matching what the JSON Track loader produced.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if not {"dx", "dy"} <= set(df.columns):
        raise RuntimeError(f"{csv_path} must contain columns 'dx' and 'dy'.")

    # ENU -> NWU: north = dy, west = -dx
    north = df["dy"].astype(float).to_numpy()
    west = (-df["dx"].astype(float)).to_numpy()

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
            return Pose3F64.from_proto(state.pose)
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
    ):
        self.client = client
        self.waypoints: Dict[int, Pose3F64] = {}
        self.last_row_waypoint_index = last_row_waypoint_index
        self.row_spacing = row_spacing
        self.headland_buffer = headland_buffer
        self.current_waypoint_index = 0
        self.current_pose: Optional[Pose3F64] = None
        self.pose_query_task: asyncio.Task | None = None
        self.should_poll: bool = True
        # Track if we have finished all row end maneuvers (total of 5)
        self.row_end_segment_index: int = 1
        if turn_direction not in ["left", "right"]:
            raise ValueError("turn_direction must be either 'left' or 'right'")
        self.turn_angle_sign: float = 1.0 if turn_direction == "left" else -1.0

        if not isinstance(waypoints_path, Path):
            waypoints_path = Path(waypoints_path)
        try:
            # Load waypoints either from Track JSON or CSV
            if waypoints_path.suffix.lower() == ".csv":
                waypoints_dict = _poses_from_csv(waypoints_path)
            else:
                track: Track = proto_from_json_file(waypoints_path, Track())
                waypoints_dict = {i: Pose3F64.from_proto(
                    p) for i, p in enumerate(track.waypoints, 1)}

        except Exception as e:
            raise RuntimeError(
                f"Failed to load waypoints from {waypoints_path}: {e}")

        # Load tool offsets
        self.tool_offset = self._load_tool_offset(tool_config_path)

        # Transform hole coordinates to robot coordinates
        self.waypoints = self._transform_holes_to_robot_poses(waypoints_dict)

        self.pose_query_task = asyncio.create_task(self._update_current_pose())

    def _load_tool_offset(self, tool_offsets_path: Path) -> Pose3F64:
        """Load tool offset from JSON file, but flip so planner aligns robot origin on waypoint first."""
        with open(tool_offsets_path, 'r') as f:
            offset_data = json.load(f)

        translation = offset_data["translation"]

        # Define tool_from_robot instead of robot_from_tool
        tool_from_robot = Pose3F64(
            a_from_b=Isometry3F64(
                translation=[translation["x"], translation["y"], translation["z"]],
                rotation=Rotation3F64()
            ),
            frame_a="tool",
            frame_b="robot",
        )
        return tool_from_robot


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
        """Micro-move: plumbob→chute/origin. Here we only need +0.22 m forward."""
        advance_m = 0.22  # your measured value
        current = await self._get_current_pose()
        tb = TrackBuilder(start=current)
        tb.create_straight_segment(next_frame_b="tool_to_origin", distance=advance_m, spacing=0.05)
        return tb.track


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
            2.0  # 2 meters | ensure we're at least 2 m behind the goal to ensure a smooth arrival
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
        self, x_fwd_m: float, y_left_m: float, standoff_m: float = 0.75, spacing: float = 0.1
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

        # Step 4: Drive straight to goal
        track_builder.create_ab_segment(
            next_frame_b="waypoint_1", final_pose=goal_pose, spacing=0.1)

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
        track_builder.create_ab_segment(
            next_frame_b="waypoint_1", final_pose=goal_pose, spacing=0.1)

        self.current_waypoint_index += 1

        return track_builder.track

    async def _create_ab_segment_to_next_waypoint(self) -> Track:
        """Create an AB segment to the next waypoint.

        Returns:
            The track segment to the next waypoint (Track)
        """
        # 1. Ensure we have the current pose
        current_pose = await self._get_current_pose()

        # 2. Create the track (AB) segment to the next waypoint
        track_builder = TrackBuilder(start=current_pose)
        self.current_waypoint_index += 1
        track_builder.create_ab_segment(
            next_frame_b=f"waypoint_{self.current_waypoint_index}",
            final_pose=self.waypoints[self.current_waypoint_index],
            spacing=0.1,
        )
        return track_builder.track

    async def _row_end_maneuver(self, index: int) -> Track:
        """Create a row end maneuver segment based on the index.

        Args:
            index: The index of the row end maneuver (1 to 5)
        Returns:
            The track segment for the row end maneuver (Track)
        """
        if index < 1 or index > 4:
            raise ValueError("index must be between 1 and 4")

        # Create a turn segment based on the index
        current_pose = await self._get_current_pose()
        track_builder = TrackBuilder(start=current_pose)
        track_segment: Track
        next_frame_b = f"row_end_{index}"
        if index == 1:
            # Drive forward – move away from the last hole into a buffer zone.
            track_builder.create_straight_segment(
                next_frame_b=next_frame_b, distance=self.headland_buffer, spacing=0.1)
            track_segment = track_builder.track
        elif index == 2 or index == 4:
            # Turn 90° – reorient the robot toward the next row.
            track_builder.create_turn_segment(
                next_frame_b=next_frame_b, angle=radians(90 * self.turn_angle_sign))
            track_segment = track_builder.track
        else:
            # Drive forward – cross the row spacing gap.
            track_builder.create_straight_segment(
                next_frame_b=next_frame_b, distance=self.row_spacing, spacing=0.1)
            track_segment = track_builder.track

        return track_segment

    async def _shutdown(self):
        """Shutdown the motion planner."""
        if self.pose_query_task is not None:
            self.should_poll = False
            await self.pose_query_task
            self.pose_query_task = None

    async def redo_last_segment(self) -> Tuple[Optional[Track], Optional[str]]:
        """Redo the last segment.
        NOTE: It does not work for row end maneuvers, only for AB segments.

        Returns:
            The last track segment (Track) and its name.
        """
        if self.current_waypoint_index == 0 or self.current_waypoint_index is None:
            logger.info("No previous segment to redo.")
            return (None, None)

        # Check if we're completing a row end maneuver
        if self.current_waypoint_index == self.last_row_waypoint_index:
            # In this case, we need to check if we are the the last waypoint of the first row (i.e., row end index == 1)
            # Or if we have already completed the row end maneuvers and want to go again to the first waypoint
            # of the next row
            if self.row_end_segment_index == 1:
                # We are about to switch to the next row, but we haven't started the row end maneuvers yet.
                # So we just reset our index and let the motion planner handle the next segment.
                self.current_waypoint_index -= 1
            # else: In this case, we are already in the row end maneuvers and we just want to redo the last segment.
            # Don't reset the index, because if so, we would end up repeating the row end maneuvers
        else:  # We're not trying to switch rows, just redo the last AB segment
            self.current_waypoint_index -= 1

        return await self.next_track_segment()

    async def next_track_segment(self) -> Tuple[Optional[Track], Optional[str]]:
        """Get the next track segment to navigate to.

        Returns:
            The next track segment (Track)
        """
        if self.current_waypoint_index >= len(self.waypoints):
            logger.info("No more waypoints to navigate to.")
            asyncio.create_task(self._shutdown())
            return (None, None)

        # Check if this is the very first maneuver:
        if self.current_waypoint_index == 0:
            seg_name: Optional[str] = "waypoint_0_to_1"
            track: Optional[Track] = None
            maneuver_type: FirstManeuver = await self._determine_first_maneuver()
            if maneuver_type == FirstManeuver.AB:
                track = await self._create_ab_segment_to_next_waypoint()
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
            track = await self._create_ab_segment_to_next_waypoint()
            next_index = self.current_waypoint_index
            seg_name = f"waypoint_{curr_index}_to_{next_index}"
            return (track, seg_name)

        # We're switching to the next row
        # 1. Check if we have finished all row end maneuvers
        if self.row_end_segment_index >= 5:
            logger.info(
                "Finished all row end maneuvers, moving to the next row.")
            seg_name = f"row_end_5_to_waypoint_{self.current_waypoint_index + 1}"
            return (await self._create_ab_segment_to_next_waypoint(), seg_name)
        else:
            # We need to return a segment from the row end maneuver
            track_segment = await self._row_end_maneuver(self.row_end_segment_index)
            self.row_end_segment_index += 1
            return (track_segment, f"row_end_{self.row_end_segment_index}")
