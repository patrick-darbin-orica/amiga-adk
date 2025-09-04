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

import logging
from pathlib import Path

import numpy as np
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.events_file_writer import proto_to_json_file
from farm_ng.track.track_pb2 import Track
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Pose3F64
from farm_ng_core_pybind import Rotation3F64

logger = logging.getLogger("Track Planner")


class TrackBuilder:
    """A class for building tracks."""

    def __init__(self, start: Pose3F64 | None = None) -> None:
        """Initialize the TrackBuilder."""
        if start is not None:
            self._start: Pose3F64 = start
        else:
            zero_tangent = np.zeros((6, 1), dtype=np.float64)
            self._start: Pose3F64 = Pose3F64(
                a_from_b=Isometry3F64(), frame_a="world", frame_b="robot", tangent_of_b_in_a=zero_tangent
            )
        self.track_waypoints: list[Pose3F64] = []
        self._segment_indices: list[int] = [0]
        self._loaded: bool = False
        self.track_waypoints = [self._start]

    @property
    def track(self) -> Track:
        """Pack the track waypoints into a Track proto message."""
        return Track(waypoints=[pose.to_proto() for pose in self.track_waypoints])

    @track.setter
    def track(self, loaded_track: Track) -> None:
        """Unpack a Track proto message into the track waypoints."""
        self._track = loaded_track
        logger.info(f"Loaded track with {len(loaded_track.waypoints)} waypoints.")
        self.track_waypoints = [Pose3F64.from_proto(pose) for pose in self._track.waypoints]
        self._loaded = True

    def _create_segment(self, next_frame_b: str, distance: float, spacing: float, angle: float = 0) -> None:
        """Create a segment with given distance and spacing."""
        # Create a container to store the track segment waypoints
        segment_poses: list[Pose3F64] = [self.track_waypoints[-1]]
        num_segments: int

        if angle != 0:
            num_segments = max(int(abs(angle) / spacing), 1)
        else:
            num_segments = max(int(distance / spacing), 1)

        delta_angle: float = angle / num_segments
        delta_distance: float = distance / num_segments
        for i in range(1, num_segments + 1):
            segment_pose: Pose3F64 = Pose3F64(
                a_from_b=Isometry3F64([delta_distance, 0, 0], Rotation3F64.Rz(delta_angle)),
                frame_a=segment_poses[-1].frame_b,
                frame_b=f"{next_frame_b}_{i - 1}",
            )
            segment_poses.append(segment_poses[-1] * segment_pose)

        segment_poses[-1].frame_b = next_frame_b
        self.track_waypoints.extend(segment_poses)
        self._segment_indices.append(len(self.track_waypoints))
        self._loaded = False

    def _angle_wrap(self, angle: float) -> float:
        """Wrap angle to [-π, π] range."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def create_straight_segment(self, next_frame_b: str, distance: float, spacing: float = 0.1) -> None:
        """Compute a straight segment."""
        self._create_segment(next_frame_b=next_frame_b, distance=distance, spacing=spacing)

    def create_ab_segment(self, next_frame_b: str, final_pose, spacing: float = 0.1) -> None:
        """Create AB segment using direct pose interpolation."""
        initial_pose = self.track_waypoints[-1]

        # Calculate total distance
        distance = np.linalg.norm(final_pose.a_from_b.translation - initial_pose.a_from_b.translation)

        if distance < 0.01:  # Too close, skip
            return

        # Number of intermediate points
        num_points = max(1, int(distance / spacing))

        # Get initial and final headings (yaw angle)
        initial_heading = initial_pose.a_from_b.rotation.log()[-1]
        final_heading = final_pose.a_from_b.rotation.log()[-1]

        # Wrap the angle difference properly
        heading_diff = self._angle_wrap(final_heading - initial_heading)

        # Interpolate poses along the path
        for i in range(1, num_points + 1):
            t = i / num_points

            # Linear interpolation of position
            interp_translation = initial_pose.a_from_b.translation * (1 - t) + final_pose.a_from_b.translation * t

            # Linear interpolation of heading
            interp_heading = initial_heading + t * heading_diff

            # Create interpolated pose
            interp_pose = Pose3F64(
                Isometry3F64(interp_translation, Rotation3F64.Rz(interp_heading)),  # Create rotation around Z-axis
                final_pose.frame_a,
                next_frame_b,
            )

            self.track_waypoints.append(interp_pose)

    def create_turn_segment(self, next_frame_b: str, angle: float, spacing: float = 0.1) -> None:
        """Compute a turn (in place) segment."""
        self._create_segment(next_frame_b=next_frame_b, distance=0, spacing=spacing, angle=angle)

    def create_arc_segment(self, next_frame_b: str, radius: float, angle: float, spacing: float = 0.1) -> None:
        """Compute an arc segment."""
        arc_length: float = abs(angle * radius)
        self._create_segment(next_frame_b=next_frame_b, distance=arc_length, spacing=spacing, angle=angle)

    def pop_last_segment(self) -> None:
        """Remove the last (appended) segment from the track."""

        if self._loaded:
            logger.warning("Cannot pop segment from a loaded track without inserting new segments first.")
            return

        if len(self._segment_indices) > 1:  # Ensure there is a segment to pop
            last_segment_start: int = self._segment_indices[-2]  # Get the start of the last segment
            # Remove the waypoints from the last segment
            self.track_waypoints = self.track_waypoints[:last_segment_start]
            # Remove the last segment index
            self._segment_indices.pop()
        else:
            logger.info("No segment to pop.")

    def unpack_track(self) -> tuple[list[float], list[float], list[float]]:
        """Unpack x and y coordinates and heading from the waypoints for plotting.

        Args: None
        Returns:
            tuple[list[float], list[float], list[float]]: The x, y, and heading coordinates of the track waypoints.
        """

        x: list[float] = []
        y: list[float] = []
        heading: list[float] = []
        for pose in self.track_waypoints:
            x.append(pose.a_from_b.translation[0])
            y.append(pose.a_from_b.translation[1])
            heading.append(pose.a_from_b.rotation.log()[-1])
        return (x, y, heading)

    def save_track(self, path: Path) -> None:
        """Save the track to a json file.

        Args:
            path (Path): The path of the file to save.
        """
        if self.track:
            proto_to_json_file(path, self.track)
            logger.info(f"Track saved to {path}")
        else:
            logger.warning("No track to save.")

    def load_track(self, path: Path) -> None:
        """Import a track from a json file.

        Args:
            path (Path): The path of the file to import.
        """
        loaded_track = proto_from_json_file(path, Track())
        self.track = loaded_track

    def merge_tracks(self, track_to_merge: Track, threshold: float = 0.5) -> bool:
        """Merge a track with the current track.

        Args:
            track (Track): The track to merge.
        """
        # Calculate the distance from the current track to the beginning and end of the track to merge
        dist_to_current_track = np.linalg.norm(
            self.track_waypoints[-1].a_from_b.translation - track_to_merge.waypoints[0].translation
        )
        if dist_to_current_track > threshold:
            logger.warning("Track to merge is too far from the current track, cannot merge.")
            return False

        self.track_waypoints.extend([Pose3F64.from_proto(pose) for pose in track_to_merge.waypoints])
        self._segment_indices.append(len(self.track_waypoints))
        self._loaded = True
        return True

    def reverse_track(self) -> None:
        """Reverse the track."""
        self.track_waypoints = [
            Pose3F64(
                a_from_b=pose.a_from_b * Isometry3F64.Rz(np.pi),
                frame_a=pose.frame_a,
                frame_b=pose.frame_b,
                tangent_of_b_in_a=pose.tangent_of_b_in_a,
            )
            for pose in reversed(self.track_waypoints)
        ]
        self._segment_indices.reverse()
