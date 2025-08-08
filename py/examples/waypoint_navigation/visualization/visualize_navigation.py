#!/usr/bin/env python3
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
"""Post-processing visualization script for navigation data.

This script loads and visualizes:
1. Surveyed waypoints from the original waypoints file
2. All track segments from navigation_progress.json
3. Robot start positions from robot_positions.json

Usage:
    python visualize_navigation.py --waypoints-path /path/to/waypoints.json [options]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class PostProcessingVisualizer:
    """Post-processing visualizer for completed navigation sessions."""

    def __init__(self, waypoints_path: Path, navigation_progress_path: Path = None, robot_positions_path: Path = None):
        """Initialize the visualizer.

        Args:
            waypoints_path: Path to the surveyed waypoints JSON file
            navigation_progress_path: Path to navigation_progress.json (optional)
            robot_positions_path: Path to robot_positions.json (optional)
        """
        self.waypoints_path = waypoints_path
        self.navigation_progress_path = navigation_progress_path or Path("navigation_progress.json")
        self.robot_positions_path = robot_positions_path or Path("robot_positions.json")

        # Data storage
        self.surveyed_waypoints = []
        self.track_segments = {}
        self.robot_positions = []

        # Coordinate normalization
        self.x_offset = 0
        self.y_offset = 0

        # Visualization settings
        self.colors = {
            'surveyed': 'red',
            'track_segments': [
                'blue',
                'green',
                'orange',
                'purple',
                'brown',
                'pink',
                'gray',
                'olive',
                'cyan',
                'magenta',
            ],
            'robot_positions': 'black',
            'start_point': 'red',
        }

        # Load all data
        self._load_all_data()
        self._calculate_offsets()

    def _load_all_data(self):
        """Load all navigation data from JSON files."""
        self._load_surveyed_waypoints()
        self._load_track_segments()
        self._load_robot_positions()

    def _load_surveyed_waypoints(self):
        """Load surveyed waypoints from JSON file."""
        try:
            with open(self.waypoints_path, 'r') as f:
                data = json.load(f)
                self.surveyed_waypoints = data.get('waypoints', [])
            print(f"✅ Loaded {len(self.surveyed_waypoints)} surveyed waypoints")
        except Exception as e:
            print(f"❌ Failed to load surveyed waypoints: {e}")
            self.surveyed_waypoints = []

    def _load_track_segments(self):
        """Load track segments from navigation_progress.json."""
        try:
            if self.navigation_progress_path.exists():
                with open(self.navigation_progress_path, 'r') as f:
                    data = json.load(f)
                    self.track_segments = data
                print(f"✅ Loaded {len(self.track_segments)} track segments")
            else:
                print(f"⚠️  Navigation progress file not found: {self.navigation_progress_path}")
        except Exception as e:
            print(f"❌ Failed to load track segments: {e}")
            self.track_segments = {}

    def _load_robot_positions(self):
        """Load robot positions from robot_positions.json."""
        try:
            if self.robot_positions_path.exists():
                with open(self.robot_positions_path, 'r') as f:
                    self.robot_positions = json.load(f)
                print(f"✅ Loaded {len(self.robot_positions)} robot positions")
            else:
                print(f"⚠️  Robot positions file not found: {self.robot_positions_path}")
        except Exception as e:
            print(f"❌ Failed to load robot positions: {e}")
            self.robot_positions = []

    def _calculate_offsets(self):
        """Calculate offsets to normalize coordinates to smaller, more readable values."""
        all_x = []
        all_y = []

        # Collect all x,y coordinates
        if self.surveyed_waypoints:
            survey_x, survey_y, _ = self.unpack_surveyed_waypoints()
            all_x.extend(survey_x)
            all_y.extend(survey_y)

        for segment_data in self.track_segments.values():
            x_coords, y_coords, _ = self.unpack_track_segment(segment_data)
            all_x.extend(x_coords)
            all_y.extend(y_coords)

        if self.robot_positions:
            all_x.extend([pos['x'] for pos in self.robot_positions])
            all_y.extend([pos['y'] for pos in self.robot_positions])

        if all_x and all_y:
            # Use the minimum values as offsets to shift coordinates closer to origin
            self.x_offset = min(all_x)
            self.y_offset = min(all_y)
            print(f"🔧 Coordinate normalization: X offset = {self.x_offset:.2f}, Y offset = {self.y_offset:.2f}")
        else:
            self.x_offset = 0
            self.y_offset = 0

    def _normalize_coords(self, x_coords, y_coords):
        """Apply coordinate normalization."""
        norm_x = [x - self.x_offset for x in x_coords]
        norm_y = [y - self.y_offset for y in y_coords]
        return norm_x, norm_y

    def unpack_surveyed_waypoints(self) -> Tuple[List[float], List[float], List[float]]:
        """Unpack surveyed waypoints from JSON format.

        Returns:
            Tuple of (x_coords, y_coords, headings)
        """
        x_coords = []
        y_coords = []
        headings = []

        for waypoint in self.surveyed_waypoints:
            # Extract translation
            translation = waypoint['aFromB']['translation']
            x_coords.append(translation['x'])
            y_coords.append(translation['y'])

            # Extract heading from quaternion
            quat = waypoint['aFromB']['rotation']['unitQuaternion']
            z_imag = quat['imag'].get('z', 0)
            real = quat['real']
            heading = 2 * np.arctan2(z_imag, real)
            headings.append(heading)

        return x_coords, y_coords, headings

    def unpack_track_segment(self, segment_data: Dict) -> Tuple[List[float], List[float], List[float]]:
        """Unpack track segment from JSON format.

        Args:
            segment_data: Track segment data from navigation_progress.json

        Returns:
            Tuple of (x_coords, y_coords, headings)
        """
        # The new format has x, y, heading as direct arrays
        x_coords = segment_data.get('x', [])
        y_coords = segment_data.get('y', [])
        headings = segment_data.get('heading', [])

        return x_coords, y_coords, headings

    def plot_comprehensive_overview(
        self,
        show_headings: bool = True,
        save_plot: bool = True,
        show_waypoint_numbers: bool = False,
        output_dir: Path = None,
    ):
        """Create a comprehensive plot showing all navigation elements.

        Args:
            show_headings: Whether to show heading arrows
            save_plot: Whether to save the plot to file
            show_waypoint_numbers: Whether to show waypoint numbers
            output_dir: Directory to save plots (default: current directory)
        """
        plt.figure(figsize=(16, 12))

        # Plot surveyed waypoints as 10-inch (25.4cm) holes
        if self.surveyed_waypoints:
            survey_x, survey_y, survey_headings = self.unpack_surveyed_waypoints()
            norm_x, norm_y = self._normalize_coords(survey_x, survey_y)

            # Plot the holes as circles (10 inches = 25.4 cm diameter)
            hole_radius_m = 0.254 / 2  # 10 inches = 25.4 cm, radius = 12.7 cm
            for x, y in zip(norm_x, norm_y):
                circle = plt.Circle((x, y), hole_radius_m, color='red', alpha=0.3, fill=True, zorder=2)
                plt.gca().add_patch(circle)
                # Add hole center point
                plt.scatter(x, y, c='red', marker='+', s=50, linewidth=2, zorder=5, alpha=0.8)

            # Add legend entry for holes
            plt.scatter([], [], c='red', marker='o', s=100, alpha=0.3, label='10" Target Holes', edgecolors='red')

            if show_waypoint_numbers:
                for i, (x, y) in enumerate(zip(norm_x, norm_y)):
                    plt.annotate(
                        f'W{i}',
                        (x, y),
                        xytext=(15, 15),
                        textcoords='offset points',
                        fontsize=8,
                        color='darkred',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                    )

            if show_headings:
                # Plot surveyed waypoint headings (smaller arrows)
                survey_u = np.cos(survey_headings)
                survey_v = np.sin(survey_headings)
                plt.quiver(
                    norm_x,
                    norm_y,
                    survey_u,
                    survey_v,
                    angles='xy',
                    scale_units='xy',
                    scale=8,
                    color='darkred',
                    alpha=0.7,
                    width=0.003,
                    zorder=4,
                )

        # Plot track segments
        for i, (segment_name, segment_data) in enumerate(self.track_segments.items()):
            color = self.colors['track_segments'][i % len(self.colors['track_segments'])]
            x_coords, y_coords, headings = self.unpack_track_segment(segment_data)

            if not x_coords:  # Skip empty segments
                continue

            # Normalize coordinates
            norm_x, norm_y = self._normalize_coords(x_coords, y_coords)

            # Plot track path (thinner line)
            plt.plot(norm_x, norm_y, color=color, linewidth=1.5, label=f'{segment_name}', alpha=0.7, zorder=2)

            # Plot start point of each segment (smaller)
            plt.scatter(
                norm_x[0], norm_y[0], c=color, marker='o', s=30, edgecolors='white', linewidth=1, alpha=1.0, zorder=3
            )

            # Plot end point of each segment (smaller)
            plt.scatter(
                norm_x[-1], norm_y[-1], c=color, marker='D', s=25, edgecolors='white', linewidth=1, alpha=1.0, zorder=3
            )

            if show_headings and len(x_coords) > 0:
                # Plot heading arrows (smaller and less frequent)
                arrow_interval = max(1, len(x_coords) // 5)
                u_coords = np.cos(headings)
                v_coords = np.sin(headings)

                for j in range(0, len(x_coords), arrow_interval):
                    plt.quiver(
                        norm_x[j],
                        norm_y[j],
                        u_coords[j],
                        v_coords[j],
                        angles='xy',
                        scale_units='xy',
                        scale=12,
                        color=color,
                        alpha=0.5,
                        width=0.002,
                        zorder=1,
                    )

        # Plot recorded robot positions (tiny dots for precision)
        if self.robot_positions:
            robot_x = [pos['x'] for pos in self.robot_positions]
            robot_y = [pos['y'] for pos in self.robot_positions]
            norm_robot_x, norm_robot_y = self._normalize_coords(robot_x, robot_y)

            plt.scatter(
                norm_robot_x, norm_robot_y, c='lime', marker='.', s=1, label='Robot Stop Positions', alpha=1.0, zorder=6
            )

            # Add position numbers (smaller)
            for i, (x, y) in enumerate(zip(norm_robot_x, norm_robot_y)):
                plt.annotate(
                    f'{i+1}',
                    (x, y),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=6,
                    color='black',
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8),
                )

            # Robot heading arrows (smaller)
            if show_headings:
                for i, pos in enumerate(self.robot_positions):
                    robot_u = np.cos(pos['heading'])
                    robot_v = np.sin(pos['heading'])
                    plt.quiver(
                        norm_robot_x[i],
                        norm_robot_y[i],
                        robot_u,
                        robot_v,
                        angles='xy',
                        scale_units='xy',
                        scale=10,
                        color='lime',
                        width=0.004,
                        alpha=0.8,
                        zorder=5,
                    )

        plt.axis('equal')

        # Add fine grid for precision analysis (10 cm = 0.1 m grid)
        ax = plt.gca()

        # Get current axis limits and round them for cleaner grid
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Major grid every 0.5m, minor grid every 0.1m (10 cm)
        x_major = np.arange(np.floor(xlim[0] / 0.5) * 0.5, np.ceil(xlim[1] / 0.5) * 0.5 + 0.5, 0.5)
        y_major = np.arange(np.floor(ylim[0] / 0.5) * 0.5, np.ceil(ylim[1] / 0.5) * 0.5 + 0.5, 0.5)
        x_minor = np.arange(np.floor(xlim[0] / 0.1) * 0.1, np.ceil(xlim[1] / 0.1) * 0.1 + 0.1, 0.1)
        y_minor = np.arange(np.floor(ylim[0] / 0.1) * 0.1, np.ceil(ylim[1] / 0.1) * 0.1 + 0.1, 0.1)

        ax.set_xticks(x_major)
        ax.set_yticks(y_major)
        ax.set_xticks(x_minor, minor=True)
        ax.set_yticks(y_minor, minor=True)

        ax.grid(True, which='major', alpha=0.5, linewidth=0.8)
        ax.grid(True, which='minor', alpha=0.2, linewidth=0.5)

        plt.xlabel('X (m) - Normalized', fontsize=12)
        plt.ylabel('Y (m) - Normalized', fontsize=12)
        plt.title('Robot Performance Analysis: 10" Target Holes vs Robot Stop Positions', fontsize=14, pad=20)

        # Create legend with better positioning
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        legend.set_zorder(10)

        plt.text(
            0.02,
            0.98,
            f"Track Segments: {len(self.track_segments)}\nRobot Positions: {len(self.robot_positions)}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.9),
            zorder=10,
        )

        if save_plot:
            if output_dir is None:
                output_dir = Path.cwd()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = output_dir / f"navigation_overview_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to {plot_path}")

        plt.tight_layout()
        plt.show()

    def plot_segments_individually(self, output_dir: Path = None):
        """Plot each track segment individually."""
        if not self.track_segments:
            print("❌ No track segments to plot")
            return

        for i, (segment_name, segment_data) in enumerate(self.track_segments.items()):
            plt.figure(figsize=(12, 10))

            color = self.colors['track_segments'][i % len(self.colors['track_segments'])]
            x_coords, y_coords, headings = self.unpack_track_segment(segment_data)

            if not x_coords:  # Skip empty segments
                continue

            # Plot the track segment
            plt.plot(x_coords, y_coords, color=color, linewidth=4, label=f'{segment_name}', zorder=2)

            # Plot waypoints
            plt.scatter(x_coords, y_coords, c=color, marker='o', s=40, alpha=0.7, zorder=3)

            # Plot start and end points
            plt.scatter(
                x_coords[0],
                y_coords[0],
                c='green',
                marker='o',
                s=150,
                label='Start',
                edgecolors='black',
                linewidth=2,
                zorder=4,
            )
            plt.scatter(
                x_coords[-1],
                y_coords[-1],
                c='red',
                marker='D',
                s=120,
                label='End',
                edgecolors='black',
                linewidth=2,
                zorder=4,
            )

            # Plot heading arrows
            u_coords = np.cos(headings)
            v_coords = np.sin(headings)

            arrow_interval = max(1, len(x_coords) // 15)
            for j in range(0, len(x_coords), arrow_interval):
                plt.quiver(
                    x_coords[j],
                    y_coords[j],
                    u_coords[j],
                    v_coords[j],
                    angles='xy',
                    scale_units='xy',
                    scale=2,
                    color=color,
                    alpha=0.8,
                    width=0.005,
                    zorder=1,
                )

            # Show corresponding robot position if available
            robot_pos = None
            for pos in self.robot_positions:
                if pos['segment_name'] == segment_name:
                    robot_pos = pos
                    break

            if robot_pos:
                plt.scatter(
                    robot_pos['x'],
                    robot_pos['y'],
                    c='lime',
                    marker='^',
                    s=180,
                    label='Robot Start Position',
                    edgecolors='black',
                    linewidth=2,
                    zorder=5,
                )

                # Robot heading arrow
                robot_u = np.cos(robot_pos['heading'])
                robot_v = np.sin(robot_pos['heading'])
                plt.quiver(
                    robot_pos['x'],
                    robot_pos['y'],
                    robot_u,
                    robot_v,
                    angles='xy',
                    scale_units='xy',
                    scale=1.5,
                    color='lime',
                    width=0.008,
                    zorder=4,
                )

            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.xlabel('X (m)', fontsize=12)
            plt.ylabel('Y (m)', fontsize=12)
            plt.title(f'Track Segment: {segment_name}', fontsize=14, pad=20)
            plt.legend(fontsize=11)

            # Add segment stats
            stats_text = f"Waypoints: {len(x_coords)}\n"
            if x_coords:
                path_length = sum(
                    np.sqrt((x_coords[j + 1] - x_coords[j]) ** 2 + (y_coords[j + 1] - y_coords[j]) ** 2)
                    for j in range(len(x_coords) - 1)
                )
                stats_text += f"Path Length: {path_length:.2f}m"

            plt.text(
                0.02,
                0.98,
                stats_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
            )

            if output_dir:
                plot_path = output_dir / f"segment_{i+1:02d}_{segment_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"💾 Segment plot saved to {plot_path}")

            plt.tight_layout()
            plt.show()


def main():
    """Main function for the post-processing visualizer."""
    parser = argparse.ArgumentParser(description="Post-processing visualization for navigation data")

    # Required arguments
    parser.add_argument("--waypoints-path", type=Path, required=True, help="Path to surveyed waypoints JSON file")

    # Optional arguments
    parser.add_argument(
        "--navigation-progress",
        type=Path,
        default=Path("navigation_progress.json"),
        help="Path to navigation_progress.json file (default: navigation_progress.json)",
    )

    parser.add_argument(
        "--robot-positions",
        type=Path,
        default=Path("robot_positions.json"),
        help="Path to robot_positions.json file (default: robot_positions.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("navigation_plots"),
        help="Directory to save plots and reports (default: navigation_plots)",
    )

    parser.add_argument("--no-headings", action="store_true", help="Don't show heading arrows in plots")

    parser.add_argument("--individual-segments", action="store_true", help="Also plot each segment individually")

    parser.add_argument(
        "--show-waypoint-numbers", action="store_true", help="Show waypoint numbers on surveyed waypoints"
    )

    parser.add_argument("--no-save", action="store_true", help="Don't save plots to files")

    args = parser.parse_args()

    # Validate waypoints file
    if not args.waypoints_path.exists():
        print(f"❌ Waypoints file not found: {args.waypoints_path}")
        return

    # Create output directory
    if not args.no_save:
        args.output_dir.mkdir(exist_ok=True)

    # Create visualizer
    print("🎨 Initializing post-processing visualizer...")
    visualizer = PostProcessingVisualizer(
        waypoints_path=args.waypoints_path,
        navigation_progress_path=args.navigation_progress,
        robot_positions_path=args.robot_positions,
    )

    # Generate comprehensive overview
    print("🖼️  Creating comprehensive navigation overview...")
    visualizer.plot_comprehensive_overview(
        show_headings=not args.no_headings,
        save_plot=not args.no_save,
        show_waypoint_numbers=args.show_waypoint_numbers,
        output_dir=args.output_dir if not args.no_save else None,
    )

    # Generate individual segment plots if requested
    if args.individual_segments:
        print("🖼️  Creating individual segment plots...")
        visualizer.plot_segments_individually(output_dir=args.output_dir if not args.no_save else None)

    print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
