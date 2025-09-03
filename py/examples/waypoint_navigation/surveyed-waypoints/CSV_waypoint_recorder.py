#!/usr/bin/env python3
"""
Amiga filter client with "press Enter to save dx/dy" to CSV.

- Mirrors the standard filter client console output.
- On every Enter key press, appends a CSV row with:
  ID,X,Y,Z,Bearing,Diameter,dx,dy
  where dy = pose.translation[0], dx = pose.translation[1] (both rounded to 7f)

Defaults:
  --service-config  ->  ./service_config.json   (same directory as this script)
  (CSV path is required)

Usage:
  python CSV_waypoint_recorder.py --csv waypointTest1.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.stamp import get_stamp_by_semantics_and_clock_type
from farm_ng.core.stamp import StampSemantics
from farm_ng.filter.filter_pb2 import DivergenceCriteria
from farm_ng_core_pybind import Pose3F64


@dataclass
class StaticFields:
    X: Optional[float] = None
    Y: Optional[float] = None
    Z: Optional[float] = None
    Bearing: Optional[float] = None
    Diameter: Optional[float] = None


class PoseSaver:
    """Keeps the latest pose and writes CSV rows on demand, auto-incrementing ID."""

    HEADER = ["ID", "X", "Y", "Z", "Bearing", "Diameter", "dx", "dy"]

    def __init__(self, csv_path: Path, static: StaticFields):
        self.csv_path = csv_path
        self.static = static
        self.latest_dx: Optional[float] = None
        self.latest_dy: Optional[float] = None
        self.current_id: int = 0
        self._prepare_csv()

    def _prepare_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)
            self.current_id = 0
        else:
            # Read last non-header row's ID to continue counting
            try:
                last_id = None
                with self.csv_path.open("r", newline="") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or row == self.HEADER:
                            continue
                        # Robustly parse ID if present
                        try:
                            last_id = int(row[0])
                        except Exception:
                            pass
                self.current_id = (last_id + 1) if last_id is not None else 0
            except Exception:
                # Fallback if reading fails
                self.current_id = 0

    def update_pose(self, pose: Pose3F64) -> None:
        """
        Map:
          translation[0] -> dy
          translation[1] -> -dx
          Filter client uses NWU convention, while robot uses NED
        """
        self.latest_dy = float(pose.translation[0])
        self.latest_dx = float(pose.translation[1])

    def save_current(self) -> bool:
        """Append a CSV row using the latest dx/dy (rounded to 7f). Returns True if saved."""
        if self.latest_dx is None or self.latest_dy is None:
            return False

        def s(v: Optional[float]) -> str:
            return "" if v is None else f"{v}"

        row = [
            str(self.current_id),
            s(self.static.X),
            s(self.static.Y),
            s(self.static.Z),
            s(self.static.Bearing),
            s(self.static.Diameter),
            f"{-self.latest_dx:.7f}", #flipped for NED convention
            f"{self.latest_dy:.7f}",
        ]

        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

        self.current_id += 1
        return True


async def print_and_track_filter(config: EventServiceConfig, saver: PoseSaver):
    """Subscribe to filter events, mirror console output, and update saver with latest pose."""
    async for event, message in EventClient(config).subscribe(config.subscriptions[0], decode=True):
        # Identify timestamp the filter calculated the state (monotonic), falling back to first ts.
        stamp = (
            get_stamp_by_semantics_and_clock_type(event, StampSemantics.SERVICE_SEND, "monotonic")
            or event.timestamps[0].stamp
        )

        # Unpack the filter state message
        pose: Pose3F64 = Pose3F64.from_proto(message.pose)
        orientation: float = message.heading
        uncertainties = [message.uncertainty_diagonal.data[i] for i in range(3)]
        divergence_criteria = [DivergenceCriteria.Name(c) for c in message.divergence_criteria]

        # Mirror original output
        print("\n###################")
        print(f"Timestamp: {stamp}")
        print("Filter state received with pose:")
        print(f"x: {pose.translation[0]:.7f} m, y: {pose.translation[1]:.7f} m, orientation: {orientation:.7f} rad")
        print(f"Parent frame: {pose.frame_a} -> Child frame: {pose.frame_b}")
        print(f"Filter has converged: {message.has_converged}")
        print("Pose uncertainties:")
        print(f"x: {uncertainties[0]:.7f} m, y: {uncertainties[1]:.7f} m, orientation: {uncertainties[2]:.7f} rad")
        if not message.has_converged:
            print(f"Filter diverged due to: {divergence_criteria}")

        # Update the saver with the newest pose (so Enter saves these values)
        saver.update_pose(pose)


async def enter_key_listener(saver: PoseSaver):
    """
    Listens for newline presses without blocking the event stream.
    Each Enter press attempts to append a row to the CSV.
    """
    print("\nPress Enter at any time to save the latest dx/dy to CSV.")
    loop = asyncio.get_running_loop()
    while True:
        _ = await loop.run_in_executor(None, sys.stdin.readline)
        if saver.save_current():
            print(f"[saved] ID={saver.current_id - 1} appended to: {saver.csv_path}")
        else:
            print("[warn] No pose received yet — nothing saved.")


def default_service_config_path() -> Path:
    # Default to service_config.json in the same directory as this script
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is not defined (rare)
        here = Path.cwd()
    return here / "service_config.json"


async def main():
    parser = argparse.ArgumentParser(
        prog="python save_dx_dy.py",
        description="Amiga filter stream that saves dx/dy to CSV on Enter."
    )
    parser.add_argument(
        "--service-config",
        type=Path,
        default=default_service_config_path(),
        help="Path to the filter service config JSON (default: ./service_config.json next to this script)."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to output CSV file.")

    # Optional fixed fields for the CSV
    parser.add_argument("--X", type=float, default=None, help="Constant X field for CSV rows.")
    parser.add_argument("--Y", type=float, default=None, help="Constant Y field for CSV rows.")
    parser.add_argument("--Z", type=float, default=None, help="Constant Z field for CSV rows.")
    parser.add_argument("--Bearing", type=float, default=None, help="Constant Bearing field for CSV rows.")
    parser.add_argument("--Diameter", type=float, default=None, help="Constant Diameter field for CSV rows.")

    args = parser.parse_args()

    # Load service config
    config: EventServiceConfig = proto_from_json_file(args.service_config, EventServiceConfig())

    static = StaticFields(
        X=args.X,
        Y=args.Y,
        Z=args.Z,
        Bearing=args.Bearing,
        Diameter=args.Diameter,
    )
    saver = PoseSaver(args.csv, static)

    tasks = [
        asyncio.create_task(print_and_track_filter(config, saver)),
        asyncio.create_task(enter_key_listener(saver)),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        for t in tasks:
            t.cancel()
        print("\nShutting down gracefully.")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    asyncio.run(main())
