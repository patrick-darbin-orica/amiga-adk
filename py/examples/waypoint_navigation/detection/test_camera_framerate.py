#!/usr/bin/env python3
"""
Test script to measure oak0 camera frame delivery rate via gRPC.

This script ONLY receives and decodes frames - no YOLO inference.
Use this to isolate whether frame rate issues are from:
1. The gRPC camera service itself (bursty delivery)
2. Inference processing overhead

Usage:
    python test_camera_framerate.py
"""

import asyncio
import argparse
import cv2
import numpy as np
from pathlib import Path
import time

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri


async def test_framerate(oak0_config_path: Path, every_n: int = 1, duration_sec: int = 30):
    """
    Measure camera frame delivery rate.

    Args:
        oak0_config_path: Path to oak0 service config
        every_n: Frame skip parameter (1 = all frames)
        duration_sec: How long to run the test
    """

    print("="*70)
    print("CAMERA FRAME RATE TEST")
    print("="*70)
    print(f"Config:     {oak0_config_path}")
    print(f"Every_n:    {every_n} (1 = all frames)")
    print(f"Duration:   {duration_sec} seconds")
    print("="*70 + "\n")

    # Load config
    if not oak0_config_path.exists():
        print(f"❌ Config not found: {oak0_config_path}")
        return

    oak0_config = proto_from_json_file(oak0_config_path, EventServiceConfig())

    # Create client
    oak0_client = EventClient(oak0_config)

    # Create subscription
    subscription = SubscribeRequest(
        uri=Uri(path="/rgb", query="service_name=oak/0"),
        every_n=every_n
    )

    print(f"✓ Connected to oak0: {oak0_config.host}:{oak0_config.port}")
    print(f"\n🚀 Starting frame rate test for {duration_sec} seconds...\n")

    # Timing variables
    start_time = time.time()
    last_frame_time = time.time()
    frame_count = 0

    # Track frame intervals for statistics
    intervals = []

    # FPS measurements
    fps_window = []
    fps_window_size = 10  # Calculate FPS over last 10 frames

    try:
        async for event, message in oak0_client.subscribe(subscription, decode=True):
            current_time = time.time()

            # Check if test duration exceeded
            elapsed = current_time - start_time
            if elapsed > duration_sec:
                break

            # Decode image (minimal processing, just to match real usage)
            image = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue

            frame_count += 1

            # Calculate interval since last frame
            interval = current_time - last_frame_time
            intervals.append(interval)
            last_frame_time = current_time

            # Calculate instantaneous FPS
            instant_fps = 1.0 / interval if interval > 0 else 0
            fps_window.append(instant_fps)
            if len(fps_window) > fps_window_size:
                fps_window.pop(0)

            avg_fps = sum(fps_window) / len(fps_window)

            # Log every 10 frames
            if frame_count % 10 == 0:
                print(f"[Frame {frame_count:3d}] Instant FPS: {instant_fps:6.1f} | "
                      f"Avg FPS (last {len(fps_window)}): {avg_fps:5.1f} | "
                      f"Interval: {interval*1000:5.1f}ms")

    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")

    # Calculate statistics
    elapsed_total = time.time() - start_time
    avg_fps_overall = frame_count / elapsed_total if elapsed_total > 0 else 0

    if intervals:
        min_interval = min(intervals)
        max_interval = max(intervals)
        avg_interval = sum(intervals) / len(intervals)

        max_fps = 1.0 / min_interval if min_interval > 0 else 0
        min_fps = 1.0 / max_interval if max_interval > 0 else 0
    else:
        min_interval = max_interval = avg_interval = 0
        max_fps = min_fps = 0

    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total frames:      {frame_count}")
    print(f"Total time:        {elapsed_total:.1f}s")
    print(f"Average FPS:       {avg_fps_overall:.2f}")
    print(f"\nFrame Interval Statistics:")
    print(f"  Min interval:    {min_interval*1000:.1f}ms  (Max FPS: {max_fps:.1f})")
    print(f"  Max interval:    {max_interval*1000:.1f}ms  (Min FPS: {min_fps:.1f})")
    print(f"  Avg interval:    {avg_interval*1000:.1f}ms")

    # Detect bursty behavior
    if max_interval > 5 * avg_interval:
        print(f"\n⚠️  BURSTY DELIVERY DETECTED!")
        print(f"   Max interval ({max_interval*1000:.0f}ms) is {max_interval/avg_interval:.1f}x larger than average")
        print(f"   This indicates irregular frame delivery (bursts + pauses)")
    elif max_interval > 2 * avg_interval:
        print(f"\n⚠️  Moderate delivery variation detected")
        print(f"   Max interval is {max_interval/avg_interval:.1f}x larger than average")
    else:
        print(f"\n✓ Frame delivery is relatively steady")

    # Count frames over/under threshold
    threshold = avg_interval * 1.5
    slow_frames = sum(1 for i in intervals if i > threshold)
    fast_frames = sum(1 for i in intervals if i < avg_interval / 1.5)

    print(f"\nFrame Distribution:")
    print(f"  Slow frames (>{threshold*1000:.0f}ms): {slow_frames} ({100*slow_frames/len(intervals):.1f}%)")
    print(f"  Fast frames (<{avg_interval*1000/1.5:.0f}ms): {fast_frames} ({100*fast_frames/len(intervals):.1f}%)")
    print(f"  Normal frames:                  {len(intervals)-slow_frames-fast_frames}")

    print(f"{'='*70}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Test oak0 camera frame delivery rate"
    )

    parser.add_argument(
        "--oak0-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "camera_client" / "service_config.json",
        help="Path to oak0 camera service config"
    )

    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Frame skip parameter (1 = all frames, 10 = every 10th frame)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds"
    )

    args = parser.parse_args()

    await test_framerate(args.oak0_config, args.every_n, args.duration)


if __name__ == "__main__":
    asyncio.run(main())
