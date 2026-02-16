#!/usr/bin/env python3
"""
Test script to verify dipbob completion signal monitoring.

This script listens for the dipbob completion signal on the CAN bus.
The dipbob script (running on Raspberry Pi) sends a completion signal
via CAN bus after measuring and logging data.

Usage:
    python test_dipbob_completion.py --timeout 10.0

    Then trigger a dipbob cycle and watch for the completion signal.
"""

import asyncio
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.canbus import wait_for_dipbob_completion


async def main(args):
    print("=" * 70)
    print("DIPBOB COMPLETION SIGNAL MONITOR")
    print("=" * 70)
    print(f"CAN Channel:       {args.can_channel}")
    print(f"CAN ID:            0x{args.can_id:X} (extended)")
    print(f"Completion byte:   data[2] = 0x{args.completion_byte:02X}")
    print(f"Timeout:           {args.timeout}s")
    print("=" * 70)
    print()
    print("Listening for dipbob completion signal...")
    print("(Trigger a dipbob cycle now)")
    print()

    # Wait for completion signal (canbus_client=None since we use python-can directly)
    result = await wait_for_dipbob_completion(
        canbus_client=None,
        timeout=args.timeout,
        can_id=args.can_id,
        completion_byte=args.completion_byte,
        can_channel=args.can_channel
    )

    print()
    if result:
        print("=" * 70)
        print("✓ SUCCESS: Dipbob completion signal received!")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("✗ TIMEOUT: No completion signal received within timeout period")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("  1. Verify dipbob script is running on Raspberry Pi")
        print("  2. Verify dipbob sends completion signal (data[2]=0x20)")
        print(f"  3. Check CAN interface is up: ip link show {args.can_channel}")
        print(f"  4. Monitor CAN traffic: candump {args.can_channel}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test dipbob completion signal monitoring"
    )
    parser.add_argument(
        "--can-channel",
        type=str,
        default="can0",
        help="CAN interface to monitor (default: can0)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds to wait for completion signal"
    )
    parser.add_argument(
        "--can-id",
        type=lambda x: int(x, 0),
        default=0x18FF0007,
        help="CAN ID to monitor (hex or decimal)"
    )
    parser.add_argument(
        "--completion-byte",
        type=lambda x: int(x, 0),
        default=0x20,
        help="Expected completion byte value in data[2]"
    )

    args = parser.parse_args()

    try:
        exit_code = asyncio.run(main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
