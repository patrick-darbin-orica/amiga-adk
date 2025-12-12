#!/usr/bin/env python3
"""
Calculate pixel tolerance for oak0 alignment based on physical distance.

This script helps determine how many pixels correspond to a given physical
distance (e.g., 2cm) at the dipper tool position, accounting for camera
FOV, resolution, and mounting height.

Usage:
    python calculate_tolerance.py --physical-distance 0.02 --camera-height 1.0
"""

import argparse
import math


def calculate_pixel_tolerance(
    physical_distance_m: float,
    camera_height_m: float,
    camera_fov_deg: float = 69.0,  # OAK-D horizontal FOV
    image_width_px: int = 1920,
    image_height_px: int = 1080,
):
    """
    Calculate pixel tolerance for a given physical distance.

    Args:
        physical_distance_m: Physical distance tolerance (meters, e.g., 0.02 for 2cm)
        camera_height_m: Height of camera above ground (meters)
        camera_fov_deg: Camera horizontal field of view (degrees)
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels

    Returns:
        tuple: (horizontal_tolerance_px, vertical_tolerance_px)
    """

    # Convert FOV to radians
    camera_fov_rad = math.radians(camera_fov_deg)

    # Calculate ground width covered at camera height
    # For a camera looking down at angle, the FOV covers:
    # width_at_ground = 2 * camera_height * tan(fov/2)
    ground_width_m = 2 * camera_height_m * math.tan(camera_fov_rad / 2)

    # Calculate pixels per meter (horizontal)
    pixels_per_meter_horizontal = image_width_px / ground_width_m

    # For vertical (assuming similar aspect ratio scaling)
    aspect_ratio = image_height_px / image_width_px
    ground_height_m = ground_width_m * aspect_ratio
    pixels_per_meter_vertical = image_height_px / ground_height_m

    # Calculate tolerance in pixels
    horizontal_tolerance_px = physical_distance_m * pixels_per_meter_horizontal
    vertical_tolerance_px = physical_distance_m * pixels_per_meter_vertical

    return horizontal_tolerance_px, vertical_tolerance_px


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pixel tolerance for oak0 alignment"
    )

    parser.add_argument(
        "--physical-distance",
        type=float,
        default=0.02,
        help="Physical distance tolerance in meters (default: 0.02 = 2cm)"
    )

    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.0,
        help="Camera mounting height above ground in meters (default: 1.0m)"
    )

    parser.add_argument(
        "--camera-fov",
        type=float,
        default=69.0,
        help="Camera horizontal FOV in degrees (default: 69.0 for OAK-D)"
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
        help="Image width in pixels (default: 1920)"
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
        help="Image height in pixels (default: 1080)"
    )

    args = parser.parse_args()

    # Calculate tolerance
    h_tol_px, v_tol_px = calculate_pixel_tolerance(
        physical_distance_m=args.physical_distance,
        camera_height_m=args.camera_height,
        camera_fov_deg=args.camera_fov,
        image_width_px=args.image_width,
        image_height_px=args.image_height,
    )

    # Display results
    print(f"\n{'='*70}")
    print("PIXEL TOLERANCE CALCULATION")
    print(f"{'='*70}")
    print(f"Physical distance:     {args.physical_distance*100:.1f} cm ({args.physical_distance} m)")
    print(f"Camera height:         {args.camera_height} m")
    print(f"Camera FOV:            {args.camera_fov}°")
    print(f"Image resolution:      {args.image_width} x {args.image_height} px")
    print(f"{'-'*70}")

    # Calculate ground coverage
    camera_fov_rad = math.radians(args.camera_fov)
    ground_width_m = 2 * args.camera_height * math.tan(camera_fov_rad / 2)
    aspect_ratio = args.image_height / args.image_width
    ground_height_m = ground_width_m * aspect_ratio

    print(f"Ground coverage:       {ground_width_m:.2f}m x {ground_height_m:.2f}m")
    print(f"Pixels per meter (H):  {args.image_width/ground_width_m:.1f} px/m")
    print(f"Pixels per meter (V):  {args.image_height/ground_height_m:.1f} px/m")
    print(f"{'-'*70}")
    print(f"Horizontal tolerance:  ±{h_tol_px:.1f} pixels")
    print(f"Vertical tolerance:    ±{v_tol_px:.1f} pixels")
    print(f"{'-'*70}")
    print(f"Recommended setting:   --tolerance-px {int(round(v_tol_px))}")
    print(f"{'='*70}\n")

    # Suggestions
    print("Notes:")
    print(f"  • For forward/backward alignment, use vertical tolerance: {int(round(v_tol_px))}px")
    print(f"  • At {args.camera_height}m height, each pixel ≈ {(ground_height_m/args.image_height)*100:.2f}cm vertically")
    print(f"  • Adjust --camera-height if oak0 mounting height differs")
    print(f"  • Consider using a more conservative (larger) tolerance for safety")
    print()


if __name__ == "__main__":
    main()
