#!/usr/bin/env python3
"""
Shared camera frame cache for oak2 camera (downward-facing alignment camera).

Provides inter-process communication between oak2 camera stream and Flask GUI.
Uses file-based shared memory for efficient frame transfer.
"""

import numpy as np
import cv2
import threading
from pathlib import Path
from typing import Optional

# Shared frame file location for oak2
FRAME_FILE_OAK2 = Path("/tmp/amiga_oak2_frame.jpg")

# Flag file to indicate inference is running (prevents Flask from overwriting processed frames)
INFERENCE_ACTIVE_FLAG = Path("/tmp/amiga_oak2_inference_active")

# Flag file to control hole alignment behavior
ALIGNMENT_ENABLED_FLAG = Path("/tmp/amiga_alignment_enabled")

_frame_lock = threading.Lock()


def set_oak2_frame(frame: np.ndarray) -> None:
    """
    Set the latest oak2 camera frame by writing to shared file (thread-safe, inter-process).

    Args:
        frame: OpenCV BGR image (numpy array)
    """
    if frame is None:
        return

    with _frame_lock:
        try:
            # Encode as JPEG with lower quality for faster processing (70 instead of 85)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            # Write atomically using temp file + rename
            temp_file = FRAME_FILE_OAK2.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(buffer.tobytes())
            temp_file.replace(FRAME_FILE_OAK2)
        except Exception as e:
            # Silently fail to avoid disrupting camera processing
            pass


def get_oak2_frame() -> Optional[np.ndarray]:
    """
    Get the latest oak2 camera frame by reading from shared file (thread-safe, inter-process).

    Returns:
        OpenCV BGR image (numpy array) or None if no frame available
    """
    with _frame_lock:
        try:
            if not FRAME_FILE_OAK2.exists():
                return None

            # Read JPEG file
            with open(FRAME_FILE_OAK2, 'rb') as f:
                buffer = f.read()

            # Decode JPEG
            frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None


def get_oak2_frame_bytes() -> Optional[bytes]:
    """
    Get the latest oak2 camera frame as JPEG bytes (optimized for Flask streaming).

    Returns:
        JPEG bytes or None if no frame available
    """
    try:
        if not FRAME_FILE_OAK2.exists():
            return None

        with open(FRAME_FILE_OAK2, 'rb') as f:
            return f.read()
    except Exception:
        return None


def set_inference_active(active: bool) -> None:
    """
    Set or clear the inference active flag.

    When inference is active, the Flask GUI's oak2 camera updater will NOT overwrite
    the processed frames with raw camera feed.

    Args:
        active: True to signal inference is running, False to clear the flag
    """
    try:
        if active:
            # Create flag file
            INFERENCE_ACTIVE_FLAG.touch()
        else:
            # Remove flag file
            if INFERENCE_ACTIVE_FLAG.exists():
                INFERENCE_ACTIVE_FLAG.unlink()
    except Exception:
        pass  # Silently fail


def is_inference_active() -> bool:
    """
    Check if inference is currently active.

    Returns:
        True if inference is running (Flask should not overwrite frames)
    """
    return INFERENCE_ACTIVE_FLAG.exists()


def enable_alignment() -> None:
    """
    Enable hole alignment - tells the alignment service to start aligning.

    This should be called by navigation_manager when the robot reaches a waypoint
    and needs fine alignment before deploying the dipbob.
    """
    try:
        ALIGNMENT_ENABLED_FLAG.touch()
    except Exception:
        pass  # Silently fail


def disable_alignment() -> None:
    """
    Disable hole alignment - tells the alignment service to stop aligning.

    This should be called after alignment is complete or when moving between waypoints.
    """
    try:
        if ALIGNMENT_ENABLED_FLAG.exists():
            ALIGNMENT_ENABLED_FLAG.unlink()
    except Exception:
        pass  # Silently fail


def is_alignment_enabled() -> bool:
    """
    Check if hole alignment is currently enabled.

    Returns:
        True if alignment should be active (robot should align with collar)
    """
    return ALIGNMENT_ENABLED_FLAG.exists()
