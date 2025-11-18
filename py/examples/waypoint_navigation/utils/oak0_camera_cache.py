#!/usr/bin/env python3
"""
Shared camera frame cache for oak0 camera (farm-ng camera service).

Provides inter-process communication between oak0 camera stream and Flask GUI.
Uses file-based shared memory for efficient frame transfer.
"""

import numpy as np
import cv2
import threading
from pathlib import Path
from typing import Optional

# Shared frame file location for oak0
FRAME_FILE_OAK0 = Path("/tmp/amiga_oak0_frame.jpg")
_frame_lock = threading.Lock()


def set_oak0_frame(frame: np.ndarray) -> None:
    """
    Set the latest oak0 camera frame by writing to shared file (thread-safe, inter-process).

    Args:
        frame: OpenCV BGR image (numpy array)
    """
    if frame is None:
        return

    with _frame_lock:
        try:
            # Encode as JPEG for efficiency
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Write atomically using temp file + rename
            temp_file = FRAME_FILE_OAK0.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(buffer.tobytes())
            temp_file.replace(FRAME_FILE_OAK0)
        except Exception as e:
            # Silently fail to avoid disrupting camera processing
            pass


def get_oak0_frame() -> Optional[np.ndarray]:
    """
    Get the latest oak0 camera frame by reading from shared file (thread-safe, inter-process).

    Returns:
        OpenCV BGR image (numpy array) or None if no frame available
    """
    with _frame_lock:
        try:
            if not FRAME_FILE_OAK0.exists():
                return None

            # Read JPEG file
            with open(FRAME_FILE_OAK0, 'rb') as f:
                buffer = f.read()

            # Decode JPEG
            frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None


def get_oak0_frame_bytes() -> Optional[bytes]:
    """
    Get the latest oak0 camera frame as JPEG bytes (optimized for Flask streaming).

    Returns:
        JPEG bytes or None if no frame available
    """
    try:
        if not FRAME_FILE_OAK0.exists():
            return None

        with open(FRAME_FILE_OAK0, 'rb') as f:
            return f.read()
    except Exception:
        return None
