#!/usr/bin/env python3
# detectionPlot.py — Host-side YOLOE detection with spatial coordinates

import cv2, time, math, sys, os
import depthai as dai
import numpy as np

# Check if headless mode is enabled (for Flask integration) - MUST be before matplotlib imports
HEADLESS_MODE = os.getenv('DETECTION_HEADLESS', '0') == '1'

import matplotlib
if HEADLESS_MODE:
    matplotlib.use('Agg')  # Non-interactive backend
    print("Running in headless mode (no matplotlib window)")
else:
    print("Running with matplotlib display")

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import json
import socket
import os
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Pose3F64
from farm_ng_core_pybind import Rotation3F64

# YOLO model
try:
    from ultralytics import YOLOE
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.pose_cache import get_latest_pose, set_latest_pose
from utils.camera_frame_cache import set_latest_frame

# IMU-based tilt compensation
from imu_tilt_compensation import IMUTiltCompensation


# ---------------- Model / rates ----------------
# MODEL_PATH = Path(__file__).parent / "collarDetectionV2.engine"
# MODEL_PATH = Path(__file__).parent / "White_Barrel.onnx"
# MODEL_PATH = Path(__file__).parent / "yoloe-11s-seg.pt"
# MODEL_PATH = Path(__file__).parent / "yoloe-11s-seg.engine"
MODEL_PATH = Path(__file__).parent / "best1.engine"  # Using ONNX with GPU for 3-5x speedup
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

FPS = 15  # Reduced from 15 to further limit CPU usage
YOLO_PROCESS_EVERY_N_FRAMES = 5  # Only run YOLO inference every Nth frame (reduces CPU by ~50%)

# ---------------- Camera / FOV ----------------
USE_RGB_FOV     = True
RGB_HFOV_DEG    = 95     # per your spec
RGB_VFOV_DEG    = 72     # vertical FOV used in projector/overlays
STEREO_HFOV_DEG = 127    # keep if you ever switch to stereo FOV overlays

# ---- Camera tilt & height (with IMU compensation) ----
CAM_HEIGHT_M = 0.880
TILT_DEG     = 30.0      # +30° pitch DOWN (nominal, when robot is level)
theta = math.radians(TILT_DEG)

# Initialize IMU tilt compensation
# NOTE: Set ENABLE_IMU_COMPENSATION = False if IMU readings are incorrect
ENABLE_IMU_COMPENSATION = False  # Enabled after calibration
imu_compensator = IMUTiltCompensation(nominal_pitch_deg=TILT_DEG, nominal_roll_deg=0.0)

# IMU calibration baseline (calibrated on level ground)
IMU_BASELINE_PITCH = -64.8  # Stable reading on level ground
IMU_BASELINE_ROLL = -0.9    # Stable reading on level ground

# Cone/Hole goal
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
GOAL_ADDR = ("127.0.0.1", 41234)
_last = (0.0, 0.0, 0.0)

# Camera frame: +X right, +Y down, +Z forward. Level camera => up = [0, -1, 0].
# With forward tilt, up tilts toward +Z: [0, -cos(theta), -sin(theta)].
# NOTE: N_UP_CAM will be updated dynamically from IMU data in the main loop
N_UP_CAM = np.array([0.0, -math.cos(theta), -math.sin(theta)], dtype=float)
N_UP_CAM /= (np.linalg.norm(N_UP_CAM) + 1e-9)


def get_current_up_vector():
    """Get current up vector from IMU compensator or default."""
    return imu_compensator.get_up_vector_camera_frame()

# ---------------- Plot & display config ----------------
KEEP_HISTORY   = False
MAX_RANGE_M_Z  = 10.0   # forward axis [m] for scatter view bounds
MAX_RANGE_M_X  = 5.0    # lateral axis [m] for scatter view bounds
PLOT_REFRESH_S = 1  # Increased from 0.03 to reduce matplotlib overhead

# ---------------- Range arcs (meters) ----------------
ARCS_METERS = [1.0, 2.0, 3.0, 4.0, 6.0]

# ---------------- Waypoint (camera-frame) ----------------
# DepthAI camera frame: +X right, +Y down, +Z forward (meters)
WAYPOINT_X_M = -0.35
WAYPOINT_Y_M =  0.35
WAYPOINT_Z_M =  2.32

# OPTIONAL: lock to a specific device (MxID or name). Leave as "" to use default device.
TARGET_DEVICE  = "14442C1001A528D700"  # OAK1
# TARGET_DEVICE = "184430100194630E00" #OAK0

# Depth filtering
DEPTH_LOWER_MM = 100
DEPTH_UPPER_MM = 5000

# ---------------- Depth utilities ----------------
def get_depth_at_point(depth_frame: np.ndarray, x_norm: float, y_norm: float,
                       radius: int = 3) -> Optional[float]:
    """
    Get depth value (mm) at normalized coordinates (0-1).
    Uses median of small region around point for robustness.
    Reduced from radius=5 to radius=3 for better spatial accuracy (7x7 vs 11x11 pixels).
    """
    h, w = depth_frame.shape
    x_px = int(x_norm * w)
    y_px = int(y_norm * h)

    # Clamp to frame bounds
    x_px = np.clip(x_px, radius, w - radius - 1)
    y_px = np.clip(y_px, radius, h - radius - 1)

    # Get median depth in small region
    roi = depth_frame[y_px-radius:y_px+radius+1, x_px-radius:x_px+radius+1]
    roi_valid = roi[roi > 0]  # Filter out invalid (0) depth values

    if len(roi_valid) == 0:
        return None

    return float(np.median(roi_valid))


def pixel_to_camera_coords(x_norm: float, y_norm: float, depth_mm: float,
                          img_w: int, img_h: int, intrinsics: dict) -> np.ndarray:
    """
    Convert normalized pixel coords + depth to 3D camera coords (meters).
    Camera frame: +X right, +Y down, +Z forward
    """
    # Convert to pixel coordinates
    u = x_norm * img_w
    v = y_norm * img_h

    # Use actual camera intrinsics from calibration
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    # Depth in meters
    z = depth_mm / 1000.0

    # Unproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z], dtype=float)


# ---------------- YOLO Detection ----------------
class HostYOLODetector:
    """YOLOv8/v11 detection running on host GPU/CPU."""

    def __init__(self, model_path: Path, conf: float = 0.3, iou: float = 0.5):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics required. Install: pip install ultralytics")

        self.model = YOLOE(str(model_path))
        self.conf = conf
        self.iou = iou
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Run detection on frame, return list of detections."""
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)

        detections = []
        h, w = frame.shape[:2]

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Only keep class 0 detections (collar class from custom training)
                if cls_id != 0:
                    continue

                # Handle class name safely
                if cls_id in self.class_names:
                    label_name = self.class_names[cls_id]
                else:
                    label_name = f"class_{cls_id}"

                xmin, ymin, xmax, ymax = xyxy
                detections.append({
                    'label_id': cls_id,
                    'label_name': label_name,
                    'confidence': conf,
                    'xmin': xmin / w,
                    'ymin': ymin / h,
                    'xmax': xmax / w,
                    'ymax': ymax / h,
                })

        return detections


# ---------------- Detection with Depth ----------------
class SpatialDetection:
    """Detection with 3D coordinates from depth."""

    def __init__(self, detection: dict, spatial_coords: np.ndarray):
        self.label = detection['label_id']
        self.label_name = detection['label_name']
        self.confidence = detection['confidence']
        self.xmin = detection['xmin']
        self.ymin = detection['ymin']
        self.xmax = detection['xmax']
        self.ymax = detection['ymax']

        # Spatial coordinates (camera frame, meters)
        self.spatialCoordinates = type('obj', (object,), {
            'x': spatial_coords[0] * 1000,  # Convert to mm for compatibility
            'y': spatial_coords[1] * 1000,
            'z': spatial_coords[2] * 1000,
        })()


def add_spatial_to_detections(detections: List[dict], depth_frame: np.ndarray,
                               img_w: int, img_h: int, intrinsics: dict,
                               use_geometric_correction: bool = False) -> List[SpatialDetection]:
    """Add 3D spatial coordinates to detections using depth map."""
    spatial_dets = []
    closest_det_info = None  # Track closest detection for logging

    # Import geometric correction if needed
    if use_geometric_correction:
        try:
            from geometric_depth_correction import pixel_to_camera_coords_corrected
            geometric_available = True
            print("[GEOM] ✓ Geometric depth correction enabled (110mm x 118mm collar, 50% blend)")
        except ImportError as e:
            print(f"⚠️  geometric_depth_correction not available: {e}")
            print("⚠️  Using sensor depth only")
            geometric_available = False
    else:
        geometric_available = False
        # print("[GEOM] Geometric correction disabled by parameter")

    for det in detections:
        # Horizontal center
        x_center = (det['xmin'] + det['xmax']) / 2.0

        # Vertical: measure at top 20% of pipe (80% from bottom)
        # This gives more consistent depth for tall pipes vs short pipes
        bbox_height = det['ymax'] - det['ymin']
        y_measurement = det['ymin'] + 0.2 * bbox_height  # 20% down from top

        # Get depth at this point
        depth_mm = get_depth_at_point(depth_frame, x_center, y_measurement)

        if depth_mm is None or depth_mm < DEPTH_LOWER_MM or depth_mm > DEPTH_UPPER_MM:
            continue  # Skip detections with invalid depth

        # Apply geometric depth correction using known collar dimensions
        if geometric_available:
            coords_3d, diagnostics = pixel_to_camera_coords_corrected(
                det['xmin'], det['ymin'], det['xmax'], det['ymax'],
                depth_mm, img_w, img_h, intrinsics,
                apply_correction=True,
                blend_weight=0.5  # 50/50 blend between geometric and sensor
            )
            # Track closest detection for logging
            if diagnostics.get('correction_applied'):
                distance = coords_3d[2]  # z-coordinate is forward distance
                if closest_det_info is None or distance < closest_det_info['distance']:
                    closest_det_info = {
                        'distance': distance,
                        'diagnostics': diagnostics
                    }
        else:
            # Fallback to standard depth (using measurement point)
            coords_3d = pixel_to_camera_coords(x_center, y_measurement, depth_mm,
                                               img_w, img_h, intrinsics)

        spatial_dets.append(SpatialDetection(det, coords_3d))

    # Log correction for closest detection only
    if closest_det_info is not None:
        diagnostics = closest_det_info['diagnostics']
        correction_cm = diagnostics.get('correction_cm', 0)
        if abs(correction_cm) > 5:  # Log if correction > 5cm
            bbox_w_px = diagnostics.get('bbox_width_px', 0)
            bbox_h_px = diagnostics.get('bbox_height_px', 0)
            print(f"[GEOM] Depth corrected by {correction_cm:+.1f}cm "
                  f"(sensor: {diagnostics['depth_sensor_m']:.2f}m → "
                  f"corrected: {diagnostics['depth_corrected_m']:.2f}m) "
                  f"bbox: {bbox_w_px:.0f}x{bbox_h_px:.0f}px")

    return spatial_dets


# ---------------- Helpers ----------------
def project_point_to_pixels(x_m, y_m, z_m, img_w, img_h, hfov_deg, vfov_deg=None):
    """
    Pinhole projection using FOV (approx). Returns (u, v) pixels in the image.
    Assumes principal point at image center, square pixels.
    If vfov_deg is None, derive it from aspect ratio assuming fx == fy.
    """
    if z_m <= 0:
        return None
    if vfov_deg is None:
        vfov_deg = math.degrees(2.0 * math.atan((img_h/img_w) * math.tan(math.radians(hfov_deg/2.0))))
    fx = (img_w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = (img_h / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
    cx, cy = img_w / 2.0, img_h / 2.0
    u = cx + fx * (x_m / z_m)
    v = cy - fy * (y_m / z_m)  # image v grows downward
    return int(round(u)), int(round(v))

def draw_waypoint_icon(img, u, v, color=(0, 255, 255)):
    """Draw a simple star marker + label at (u,v)."""
    h, w = img.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.drawMarker(img, (u, v), color, markerType=cv2.MARKER_STAR, markerSize=24, thickness=2)
        cv2.putText(img, "WP", (u+8, v-8), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color, 1, cv2.LINE_AA)
    else:
        cx, cy = w//2, h//2
        dir_vec = np.array([u - cx, v - cy], dtype=float)
        n = np.linalg.norm(dir_vec)
        if n < 1e-6: return
        dir_vec /= (n + 1e-9)
        edge_pt = (int(cx + dir_vec[0]*min(w, h)*0.45), int(cy + dir_vec[1]*min(w, h)*0.45))
        cv2.arrowedLine(img, (cx, cy), edge_pt, color, 2, tipLength=0.2)
        cv2.putText(img, "WP off FOV", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color, 1, cv2.LINE_AA)

def _ground_basis_from_up(n_up_cam: np.ndarray):
    """
    Build an orthonormal basis (fwd, left) on the ground plane (normal = n_up_cam).
    'fwd' is the camera's forward axis projected into the ground plane.
    """
    n = n_up_cam / (np.linalg.norm(n_up_cam) + 1e-9)
    ez = np.array([0.0, 0.0, 1.0], dtype=float)  # camera +Z (forward)
    ex = np.array([1.0, 0.0, 0.0], dtype=float)  # camera +X (right)

    # Project camera forward onto plane; fall back to X if near-collinear with normal
    f = ez - (ez @ n) * n
    if np.linalg.norm(f) < 1e-6:
        f = ex - (ex @ n) * n
    f /= (np.linalg.norm(f) + 1e-9)

    # Left vector: n x f  (so that (f, left, n) is right-handed in camera coords)
    left = np.cross(n, f)
    left /= (np.linalg.norm(left) + 1e-9)

    # Ground point directly below the camera (camera's ground projection)
    center = -n * CAM_HEIGHT_M
    return center, f, left

def draw_wedge_arcs_on_image(img, radii_m, hfov_deg, vfov_deg=None, color=(200,200,200)):
    """
    Draw arc portions centered at the camera origin (0,0,0) in the X–Z plane (y=0),
    clipped to the camera HFOV, exactly like the scatter's wedge arcs.
    """
    h, w = img.shape[:2]
    half = math.radians(hfov_deg / 2.0)
    thetas = np.linspace(-half, half, 181)  # angles inside the wedge

    for r in radii_m:
        pts = []
        for th in thetas:
            x = r * math.sin(th)
            y = 0.0               # same “flat slice” as the scatter wedges
            z = r * math.cos(th)
            uv = project_point_to_pixels(x, y, z, w, h, hfov_deg, vfov_deg)
            if uv is not None:
                u, v = uv
                if 0 <= u < w and 0 <= v < h:
                    pts.append([u, v])

        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=False,
                          color=color, thickness=1, lineType=cv2.LINE_AA)
            # Label near straight-ahead point (theta = 0 => x=0, z=r)
            mid_uv = project_point_to_pixels(0.0, 0.0, r, w, h, hfov_deg, vfov_deg)
            if mid_uv is not None:
                um, vm = mid_uv
                if 0 <= um < w and 0 <= vm < h:
                    cv2.putText(img, f"{int(r)} m", (um+6, vm-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_ground_arcs_on_image(img, radii_m, n_up_cam, hfov_deg, vfov_deg=None, color=(200,200,200)):
    h, w = img.shape[:2]
    center, fwd, left = _ground_basis_from_up(n_up_cam)

    # --- Shift origin to the optical-axis ⟂ ground intersection ---
    n  = n_up_cam / (np.linalg.norm(n_up_cam) + 1e-9)
    ez = np.array([0.0, 0.0, 1.0], dtype=float)

    denom = float(np.dot(n, ez))
    if abs(denom) > 1e-6:
        t_hit = float(np.dot(n, center)) / denom
        P_hit = t_hit * ez
        D_forward = float(np.dot(P_hit - center, fwd))
        ARC_FORWARD_BIAS_M = 0.0
        center = center + fwd * (D_forward + ARC_FORWARD_BIAS_M)

    thetas = np.linspace(-math.pi/2, math.pi/2, 181)
    for r in radii_m:
        pts_px = []
        for th in thetas:
            P = center + r * (math.cos(th)*fwd + math.sin(th)*left)
            uv = project_point_to_pixels(P[0], P[1], P[2], w, h, hfov_deg, vfov_deg)
            if uv is None: continue
            u, v = uv
            if 0 <= u < w and 0 <= v < h:
                pts_px.append([u, v])
        if len(pts_px) >= 2:
            cv2.polylines(img, [np.array(pts_px, dtype=np.int32)], isClosed=False,
                          color=color, thickness=1, lineType=cv2.LINE_AA)
            Pmid = center + r * fwd
            uv_mid = project_point_to_pixels(Pmid[0], Pmid[1], Pmid[2], w, h, hfov_deg, vfov_deg)
            if uv_mid is not None and 0 <= uv_mid[0] < w and 0 <= uv_mid[1] < h:
                cv2.putText(img, f"{int(r)} m", (uv_mid[0]+6, uv_mid[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def ground_arcs_xz_points(radii_m, n_up_cam):
    center, fwd, left = _ground_basis_from_up(n_up_cam)
    thetas = np.linspace(-math.pi/2, math.pi/2, 181)
    curves = []
    for r in radii_m:
        Ps = np.array([center + r * (math.cos(th)*fwd + math.sin(th)*left) for th in thetas])
        curves.append((Ps[:,0], Ps[:,2]))  # (x, z)
    return curves

def gnd_to_nadir(p_cam_m, n_up_cam, cam_height_m):
    n = n_up_cam / (np.linalg.norm(n_up_cam) + 1e-9)
    _, fwd, left = _ground_basis_from_up(n)
    d_fwd  = float(np.dot(p_cam_m, fwd))
    d_left = float(np.dot(p_cam_m, left))
    gnd_dist = math.hypot(d_fwd, d_left)
    return gnd_dist, d_fwd, d_left

class DetectionAverager:
    """Rolling average filter for detection positions to reduce noise."""
    def __init__(self, window_size=5, max_age_seconds=1.0):
        self.window_size = window_size
        self.max_age_seconds = max_age_seconds
        self.detections = []  # List of (x_fwd, y_left, conf, timestamp)

    def add_detection(self, x_fwd, y_left, conf):
        """Add a new detection to the rolling window."""
        now = time.time()
        self.detections.append((x_fwd, y_left, conf, now))

        # Remove old detections beyond time window
        self.detections = [d for d in self.detections if (now - d[3]) < self.max_age_seconds]

        # Keep only most recent N detections
        if len(self.detections) > self.window_size:
            self.detections = self.detections[-self.window_size:]

    def get_averaged_position(self):
        """Get averaged position, or None if insufficient data."""
        if len(self.detections) < 3:  # Need at least 3 samples
            return None

        # Weight by confidence and recency
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        now = time.time()

        for x, y, conf, timestamp in self.detections:
            # Recency weight: newer detections weighted higher (exponential decay)
            age = now - timestamp
            recency_weight = math.exp(-age / 0.5)  # 0.5s time constant

            # Combined weight
            weight = conf * recency_weight
            total_weight += weight
            weighted_x += x * weight
            weighted_y += y * weight

        if total_weight < 0.01:
            return None

        avg_x = weighted_x / total_weight
        avg_y = weighted_y / total_weight
        avg_conf = sum(d[2] for d in self.detections) / len(self.detections)

        return avg_x, avg_y, avg_conf

    def clear(self):
        """Clear all detections."""
        self.detections = []

# Global detection averager
detection_averager = DetectionAverager(window_size=5, max_age_seconds=1.0)

def maybe_emit_cone_goal(x_fwd, y_left, conf, label_id, label_map, r_max=10.0, min_period=0.8):
    global _last  # <-- important

    # Add detection to rolling average
    detection_averager.add_detection(x_fwd, y_left, conf)

    # Get averaged position
    averaged = detection_averager.get_averaged_position()
    if averaged is None:
        return  # Not enough samples yet

    avg_x, avg_y, avg_conf = averaged

    r = math.hypot(avg_x, avg_y)
    now = time.time()

    # Emit averaged position (not instantaneous)
    if r <= r_max and (now - _last[2]) >= min_period:
        try:
            class_name = label_map[label_id]
        except Exception:
            class_name = str(label_id)
        msg = {
            "class_name": class_name,
            "class_id": int(label_id),
            "x_fwd_m": float(avg_x),      # Averaged!
            "y_left_m": float(avg_y),     # Averaged!
            "confidence": float(avg_conf),
            "stamp": now,
        }
        print(msg)
        try:
            sock.sendto(json.dumps(msg).encode("utf-8"), GOAL_ADDR)
            _last = (avg_x, avg_y, now)
        except Exception as e:
            print(f"[vision] UDP send failed: {e}")

def robot_to_world(xr: float, yr: float, yaw: float, dx: float, dy: float):
    c = math.cos(yaw); s = math.sin(yaw)
    X = xr + dx * c - dy * s
    Y = yr + dx * s + dy * c
    return X, Y

def handle_detection_msg(msg) -> None:
    pose = get_latest_pose()
    if pose is None:
        print("[vision] pose unavailable; cannot compute world coords")
        return

    dx = float(msg.x)  # forward (+x robot)
    dy = float(msg.y)  # right   (+y robot)

    Xw, Yw = robot_to_world(pose.x, pose.y, pose.yaw, dx, dy)
    print(f"[vision] obj world: X={Xw:.2f} m, Y={Yw:.2f} m  | robot @ ({pose.x:.2f}, {pose.y:.2f}), ψ={pose.yaw:.2f} rad")
    
        
# ---------------- Visualizer ----------------
class SpatialVisualizer:
    def __init__(self, n_up_cam):
        matplotlib.rcParams['toolbar'] = 'None'
        plt.ion()
        # Create figure with 1 row, 2 columns
        self.fig, (self.ax_img, self.ax) = plt.subplots(1, 2, num="OAK-D: Camera + Scatter", figsize=(16, 6))

        # Image subplot (left)
        self.ax_img.set_title("RGB Camera")
        self.ax_img.axis('off')
        self.img_display = None

        # Scatter subplot (right) - existing code
        self.ax.set_xlabel("Y (m)  [left +, right -]")
        self.ax.set_ylabel("X (m)  [forward +]")
        self.ax.set_xlim(-MAX_RANGE_M_X, MAX_RANGE_M_X)
        self.ax.set_ylim(0.0, MAX_RANGE_M_Z)
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self._draw_fov_wedge()
        self._draw_robot_rings()

        # Detections + optional history
        self.scatter_current = self.ax.scatter([], [], s=35)
        self.scatter_hist    = self.ax.scatter([], [], s=8, alpha=0.25) if KEEP_HISTORY else None
        self.hist_x, self.hist_z = [], []

        # Waypoint marker on the X–Z map (camera-frame coordinates)
        # self.waypoint_handle = self.ax.scatter([WAYPOINT_X_M], [WAYPOINT_Z_M], marker="X", s=90)

        self._last_plot = time.time()
        self.labelMap = []

        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

    def _draw_fov_wedge(self):
        hfov = RGB_HFOV_DEG if USE_RGB_FOV else STEREO_HFOV_DEG
        half = math.radians(hfov / 2.0)

        # FOV wedge boundaries (x = ± z * tan(half))
        z = np.linspace(0.0, MAX_RANGE_M_Z, 200)
        tanh = math.tan(half)
        xL = -tanh * z
        xR = +tanh * z
        xL = np.clip(xL, -MAX_RANGE_M_X, MAX_RANGE_M_X)
        xR = np.clip(xR, -MAX_RANGE_M_X, MAX_RANGE_M_X)

        # Wedge fill + edges
        self.ax.fill_betweenx(z, xL, xR, alpha=0.10, label=f"FOV {hfov:.1f}°")
        self.ax.plot(xL, z, linestyle="--")
        self.ax.plot(xR, z, linestyle="--")

        # Range arc portions originating at (0,0), clipped to the wedge
        thetas = np.linspace(-half, half, 361)
        for r in ARCS_METERS:
            x = r * np.sin(thetas)
            zz = r * np.cos(thetas)
            mask = (zz >= 0) & (np.abs(x) <= MAX_RANGE_M_X) & (zz <= MAX_RANGE_M_Z)
            if np.any(mask):
                self.ax.plot(x[mask], zz[mask], linestyle=":", alpha=0.7)
                if r <= MAX_RANGE_M_Z:
                    self.ax.text(0.05, r * 0.98, f"{int(r)} m", fontsize=9, alpha=0.8)

        self.ax.legend(loc="upper right")

    def _draw_robot_rings(self):
        """Draw concentric ground-distance rings centered at the robot origin."""
        for r in ARCS_METERS:
            th = np.linspace(-np.pi, np.pi, 361)
            y = r * np.sin(th)  # left/right
            x = r * np.cos(th)  # forward
            # Limit to current axis bounds (only draw what's visible)
            mask = (x >= 0.0) & (x <= MAX_RANGE_M_Z) & (np.abs(y) <= MAX_RANGE_M_X)
            if np.any(mask):
                self.ax.plot(y[mask], x[mask], linestyle=":", alpha=0.7)
                self.ax.text(0.05, r * 0.98, f"{int(r)} m", fontsize=9, alpha=0.8)
              
    def update_image(self, rgb_frame):
          """Update the RGB camera subplot with FPS counter."""
          # Draw FPS on frame (will be updated from main loop)
          frame_with_fps = rgb_frame.copy()
          cv2.putText(frame_with_fps, f"FPS: {self._current_fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

          # Convert BGR (OpenCV) to RGB (matplotlib)
          rgb = cv2.cvtColor(frame_with_fps, cv2.COLOR_BGR2RGB)

          if self.img_display is None:
              self.img_display = self.ax_img.imshow(rgb)
          else:
              self.img_display.set_data(rgb)
                
    def processDepthFrame(self, depthFrame):
        dds = depthFrame[::4]
        min_depth = 0 if np.all(dds == 0) else np.percentile(dds[dds != 0], 1)
        max_depth = np.percentile(dds, 99)
        depth8 = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        return cv2.applyColorMap(depth8, cv2.COLORMAP_HOT)

    def drawBBoxOnDepth(self, depthColor, det):
        roi = det.boundingBoxMapping.roi.denormalize(depthColor.shape[1], depthColor.shape[0])
        tl, br = roi.topLeft(), roi.bottomRight()
        cv2.rectangle(depthColor, (int(tl.x), int(tl.y)), (int(br.x), int(br.y)), (255,255,255), 1)
    
    def drawDetOnRgb(self, frame, det, W, H, color=(255, 255, 255)):
        x1 = int(det.xmin * W); x2 = int(det.xmax * W)
        y1 = int(det.ymin * H); y2 = int(det.ymax * H)

        p = np.array([
            det.spatialCoordinates.x, 
            det.spatialCoordinates.y, 
            det.spatialCoordinates.z
        ], dtype=float) / 1000.0  # -> meters (camera frame)

        # Convert to robot/world frame using farm-ng Poses
        robot_from_object = xfm.robot_pose_from_cam_point(p)
        v_r = np.array(robot_from_object.a_from_b.translation, dtype=float)

        
        d_robot_3d     = float(np.linalg.norm(v_r))
        d_robot_ground = float(math.hypot(v_r[0], v_r[1]))

        

        # Camera-frame readout 
        try:
            label = self.labelMap[det.label]
        except Exception:
            label = str(det.label)

        cv2.putText(frame, f"{label}", (x1+10,y1+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        # cv2.putText(frame, "{:.2f}".format(det.confidence*100), (x1+10,y1+35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"{d_robot_3d:.2f}m from bot",   (x1+10, y1+35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {p[0]:.2f} m", (x1+10,y1+50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {p[1]:.2f} m", (x1+10,y1+65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {p[2]:.2f} m", (x1+10,y1+80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)

    def update_plot(self, detections):
        # Robot-frame ground coords (for the scatter)
        ys_left, xs_fwd = [], []

        # Track the closest detection to emit (only one message per frame)
        # NOW: Emit for ALL detections (collars are class 0 / "object0")
        closest_detection = None
        closest_distance = float('inf')

        for det in detections:
            # Camera-frame point (meters)
            p_cam = np.array([
                det.spatialCoordinates.x,
                det.spatialCoordinates.y,
                det.spatialCoordinates.z
            ], dtype=float) / 1000.0

            # Skip invalid/behind-camera points
            if not np.all(np.isfinite(p_cam)) or p_cam[2] <= 0.0:
                continue

            # Transform to robot frame: v_r = [X_fwd, Y_left, Z_up]
            robot_from_object = xfm.robot_pose_from_cam_point(p_cam)
            v_r = np.array(robot_from_object.a_from_b.translation, dtype=float)
            x_fwd, y_left = v_r[0], v_r[1]

            # Track closest detection for emission (all detections, not just specific class)
            distance = math.hypot(x_fwd, y_left)
            if distance < closest_distance:
                closest_distance = distance
                closest_detection = (x_fwd, y_left, det.confidence, det.label)

            # Keep plotting ALL detections
            if math.isfinite(x_fwd) and math.isfinite(y_left):
                y_plot = -y_left   # flip left/right for the scatter view
                ys_left.append(y_plot)
                xs_fwd.append(x_fwd)
                if KEEP_HISTORY:
                    self.hist_x.append(y_plot)
                    self.hist_z.append(x_fwd)

        # Emit only the closest detection (if any detection was found this frame)
        if closest_detection is not None:
            x_fwd, y_left, conf, label = closest_detection
            maybe_emit_cone_goal(x_fwd, y_left, conf, label, self.labelMap)

        now = time.time()
        if now - self._last_plot >= PLOT_REFRESH_S:
            self.scatter_current.set_offsets(np.c_[ys_left, xs_fwd])
            if KEEP_HISTORY and self.hist_x:
                self.scatter_hist.set_offsets(np.c_[self.hist_x, self.hist_z])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self._last_plot = now

# ---------------- Transforms (camera -> robot via Pose/Isometry) ----------------
class Transforms:
    """
    Map OAK detections from camera frame -> robot (world≡robot) frame using farm-ng Pose3F64.
    DepthAI camera frame: +X right, +Y down, +Z forward
    Robot (NWU):          +X forward, +Y left, +Z up
    """
    def __init__(self, camera_offsets_path: Path):
        with open(camera_offsets_path, "r") as f:
            data = json.load(f)

        # Translation: camera origin in the robot (NWU) frame, meters
        t = data["translation"]
        tx, ty, tz = float(t["x"]), float(t["y"]), float(t["z"])

        # READ RPY FROM JSON FIRST (fixes NameError)
        rpy = data.get("rotation_rpy_deg", {"roll": 0.0, "pitch": 30.0, "yaw": 0.0})
        roll_deg  = float(rpy.get("roll",  0.0))
        pitch_deg = float(rpy.get("pitch", 30.0))  # +30 means camera pitched DOWN
        yaw_deg   = float(rpy.get("yaw",   0.0))

        # Fixed axis alignment (DepthAI cam -> NWU robot):
        #   Xr <- Zc,  Yr <- -Xc,  Zr <- -Yc
        # This equals Rx(-90°) * Ry(+90°) (apply rightmost first).
        R_align = Rotation3F64.Rx(math.radians(-90.0)) * Rotation3F64.Ry(math.radians(+90.0))

        # Mount correction in the CAMERA frame:
        # Down-tilt is a rotation about camera X (right). Positive "pitch" (down) => Rx(-pitch).
        # (You can extend with yaw/roll if needed; kept simple & physically correct here.)
        R_mount_cam = Rotation3F64.Rx(math.radians(-pitch_deg))

        # Net camera->robot rotation
        R_cam_to_robot = R_align * R_mount_cam

        # Store components for dynamic IMU updates
        self.tx, self.ty, self.tz = tx, ty, tz
        self.R_align = R_align
        self.nominal_pitch_deg = pitch_deg

        # Pose: robot_from_camera (will be updated dynamically with IMU)
        self.robot_from_camera = Pose3F64(
            a_from_b=Isometry3F64([tx, ty, tz], R_cam_to_robot),
            frame_a="robot",
            frame_b="camera",
        )

    def update_with_imu(self, effective_pitch_deg: float, effective_roll_deg: float):
        """Update camera-to-robot transform with IMU-compensated tilt angles."""
        # Mount correction in CAMERA frame with IMU compensation
        R_mount_cam = (Rotation3F64.Rx(math.radians(-effective_pitch_deg)) *
                       Rotation3F64.Rz(math.radians(+effective_roll_deg)))

        # Net camera->robot rotation
        R_cam_to_robot = self.R_align * R_mount_cam

        # Update pose
        self.robot_from_camera = Pose3F64(
            a_from_b=Isometry3F64([self.tx, self.ty, self.tz], R_cam_to_robot),
            frame_a="robot",
            frame_b="camera",
        )

    def robot_pose_from_cam_point(self, p_cam_m: np.ndarray) -> Pose3F64:
        """
        Represent the detection as a pose in the camera frame (identity rotation),
        then compose via robot_from_camera to get object in the robot frame.
        """
        camera_from_object = Pose3F64(
            a_from_b=Isometry3F64(p_cam_m.tolist(), Rotation3F64()),
            frame_a="camera",
            frame_b="object",
        )
        return self.robot_from_camera * camera_from_object


# ---------------- Build pipeline (host-side detection) ----------------
def create_pipeline():
    """Create OAK-D pipeline that streams RGB + Depth + IMU to host."""
    if TARGET_DEVICE:
        device = dai.Device(TARGET_DEVICE)
        pipeline = dai.Pipeline(device)
    else:
        pipeline = dai.Pipeline()
        device = None

    # Camera nodes
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoL = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoR = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setExtendedDisparity(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB
    if pipeline.getDefaultDevice().getPlatform() == dai.Platform.RVC2:
        stereo.setOutputSize(640, 400)

    monoL.requestOutput((640, 400)).link(stereo.left)
    monoR.requestOutput((640, 400)).link(stereo.right)

    # RGB output (full res for YOLO)
    xoutRgb = camRgb.requestOutput((IMG_SIZE, IMG_SIZE))

    # Depth output
    xoutDepth = stereo.depth

    # IMU node (for tilt compensation) - DepthAI v3 API
    imu = pipeline.create(dai.node.IMU)

    # Enable ROTATION_VECTOR sensor at 100 Hz (provides orientation quaternion)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # Create output queues (matching v3 pattern used for RGB/Depth)
    qRgb = xoutRgb.createOutputQueue(maxSize=1, blocking=False)
    qDepth = xoutDepth.createOutputQueue(maxSize=1, blocking=False)
    qImu = imu.out.createOutputQueue(maxSize=50, blocking=False)

    return pipeline, qRgb, qDepth, qImu, device


# Load YOLO model
print("\n" + "="*70)
print("HOST-SIDE YOLOE DETECTION WITH SPATIAL COORDINATES")
print("="*70)
print(f"Model:  {MODEL_PATH}")
print(f"Conf:   {CONF_THRESHOLD}")
print("="*70 + "\n")

if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    sys.exit(1)

detector = HostYOLODetector(MODEL_PATH, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
print(f"✓ Loaded YOLO model with {len(detector.class_names)} classes")

# Create vision_running flag file to signal navigation system
VISION_FLAG_FILE = Path(__file__).parent.parent / ".vision_running"
try:
    VISION_FLAG_FILE.touch()
    print(f"✓ Created vision flag: {VISION_FLAG_FILE}")
except Exception as e:
    print(f"⚠️  Could not create vision flag: {e}")

# Create pipeline
pipeline, qRgb, qDepth, qImu, device = create_pipeline()

# Get camera intrinsics
if device is None:
    device = pipeline.getDefaultDevice()

# Check IMU type
imuType = device.getConnectedIMU()
print(f"\n📡 IMU type: {imuType}")
if imuType == "BNO086":
    if ENABLE_IMU_COMPENSATION:
        print("✓ IMU tilt compensation ENABLED (pitch + roll)")
        print(f"   Baseline calibration: pitch={IMU_BASELINE_PITCH:.1f}°, roll={IMU_BASELINE_ROLL:.1f}°")
    else:
        print("⚠️  IMU tilt compensation DISABLED")
        print("   Using fixed 30° tilt (see ENABLE_IMU_COMPENSATION in code)")
        print("   IMU readings will be logged but not applied")
else:
    print(f"⚠️  IMU type {imuType} detected - rotation vector may not be available")
    print("   Using fixed tilt mode")

calibData = device.readCalibration()
intrinsics_matrix = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, IMG_SIZE, IMG_SIZE)

intrinsics = {
    'fx': intrinsics_matrix[0][0],
    'fy': intrinsics_matrix[1][1],
    'cx': intrinsics_matrix[0][2],
    'cy': intrinsics_matrix[1][2],
}

print(f"\n📷 Camera Intrinsics (from calibration):")
print(f"   fx: {intrinsics['fx']:.2f}, fy: {intrinsics['fy']:.2f}")
print(f"   cx: {intrinsics['cx']:.2f}, cy: {intrinsics['cy']:.2f}")

# Visualizer
# Only create visualizer if not in headless mode
vis = SpatialVisualizer(N_UP_CAM) if not HEADLESS_MODE else None
if vis is not None:
    vis.labelMap = list(detector.class_names.values())

# Transforms
xfm = Transforms(Path(os.environ.get("CAMERA_OFFSET_CONFIG", "camera_offset_config.json")))

# ---------------- Run (latest-only draining) ----------------
def drain_latest(q):
    last = None
    while q.has():
        last = q.get()
    return last

print("\n🚀 Starting detection loop...")
print("   Close matplotlib window to exit\n")

pipeline.start()
with pipeline:
    # Initialize IMU logging variables (local to this scope)
    last_calibrated_pitch = 0.0
    last_calibrated_roll = 0.0
    last_effective_pitch = TILT_DEG
    last_effective_roll = 0.0

    latestDepth = None
    latestRgb   = None

    hfov_deg = RGB_HFOV_DEG if USE_RGB_FOV else STEREO_HFOV_DEG
    img_w = img_h = None

    # FPS tracking for actual camera frames
    frame_count = 0
    fps_start_time = time.time()

    # Rate limiting to reduce CPU load
    target_process_fps = FPS  # Use FPS constant from top of file
    min_frame_interval = 1.0 / target_process_fps
    last_process_time = time.time()

    # Initialize detections list for frame skipping (reuse last detections when not running inference)
    detections = []

    try:
        while pipeline.isRunning():
            # Rate limiting: Skip processing if we're going too fast
            current_time = time.time()
            time_since_last_process = current_time - last_process_time

            if time_since_last_process < min_frame_interval:
                # Sleep briefly to avoid busy-waiting and yield CPU to other processes
                time.sleep(0.001)  # 1ms sleep to prevent tight loop
                continue

            last_process_time = current_time

            # Update IMU data (tilt compensation)
            imuData = drain_latest(qImu)
            if imuData is not None:
                imuPackets = imuData.packets
                # Use the latest packet in this batch
                if len(imuPackets) > 0:
                    rVvalues = imuPackets[-1].rotationVector
                    # Update compensator with quaternion (up vector updated internally)
                    imu_compensator.update_from_quaternion(
                        rVvalues.i, rVvalues.j, rVvalues.k, rVvalues.real
                    )

                    # Apply IMU compensation only if enabled
                    if ENABLE_IMU_COMPENSATION:
                        # Apply baseline calibration offset
                        diag = imu_compensator.get_diagnostics()
                        calibrated_pitch = diag['imu_pitch_deg'] - IMU_BASELINE_PITCH
                        calibrated_roll = diag['imu_roll_deg'] - IMU_BASELINE_ROLL

                        # Update effective angles with calibrated values
                        effective_pitch = TILT_DEG - calibrated_pitch
                        effective_roll = 0.0 - calibrated_roll

                        # Update camera-to-robot transform with IMU-compensated angles
                        xfm.update_with_imu(effective_pitch, effective_roll)

                        # Store calibrated values for logging
                        last_calibrated_pitch = calibrated_pitch
                        last_calibrated_roll = calibrated_roll
                        last_effective_pitch = effective_pitch
                        last_effective_roll = effective_roll

            depthMsg = drain_latest(qDepth)
            if depthMsg is not None:
                latestDepth = depthMsg.getFrame()

            rgbMsg = drain_latest(qRgb)
            if rgbMsg is not None:
                latestRgb = rgbMsg.getCvFrame()
                if img_w is None:
                    img_h, img_w = latestRgb.shape[:2]

                # Count every camera frame received
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed > 0 and vis is not None:
                    vis._current_fps = frame_count / elapsed

            if latestRgb is None or latestDepth is None:
                continue

            # Frame skipping for YOLO inference (reduce CPU load)
            # Only run inference every Nth frame, reuse last detections otherwise
            should_run_inference = (frame_count % YOLO_PROCESS_EVERY_N_FRAMES) == 0

            if should_run_inference:
                # Run YOLO detection on host
                detections = detector.detect(latestRgb)
            # else: reuse detections from previous frame (already stored in 'detections')

            # Add spatial coordinates from depth using actual calibration
            h, w = latestRgb.shape[:2]
            spatial_detections = add_spatial_to_detections(
                detections, latestDepth, w, h, intrinsics
            )

            # Draw detections on frame
            display_frame = latestRgb.copy()
            for det in spatial_detections:
                x1 = int(det.xmin * w)
                y1 = int(det.ymin * h)
                x2 = int(det.xmax * w)
                y2 = int(det.ymax * h)

                # Draw box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label with spatial info
                p_cam = np.array([
                    det.spatialCoordinates.x,
                    det.spatialCoordinates.y,
                    det.spatialCoordinates.z
                ]) / 1000.0  # back to meters

                robot_pose = xfm.robot_pose_from_cam_point(p_cam)
                v_r = np.array(robot_pose.a_from_b.translation)
                dist = float(np.linalg.norm(v_r))

                label = f"{det.label_name} {det.confidence:.2f} {dist:.2f}m"
                cv2.putText(display_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Share frame with Flask GUI
            set_latest_frame(display_frame)

            # Export detection data for Flask scatter plot
            detection_data = []
            for det in spatial_detections:
                p_cam = np.array([
                    det.spatialCoordinates.x,
                    det.spatialCoordinates.y,
                    det.spatialCoordinates.z
                ]) / 1000.0

                robot_pose = xfm.robot_pose_from_cam_point(p_cam)
                v_r = np.array(robot_pose.a_from_b.translation)

                detection_data.append({
                    'x': float(v_r[0]),  # Forward
                    'y': float(v_r[1]),  # Left/Right
                    'z': float(v_r[2]),  # Up/Down
                    'label': det.label_name,
                    'confidence': float(det.confidence),
                    'distance': float(np.linalg.norm(v_r))
                })

            # Write to shared file for Flask (throttled to reduce I/O load)
            # Only write every 15 frames (1 Hz at 15 FPS) to reduce disk I/O
            if frame_count % 15 == 0:
                try:
                    with open('/tmp/amiga_detections.json', 'w') as f:
                        json.dump(detection_data, f)
                except Exception:
                    pass

            # ALWAYS emit closest detection (regardless of headless mode)
            # Find closest detection and send UDP message
            closest_detection = None
            closest_distance = float('inf')
            for det in spatial_detections:
                p_cam = np.array([
                    det.spatialCoordinates.x,
                    det.spatialCoordinates.y,
                    det.spatialCoordinates.z
                ]) / 1000.0

                robot_pose = xfm.robot_pose_from_cam_point(p_cam)
                v_r = np.array(robot_pose.a_from_b.translation)
                x_fwd, y_left = v_r[0], v_r[1]

                distance = math.hypot(x_fwd, y_left)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_detection = (x_fwd, y_left, det.confidence, det.label)

            # Emit the closest detection
            if closest_detection is not None:
                x_fwd, y_left, conf, label = closest_detection
                maybe_emit_cone_goal(x_fwd, y_left, conf, label, list(detector.class_names.values()))

                # Log IMU tilt compensation (every 10th detection to avoid spam)
                if frame_count % 10 == 0:
                    diag = imu_compensator.get_diagnostics()
                    if abs(diag['imu_pitch_deg']) > 1.0 or abs(diag['imu_roll_deg']) > 1.0:
                        if ENABLE_IMU_COMPENSATION:
                            # Show calibrated values (what's actually being used)
                            print(f"[IMU-APPLIED] raw: pitch={diag['imu_pitch_deg']:+.1f}° roll={diag['imu_roll_deg']:+.1f}° | "
                                  f"calibrated Δ: pitch={last_calibrated_pitch:+.1f}° roll={last_calibrated_roll:+.1f}° → "
                                  f"cam effective: {last_effective_pitch:.1f}°/{last_effective_roll:+.1f}°")
                        else:
                            print(f"[IMU-DISABLED] pitch: {diag['imu_pitch_deg']:+.1f}° roll: {diag['imu_roll_deg']:+.1f}° "
                                  f"→ cam effective: {diag['effective_pitch_deg']:.1f}°/{diag['effective_roll_deg']:+.1f}°")

            # Update matplotlib figure with both views (skip in headless mode)
            if vis is not None:
                try:
                    vis.update_image(display_frame)  # Add camera view
                    vis.update_plot(spatial_detections)   # Update scatter plot
                except Exception:
                    # Matplotlib window was closed, exit gracefully
                    break

            # FPS counter
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                print(f"FPS: {fps:.1f}  |  Detections: {len(spatial_detections)}")
                fps_start_time = time.time()

            # Check for quit key or if matplotlib window is closed
            if cv2.waitKey(1) == ord('q'):
                break
            if vis is not None and not plt.fignum_exists(vis.fig.number):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    finally:
        # Remove vision_running flag file
        try:
            if VISION_FLAG_FILE.exists():
                VISION_FLAG_FILE.unlink()
                print(f"✓ Removed vision flag: {VISION_FLAG_FILE}")
        except Exception as e:
            print(f"⚠️  Could not remove vision flag: {e}")

        cv2.destroyAllWindows()
        plt.ioff()
        # Only close figure if vis exists and has a figure
        if vis is not None and hasattr(vis, 'fig') and vis.fig is not None:
            try:
                if plt.fignum_exists(vis.fig.number):
                    plt.close(vis.fig)
            except Exception:
                pass  # Ignore cleanup errors
        print("\nDetection script exited cleanly")
