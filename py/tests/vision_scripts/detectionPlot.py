#!/usr/bin/env python3
# detectionPlot_v3.py — DepthAI v3 “latest-only” rendering with FOV overlay + waypoint icon + ground-aware range arcs

import cv2, time, math
import depthai as dai
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import json
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Pose3F64
from farm_ng_core_pybind import Rotation3F64

# ---------------- Model / rates ----------------
modelDescription = dai.NNModelDescription("luxonis/ppe-detection:640x640")
FPS = 30

# ---------------- Camera / FOV ----------------
USE_RGB_FOV     = True
RGB_HFOV_DEG    = 95     # per your spec
RGB_VFOV_DEG    = 72     # vertical FOV used in projector/overlays
STEREO_HFOV_DEG = 127    # keep if you ever switch to stereo FOV overlays

# ---- Fixed camera tilt & height (no IMU) ----
CAM_HEIGHT_M = 0.865
TILT_DEG     = 30.0      # +30° pitch DOWN
theta = math.radians(TILT_DEG)

# Camera frame: +X right, +Y down, +Z forward. Level camera => up = [0, -1, 0].
# With forward tilt, up tilts toward +Z: [0, -cos(theta), -sin(theta)].
N_UP_CAM = np.array([0.0, -math.cos(theta), -math.sin(theta)], dtype=float)
N_UP_CAM /= (np.linalg.norm(N_UP_CAM) + 1e-9)

# ---------------- Plot & display config ----------------
KEEP_HISTORY   = False
MAX_RANGE_M_Z  = 10.0   # forward axis [m] for scatter view bounds
MAX_RANGE_M_X  = 5.0    # lateral axis [m] for scatter view bounds
PLOT_REFRESH_S = 0.03

# ---------------- Range arcs (meters) ----------------
ARCS_METERS = [1.0, 2.0, 3.0, 4.0, 6.0]

# ---------------- Waypoint (camera-frame) ----------------
# DepthAI camera frame: +X right, +Y down, +Z forward (meters)
WAYPOINT_X_M = -0.35
WAYPOINT_Y_M =  0.35
WAYPOINT_Z_M =  2.32

# OPTIONAL: lock to a specific device (MxID or name). Leave as "" to use default device.
TARGET_DEVICE  = "14442C1001A528D700"  # or ""

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

# ---------------- Visualizer ----------------
class SpatialVisualizer:
    def __init__(self, n_up_cam):
        matplotlib.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig, self.ax = plt.subplots(num="OAK-D: Top-Down Scatter (robot frame)")
        # Robot/world (NWU): horizontal = Y (left +, right -), vertical = X (forward +)
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
        self.waypoint_handle = self.ax.scatter([WAYPOINT_X_M], [WAYPOINT_Z_M], marker="X", s=90)

        self._last_plot = time.time()
        self.labelMap = []

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

        cv2.putText(frame, f"{d_robot_3d:.2f}m from bot centre",   (x1+10, y1+95),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        # Camera-frame readout (kept since you said XYZ are accurate)
        try:
            label = self.labelMap[det.label]
        except Exception:
            label = str(det.label)

        cv2.putText(frame, f"{label}", (x1+10,y1+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(det.confidence*100), (x1+10,y1+35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        # cv2.putText(frame, f"X: {p[0]:.2f} m", (x1+10,y1+50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        # cv2.putText(frame, f"Y: {p[1]:.2f} m", (x1+10,y1+65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        # cv2.putText(frame, f"Z: {p[2]:.2f} m", (x1+10,y1+80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)

    def update_plot(self, detections):
        # Robot-frame ground coords
        ys_left, xs_fwd = [], []

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

            if math.isfinite(x_fwd) and math.isfinite(y_left):
                ys_left.append(y_left)
                xs_fwd.append(x_fwd)
                if KEEP_HISTORY:
                    # reuse history buffers: store (y_left, x_fwd)
                    self.hist_x.append(y_left)
                    self.hist_z.append(x_fwd)

        now = time.time()
        if now - self._last_plot >= PLOT_REFRESH_S:
            # set_offsets expects [x(horiz), y(vert)] -> [Y_left, X_fwd]
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

        # Pose: robot_from_camera
        self.robot_from_camera = Pose3F64(
            a_from_b=Isometry3F64([tx, ty, tz], R_cam_to_robot),
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


# ---------------- Build pipeline (v3 style) ----------------
if TARGET_DEVICE:
    device = dai.Device(TARGET_DEVICE)  # Lock to this OAK
    pipeline = dai.Pipeline(device)
else:
    pipeline = dai.Pipeline()

# Nodes
camRgb  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
monoL   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoR   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo  = pipeline.create(dai.node.StereoDepth)
sdn     = pipeline.create(dai.node.SpatialDetectionNetwork).build(camRgb, stereo, modelDescription, fps=FPS)

# Stereo & NN config
stereo.setExtendedDisparity(True)
if pipeline.getDefaultDevice().getPlatform() == dai.Platform.RVC2:
    stereo.setOutputSize(640, 400)

sdn.input.setBlocking(False)
sdn.setBoundingBoxScaleFactor(0.5)
sdn.setDepthLowerThreshold(100)
sdn.setDepthUpperThreshold(5000)

# Links (request mono outputs, feed StereoDepth)
monoL.requestOutput((640, 400)).link(stereo.left)
monoR.requestOutput((640, 400)).link(stereo.right)

# Host queues (v3): create directly from node outputs (latest-only drained)
qDepth = stereo.depth.createOutputQueue()
qDet   = sdn.out.createOutputQueue()
qRgb   = sdn.passthrough.createOutputQueue()
for q in (qDepth, qDet, qRgb):
    q.setBlocking(False); q.setMaxSize(1)

# Visualizer
vis = SpatialVisualizer(N_UP_CAM)
vis.labelMap = sdn.getClasses()

# Transforms
xfm = Transforms(Path("camera_offset.json"))

# ---------------- Run (latest-only draining) ----------------
def drain_latest(q):
    last = None
    while q.has():
        last = q.get()
    return last

pipeline.start()
with pipeline:
    latestDepth = None
    latestRgb   = None
    latestDets  = []

    hfov_deg = RGB_HFOV_DEG if USE_RGB_FOV else STEREO_HFOV_DEG
    img_w = img_h = None

    while pipeline.isRunning():
        depthMsg = drain_latest(qDepth)
        if depthMsg is not None:
            latestDepth = depthMsg.getFrame()

        detMsg = drain_latest(qDet)
        if detMsg is not None:
            latestDets = detMsg.detections

        rgbMsg = drain_latest(qRgb)
        if rgbMsg is not None:
            latestRgb = rgbMsg.getCvFrame()
            if img_w is None:
                img_h, img_w = latestRgb.shape[:2]

        if latestRgb is not None and latestDepth is not None:
            depthColor = vis.processDepthFrame(latestDepth.copy())
            h, w, _ = latestRgb.shape

            # Draw detections
            for det in latestDets:
                try: 
                    vis.drawBBoxOnDepth(depthColor, det)
                except Exception:
                    pass
                vis.drawDetOnRgb(latestRgb, det, w, h)

            # Waypoint + (optional) FOV/ground-aware arcs on the NN view
            if img_w is not None:
                uv = project_point_to_pixels(
                    WAYPOINT_X_M, WAYPOINT_Y_M, WAYPOINT_Z_M,
                    img_w, img_h, hfov_deg, vfov_deg=RGB_VFOV_DEG
                )
                # Optional overlays:
                # if uv is not None:
                #     draw_waypoint_icon(latestRgb, *uv, color=(0, 255, 255))
                # draw_wedge_arcs_on_image(latestRgb, ARCS_METERS, hfov_deg,
                #                          vfov_deg=RGB_VFOV_DEG, color=(200,200,200))
                # draw_ground_arcs_on_image(latestRgb, ARCS_METERS, N_UP_CAM,
                #                           hfov_deg, vfov_deg=RGB_VFOV_DEG, color=(200,200,200))

            # Show windows
            # cv2.imshow("depth", depthColor)
            cv2.imshow("rgb", latestRgb)

            # Update X–Z map
            vis.update_plot(latestDets)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
plt.close('all')
