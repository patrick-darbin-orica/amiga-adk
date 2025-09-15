#!/usr/bin/env python3
""" Example script using yolo spatial detection network 
based off https://docs.luxonis.com/software/depthai/examples/spatial_tiny_yolo/"""

import time, cv2, numpy as np
from pathlib import Path
import depthai as dai

# ==== CONFIG ====
BLOB = Path("/mnt/managed_home/farm-ng-user-patrick-orica/farm-ng-amiga/py/examples/waypoint_navigation/detection/yolov8nCones_openvino_2022.1_5shave.blob")
TARGET_MXID = "14442C1001A528D700"  # from oakDiscover.py
FPS = 30

if not BLOB.exists():
    raise FileNotFoundError(f"Blob not found: {BLOB}")

def colorize_depth(depth_frame: np.ndarray) -> np.ndarray:
    dd = depth_frame[::4]
    mn = 0 if np.all(dd == 0) else np.percentile(dd[dd != 0], 1)
    mx = np.percentile(dd, 99)
    depth_u8 = np.interp(depth_frame, (mn, mx), (0, 255)).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_HOT)

# --- Connect to the specific device (or fallback to any) with OpenVINO 2022.1 ---
cfg = dai.Device.Config()
cfg.version = dai.OpenVINO.Version.VERSION_2022_1

dev = None
try:
    dev = dai.Device(cfg, dai.DeviceInfo(TARGET_MXID))
except Exception:
    dev = dai.Device(cfg)

with dev:
    pipeline = dai.Pipeline(dev)

    # ==== NODES (v3) ====
    # Cameras
    camRgb    = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setExtendedDisparity(True)
    if pipeline.getDefaultDevice().getPlatform() == dai.Platform.RVC2:
        stereo.setOutputSize(640, 400)
    monoLeft .requestOutput((640, 400)).link(stereo.left)
    monoRight.requestOutput((640, 400)).link(stereo.right)

    # Image preprocessor: force BGR planar 640x640 for YOLO
    manip = pipeline.create(dai.node.ImageManip)
    # Some builds expose RawImgFrame.Type, others ImgFrame.Type — try both
    try:
        manip.ImageManipConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    except AttributeError:
        manip.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setResize(640, 640)
    # Keep aspect if you prefer letterboxing; comment out if you want stretch
    try:
        manip.setKeepAspectRatio(True)
    except AttributeError:
        pass

    # Spatial detection network
    sdn = pipeline.create(dai.node.SpatialDetectionNetwork)
    sdn.setBlobPath(str(BLOB))           # local blob (no Hub)
    sdn.input.setBlocking(False)
    sdn.setConfidenceThreshold(0.5)
    sdn.setBoundingBoxScaleFactor(0.5)
    sdn.setDepthLowerThreshold(100)
    sdn.setDepthUpperThreshold(5000)

    # --- Linking (v3 style) ---
    # RGB -> ImageManip (convert/resize) -> SDN
    rgb_stream = camRgb.requestOutput((640, 640), fps=FPS)  # source frames
    rgb_stream.link(manip.input)
    manip.out.link(sdn.input)

    # Stereo depth -> SDN
    stereo.depth.link(sdn.inputDepth)

    # ==== OUTPUT QUEUES (no XLinkOut in v3) ====
    # Use SDN passthrough so RGB is NN-synced (after manip)
    q_rgb   = sdn.passthrough.createOutputQueue(maxSize=4, blocking=False)
    q_det   = sdn.out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = sdn.passthroughDepth.createOutputQueue(maxSize=4, blocking=False)

    # Labels embedded in archive/blob may be missing; default to cone
    try:
        label_map = sdn.getClasses() or ["cone"]
    except Exception:
        label_map = ["cone"]

    # Start
    pipeline.start()

    last = time.monotonic()
    frames = 0
    fps = 0.0

    while pipeline.isRunning():
        rgb_m   = q_rgb.get()
        det_m   = q_det.get()
        depth_m = q_depth.get()

        frame = rgb_m.getCvFrame()
        depth = depth_m.getFrame()
        depth_col = colorize_depth(depth)

        frames += 1
        now = time.monotonic()
        if now - last > 1:
            fps = frames / (now - last)
            frames = 0
            last = now

        H, W = frame.shape[:2]
        for d in det_m.detections:
            # ROI on depth
            roi = d.boundingBoxMapping.roi.denormalize(depth_col.shape[1], depth_col.shape[0])
            tl, br = roi.topLeft(), roi.bottomRight()
            cv2.rectangle(depth_col, (int(tl.x), int(tl.y)), (int(br.x), int(br.y)), (255,255,255), 1)

            # RGB bbox + labels/XYZ
            x1, y1 = int(d.xmin * W), int(d.ymin * H)
            x2, y2 = int(d.xmax * W), int(d.ymax * H)
            label = label_map[d.label] if 0 <= d.label < len(label_map) else str(d.label)

            cv2.putText(frame, label, (x1+10, y1+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{d.confidence*100:.1f}", (x1+10, y1+35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X:{int(d.spatialCoordinates.x)}", (x1+10, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y:{int(d.spatialCoordinates.y)}", (x1+10, y1+65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z:{int(d.spatialCoordinates.z)}", (x1+10, y1+80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 1)

        cv2.putText(frame, f"NN fps: {fps:.2f}", (2, H-4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
        cv2.imshow("depth", depth_col)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()
