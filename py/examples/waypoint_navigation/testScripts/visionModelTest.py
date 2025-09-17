#!/usr/bin/env python3
import time, cv2, os
import depthai as dai

ARC = "/mnt/managed_home/farm-ng-user-patrick-orica/farm-ng-amiga/py/examples/waypoint_navigation/testScripts/nnarchives/yolov8nCones.tar.xz"
# If you want to force a specific OAK, do this in your shell before running:
#   export DEPTHAI_DEVICE_MXID=14442C1001A528D700

def _has_gui():
    try: bi = cv2.getBuildInformation()
    except Exception: return False
    gui = next((l for l in bi.splitlines() if l.strip().startswith("GUI:")), "")
    return hasattr(cv2,"imshow") and hasattr(cv2,"waitKey") and os.environ.get("DISPLAY") and "None" not in gui
HEADLESS = not _has_gui()

def maybe_imshow(name, img):
    try:
        cv2.imshow(name, img)
    except cv2.error:
        # headless build: silently ignore drawing
        pass

def maybe_quit():
    try:
        # if GUI is available this works; if headless it throws
        return (cv2.waitKey(1) & 0xFF) == ord('q')
    except cv2.error:
        # headless fallback: no keyboard; let Ctrl+C stop the script
        time.sleep(0.01)
        return False
    
def frameNorm(frame, bbox):
    h, w = frame.shape[:2]
    x1 = int(max(0,min(1,bbox[0]))*w); y1 = int(max(0,min(1,bbox[1]))*h)
    x2 = int(max(0,min(1,bbox[2]))*w); y2 = int(max(0,min(1,bbox[3]))*h)
    return x1,y1,x2,y2

with dai.Pipeline() as pipeline:
    # --- Cameras
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoL  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoR  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # --- Stereo depth: you MUST feed left/right
    stereo = pipeline.create(dai.node.StereoDepth)
    leftOut  = monoL.requestOutput((640, 400))   # produces ImgFrame
    rightOut = monoR.requestOutput((640, 400))
    leftOut.link(stereo.left)
    rightOut.link(stereo.right)

    # --- Spatial Detection from your local NNArchive (must contain .blob + exactly one YOLO head)
    arc = dai.NNArchive(ARC)
    sdn = pipeline.create(dai.node.SpatialDetectionNetwork).build(camRgb, stereo, arc, fps=15)
    sdn.input.setBlocking(False)
    sdn.setBoundingBoxScaleFactor(0.5)
    sdn.setDepthLowerThreshold(100)     # 10 cm
    sdn.setDepthUpperThreshold(5000)    # 5 m

    # --- Outputs (v3 style)
    qRgb = sdn.passthrough.createOutputQueue()   # ImgFrame aligned to NN input
    qDet = sdn.out.createOutputQueue()           # ImgDetections (with spatial coords)
    labels = sdn.getClasses()

    # Start pipeline (uses DEPTHAI_DEVICE_MXID if set)
    pipeline.start()
    print("Pipeline started. Press 'q' to quit.")

    start = time.monotonic(); counter = 0
    frame = None

    while pipeline.isRunning():
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

        if inDet is not None and frame is not None:
            dets = inDet.detections
            counter += 1
            for d in dets:
                x1,y1,x2,y2 = frameNorm(frame, (d.xmin,d.ymin,d.xmax,d.ymax))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                name = labels[d.label] if labels and 0 <= d.label < len(labels) else str(d.label)
                cv2.putText(frame, f"{name} {d.confidence:.2f}", (x1,max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                sx,sy,sz = d.spatialCoordinates.x/1000, d.spatialCoordinates.y/1000, d.spatialCoordinates.z/1000
                cv2.putText(frame, f"({sx:.2f},{sy:.2f},{sz:.2f}) m",
                            (x1, min(frame.shape[0]-5, y1+18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if frame is not None:
            fps = counter / max(1e-6, (time.monotonic()-start))
            cv2.putText(frame, f"NN fps: {fps:.2f}", (2, frame.shape[0]-6),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)
            maybe_imshow("sdn_passthrough", frame)

        if maybe_quit():
            pipeline.stop()
            break

        if inRgb is None and inDet is None:
            time.sleep(0.005)
