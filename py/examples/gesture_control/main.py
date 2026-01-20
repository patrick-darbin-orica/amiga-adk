from flask import Flask, Response
import threading
import time
import signal
import sys
import traceback
sys.path.insert(0, '/mnt/managed_home/farm-ng-user-patrick-orica')
import nms_patch

import depthai as dai
import cv2

from utils.pose_recognition import poseKeypoints

from utils.detection import human_pose_recognition


from ultralytics import YOLO

# Initialise the browser "app" created by flask
app = Flask(__name__)

# Data processing parameters
CONFIDENCE_THRESHOLD = 0.5
fps_limit = 20

# Load the YOLO model (gesture recognition - instead of depthai models)
model = YOLO("yolo26n-pose.engine")

# Initialise pose classifier
pose_classifier = poseKeypoints(confidence_threshold=0.3)

# Device information
DEVICE = "14442C1001A528D700"

# Thread states
current_frame = None
pipeline_ready = threading.Event()
shutdown_event = threading.Event()
frame_lock = threading.Lock()

# FPS variables
camera_fps = 0.0
fps_lock = threading.Lock()


# Calculate new ROI coordinates each time a bounding box is detected
def roi_coords(xmin, xmax, ymin, ymax, frame_width, frame_height, config, inputConfigQ):
    topLeft = dai.Point2f(xmin / frame_width, ymin / frame_height)
    bottomRight = dai.Point2f(xmax / frame_width, ymax / frame_height)

    config.roi = dai.Rect(topLeft, bottomRight)

    cfg = dai.SpatialLocationCalculatorConfig()
    cfg.addROI(config)
    inputConfigQ.send(cfg)


# ------- Camera Initialisation & Gesture Recognition -------
def camera_thread():
    global current_frame, camera_fps
    device = None
    pipeline = None

    # Initialise ROI coords
    topLeft = dai.Point2f(0.1, 0.1)
    bottomRight = dai.Point2f(0.1, 0.1)

    try:
        # Initialise device and pipeline
        device = dai.Device(DEVICE)
        print(f"Connected to device: {DEVICE}. Creating pipeline...\n")
        pipeline = dai.Pipeline(device)
        print("Pipeline created successfully. Starting pipeline...\n")

        # Initialise camera nodes (sources)
        RGB = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        monoL = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        monoR = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        # Initialise stereo camera and spatial location calculator
        stereo = pipeline.create(dai.node.StereoDepth)
        spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

        # RGB and stereo outputs
        RGBout = RGB.requestOutput((480, 480))
        monoLout = monoL.requestOutput((480, 480))
        monoLout.link(stereo.left)
        monoRout = monoR.requestOutput((480, 480))
        monoRout.link(stereo.right)

        # Initial stereo depth configuration
        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)

        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 10  # in mm
        config.depthThresholds.upperThreshold = 10000  # in mm
        config.roi = dai.Rect(topLeft, bottomRight)

        spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        spatialLocationCalculator.initialConfig.addROI(config)

        # Create output queues
        RGBQ = RGBout.createOutputQueue(maxSize=1, blocking=False)
        spatialQ = spatialLocationCalculator.out.createOutputQueue()

        # Link stereo depth calculator and create input queue
        stereo.depth.link(spatialLocationCalculator.inputDepth)
        inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()

        # Start the pipeline
        pipeline.start()
        pipeline_ready.set()
        print("Pipeline has started.\n")
        print("Detectable poses:")
        print(" - T-Pose: Both arms extended horizontally.")
        print(" - Both Hands Up: Both arms extended vertically.")
        print(" - Left Arm Wide: Left arm extended horizontally.")
        print(" - Right Arm Wide: Right arm extended horizontally.")
        print(" - Left Arm Up: Left arm extended vertically.")
        print(" - Right Arm Up: Right arm extended vertically.\n")
        print("To terminate the camera stream, press 'CTRL+C' in terminal.")

        with pipeline:
            latestRGB = None

            while not shutdown_event.is_set() and pipeline.isRunning():
                # Get RGB frames
                while RGBQ.has():
                    RGBMsg = RGBQ.get()
                    latestRGB = RGBMsg.getCvFrame()

                # Use RGB frame for camera feed
                if latestRGB is not None:
                    gesture = model(latestRGB, verbose=False, conf=CONFIDENCE_THRESHOLD)
                    gesture_frame = gesture[0].plot()

                    # If there are keypoints (determined by the mode), classify the pose
                    if gesture[0].keypoints is not None:
                        gesture_detection = pose_classifier.YOLO11classifyPose(gesture[0].keypoints)

                        # If a gesture is detected, display it to that frame
                        if gesture_detection:
                            gesture_display = f"Pose: {gesture_detection.pose_name}, Confidence: {gesture_detection.confidence:.2f}"
                            cv2.putText(gesture_frame, gesture_display, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # For each time a bounding box is detected, determine the ROI coordinates
                    for gesture in gesture[0].boxes.xyxy.cpu():
                        xmin, ymin, xmax, ymax = gesture
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                        roi_coords(xmin, xmax, ymin, ymax, 480, 480, config, inputConfigQueue)

                        # Using the updated ROI coordinates, determine the spatial data of the bounding box and
                        # display the spatial coordinates (the result) to that frame
                        while spatialQ.has():
                            spatialData = spatialQ.get().getSpatialLocations()

                        if spatialData:
                            spatialData_now = spatialData[0]

                            cv2.putText(gesture_frame, f"X: {int(spatialData_now.spatialCoordinates.x)} mm", (xmin + 10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(gesture_frame, f"Y: {int(spatialData_now.spatialCoordinates.y)} mm", (xmin + 10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(gesture_frame, f"Z: {int(spatialData_now.spatialCoordinates.z)} mm", (xmin + 10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Assign the gesture frame to the current frame and lock the frame (threading)
                    with frame_lock:
                        current_frame = gesture_frame

    except Exception as e:
        print(f"Camera error: {e}")
        traceback.print_exc()
    finally:
        print("Stopping camera...")
        if pipeline:
            pipeline.stop()
        if device:
            device.close()
        print("Camera successfully stopped ")


def generate_frames():
    # Wait for camera initialisation
    if not pipeline_ready.wait(timeout=10):
        print("Camera initialisation failed")
        return

    frame_interval = 1.0 / fps_limit
    last_send_time = 0

    while not shutdown_event.is_set():
        now = time.monotonic()
        elapsed_time = now - last_send_time

        # If not enough time has passed between frames, sleep
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)

        last_send_time = time.monotonic()
        # Get most recent frame
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])

        # Yield in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/')
def index():
    """Home page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DepthAI Pose Detection</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: #1a1a1a;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            h1 {
                color: #fff;
                margin-bottom: 10px;
            }
            .status {
                color: #4CAF50;
                margin: 10px;
            }
            img {
                max-width: 80%;
                height: auto;
                border: 2px solid #333;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <h1>Human Pose Detection and Spatial Coordinates</h1>
        <div class="status">● Live</div>
        <img src="/video_feed" alt="Camera Feed">
    </body>
    </html>
    '''


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def signal_handler(sig, frame):
    print("Shutdown signal received...\n")
    shutdown_event.set()
    time.sleep(0.5)  # Give threads time to clean up
    print("Exiting.\n")
    sys.exit(0)


if __name__ == '__main__':
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start camera in background thread
    print("Starting camera initialisation...")
    cam_thread = threading.Thread(target=camera_thread, daemon=False)
    cam_thread.start()
    time.sleep(2)

    # Start web server
    print("To view the Amiga camera stream, visit: http://192.168.1.70:5500 \n")

    try:
        app.run(host='0.0.0.0', port=5500, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        cam_thread.join(timeout=2)
        print("Cleanup complete")
