from flask import Flask, Response
import numpy as np
import threading
import time
import signal
import sys
import logging
import traceback
import argparse
import os
sys.path.insert(0, '/mnt/managed_home/farm-ng-user-patrick-orica')
import nms_patch

import depthai as dai
import cv2
import asyncio

from pathlib import Path
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file

from utils.pose_recognition import poseKeypoints

from ultralytics import YOLO

from utils.amiga_movement import move_forwards, move_backwards, coord_move_forwards

# Initialise the browser "app" created by flask
app = Flask(__name__)

# Data processing parameters
CONFIDENCE_THRESHOLD = 0.5
fps_limit = 20

# Device information
DEVICE = "14442C1001A528D700"

# Thread states
current_frame = None
shutdown_event = None
frame_lock = threading.Lock()
shutdown_attempts = 0

# Gesture detection frame variables
leftarmwide_frames = 0
leftarmup_frames = 0
rightarmwide_frames = 0
rightarmup_frames = 0
tpose_frames = 0
armsup_frames = 0


# Calculate new ROI coordinates each time a bounding box is detected
def roi_coords(xmin, xmax, ymin, ymax, frame_width, frame_height, spatial_config, inputConfigQ):
    topLeft = dai.Point2f(xmin / frame_width, ymin / frame_height)
    bottomRight = dai.Point2f(xmax / frame_width, ymax / frame_height)

    spatial_config.roi = dai.Rect(topLeft, bottomRight)
    cfg = dai.SpatialLocationCalculatorConfig()
    cfg.addROI(spatial_config)
    inputConfigQ.send(cfg)


# ------- Camera Initialisation & Gesture Recognition -------
async def camera_thread(client, config):
    global current_frame, shutdown_event, leftarmwide_frames, leftarmup_frames, rightarmwide_frames
    global rightarmup_frames, tpose_frames, armsup_frames
    device = None
    pipeline = None
    spatialData_now = None

    # Initialise ROI coords (pre-detection)
    topLeft = dai.Point2f(0.1, 0.1)
    bottomRight = dai.Point2f(0.1, 0.1)

    # Initialise shutdown event (to terminate camera stream)
    shutdown_event = asyncio.Event()

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

        spatial_config = dai.SpatialLocationCalculatorConfigData()
        spatial_config.depthThresholds.lowerThreshold = 10  # in mm
        spatial_config.depthThresholds.upperThreshold = 10000  # in mm
        calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        spatial_config.calculationAlgorithm = calculationAlgorithm
        spatial_config.roi = dai.Rect(topLeft, bottomRight)

        spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        spatialLocationCalculator.initialConfig.addROI(spatial_config)

        # Create output queues
        RGBQ = RGBout.createOutputQueue(maxSize=1, blocking=False)
        spatialQ = spatialLocationCalculator.out.createOutputQueue()

        # Link stereo depth calculator and create input queue
        stereo.depth.link(spatialLocationCalculator.inputDepth)
        inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()

        # Start the pipeline
        pipeline.start()
        print("Pipeline has started.\n")
        print("Detectable poses:")
        print(" - T-Pose: Both arms extended horizontally.")
        print(" - Both Hands Up: Both arms extended vertically.")
        print(" - Left Arm Wide: Left arm extended horizontally.")
        print(" - Right Arm Wide: Right arm extended horizontally.")
        print(" - Left Arm Up: Left arm extended vertically.")
        print(" - Right Arm Up: Right arm extended vertically.\n")
        print("To terminate the camera stream, press 'CTRL+C' in terminal.")

        # Load the YOLO model (gesture recognition - instead of depthai models)
        model = YOLO("yolo26n-pose.engine")

        # Initialise pose classifier
        pose_classifier = poseKeypoints(confidence_threshold=0.3)

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
                            # TODO: If a specific gesture has been detected, increment its "frame count" by 1
                            #       if statement is an OR condition (1 gesture/frame)
                            # TODO: Assign the specific gesture detected to the correct frame variable and increment by 1
                            if gesture_detection.pose_name == 'Left Arm Wide':
                                leftarmwide_frames += 1
                            elif gesture_detection.pose_name == 'Left Arm Up':
                                leftarmup_frames += 1
                            elif gesture_detection.pose_name == 'Right Arm Wide':
                                rightarmwide_frames += 1
                            elif gesture_detection.pose_name == 'Right Arm Up':
                                rightarmup_frames += 1
                            elif gesture_detection.pose_name == 'T-Pose':
                                tpose_frames += 1
                            elif gesture_detection.pose_name == 'Both Arms Up':
                                armsup_frames += 1

                            gesture_count = {
                                'Left Arm Wide': leftarmwide_frames,
                                'Left Arm Up': leftarmup_frames,
                                'Right Arm Wide': rightarmwide_frames,
                                'Right Arm Up': rightarmup_frames,
                                'T-Pose': tpose_frames,
                                'Both Arms Up': armsup_frames}

                            # TODO: Implement the user input inside of an if statement. If a certain gesture has reached
                            #       x amount of frames, require the user input to proceed before commencing the specific
                            #       action allocated by the gesture
                            max_gesture = max(gesture_count, key=gesture_count.get)
                            detected_gesture = gesture_count[max_gesture]

                            if detected_gesture >= 75:
                                print(f"\nGesture '{max_gesture}' detected for 75 frames!")
                                z_coord_now = spatialData_now.spatialCoordinates.z
                                print(f"The person is {z_coord_now} mm away")
                                user_result = await pose_classifier.gesture_user_input(gesture_detection)
                                if user_result == "commence":
                                    await coord_move_forwards(config, client, z_coord_now)

                                leftarmwide_frames = 0
                                leftarmup_frames = 0
                                rightarmwide_frames = 0
                                rightarmup_frames = 0
                                tpose_frames = 0
                                armsup_frames = 0
                        else:
                            leftarmwide_frames = max(0, leftarmwide_frames - 1)
                            leftarmup_frames = max(0, leftarmup_frames - 1)
                            rightarmwide_frames = max(0, rightarmwide_frames - 1)
                            rightarmup_frames = max(0, rightarmup_frames - 1)
                            tpose_frames = max(0, tpose_frames - 1)
                            armsup_frames = max(0, armsup_frames - 1)

                        # For each time a bounding box is detected, determine the ROI coordinates
                        for bbox in gesture[0].boxes.xyxy.cpu():
                            xmin, ymin, xmax, ymax = bbox
                            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                            roi_coords(xmin, xmax, ymin, ymax, 480, 480, spatial_config, inputConfigQueue)

                            # Using the updated ROI coordinates, determine the spatial data of the bounding box and
                            # display the spatial coordinates (the result) to that frame
                            if spatialQ.has():
                                spatialData = spatialQ.get().getSpatialLocations()

                                if spatialData:
                                    spatialData_now = spatialData[0]

                                    cv2.putText(gesture_frame, f"X: {int(spatialData_now.spatialCoordinates.x)} mm", (xmin + 10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    cv2.putText(gesture_frame, f"Y: {int(spatialData_now.spatialCoordinates.y)} mm", (xmin + 10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    cv2.putText(gesture_frame, f"Z: {int((spatialData_now.spatialCoordinates.z)/1.9)} mm", (xmin + 10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Assign the gesture frame to the current frame
                    with frame_lock:
                        current_frame = gesture_frame

    except asyncio.CancelledError:
        print("Camera stream was cancelled")
        raise
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
    # Initialise frame variables
    frame_interval = 1.0 / fps_limit
    last_send_time = 0

    while True:
        now = time.monotonic()
        elapsed_time = now - last_send_time

        # If not enough time has passed between frames, sleep
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)

        last_send_time = time.monotonic()
        # Get most recent frame
        with frame_lock:
            if current_frame is None:
                waiting_frame = np.zeros((480, 480, 3), dtype=np.uint8)
                frame = waiting_frame
            else:
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
    global shutdown_attempts
    shutdown_attempts += 1
    print("Shutdown signal received...\n")
    if shutdown_attempts == 1:
        if shutdown_event is not None:
            try:
                asyncio.get_event_loop().call_soon_threadsafe(shutdown_event.set)
            except RuntimeError:
                pass
    else:
        os._exit(0)
    print("Exiting.\n")


async def main(service_config_path: Path) -> None:
    # Create a client to the canbus service
    config: EventServiceConfig = proto_from_json_file(service_config_path, EventServiceConfig())
    client: EventClient = EventClient(config)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start web server
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5500, threaded=True, debug=False, use_reloader=False),
        daemon=True)
    flask_thread.start()

    await asyncio.sleep(2)
    print("The Amiga camera stream is available at: http://192.168.1.70:5500 \n")

    # Start camera in background thread
    print("Initialising the camera...")
    cam_thread = asyncio.create_task(camera_thread(client, config))
    await asyncio.sleep(2)

    print("The Amiga has finished initialising. The camera feed should now be visible.")

    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        await asyncio.sleep(0.5)

        cam_thread.cancel()
        try:
            await asyncio.wait_for(cam_thread, timeout=2.0)
        except asyncio.CancelledError:
            pass

        await asyncio.sleep(0.5)
        print("Camera stream has been terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python3 test.py", description="Run gesture control via camera stream on the Amiga."
    )
    parser.add_argument("--service-config", type=Path, required=True, help="The canbus service config.")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.service_config))
    except KeyboardInterrupt:
        print('Camera stream terminated')
        os._exit(0)
