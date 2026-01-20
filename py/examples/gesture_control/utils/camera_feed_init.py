import cv2

from flask import Flask, Response
import threading
import time

# Shared state
current_frame = None
frame_lock = threading.Lock()
pipeline_ready = threading.Event()
shutdown_event = threading.Event()


def createApp(pipeline):
    app = Flask(__name__)

    def generate_frames():
        """Generate MJPEG stream"""
        # Wait for camera initialization
        if not pipeline.pipeline_ready.wait(timeout=10):
            print("Camera initialization failed")
            return

        while not pipeline.shutdown_event.is_set():
            # Get latest frame
            with pipeline.frame_lock:
                if pipeline.current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = pipeline.current_frame.copy()

            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

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
        """Video stream endpoint"""
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app
