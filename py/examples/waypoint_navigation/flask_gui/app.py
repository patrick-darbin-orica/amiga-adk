#!/usr/bin/env python3
"""
Flask GUI for Waypoint Navigation System
Provides web-based monitoring and control interface for the Amiga robot.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import subprocess
import signal
import time
import asyncio
from typing import Optional

# Add parent directory to path to import from navigation system
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.pose_cache import get_latest_pose, set_latest_pose
from utils.navigation_state import get_navigation_state, get_waypoint_status
from utils.camera_frame_cache import get_latest_frame_bytes
from utils.oak0_camera_cache import get_oak0_frame_bytes, set_oak0_frame

# Import filter client dependencies
from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng_core_pybind import Pose3F64
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class NavigationState:
    """Shared state for navigation system"""
    def __init__(self):
        self.navigation_process: Optional[subprocess.Popen] = None
        self.camera_frame = None
        self.detections = []
        self.waypoints = []
        self.robot_pose = None
        self.track_status = "IDLE"
        self.current_waypoint_index = 0
        self.total_waypoints = 0
        self.gps_quality = "UNKNOWN"
        self.vision_active = False

    def is_navigation_running(self) -> bool:
        """Check if navigation process is running"""
        if self.navigation_process is None:
            return False
        return self.navigation_process.poll() is None

state = NavigationState()

# ==================== Routes ====================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """
    Stream camera feed with detections overlay.
    Reads the latest frame from detectionPlot via shared camera frame file.
    """
    def generate():
        while True:
            frame_bytes = get_latest_frame_bytes()
            if frame_bytes is not None:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error streaming frame: {e}")
            time.sleep(1/15)  # 15 FPS max (matches detectionPlot FPS)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    """
    Stream oak0 camera feed (farm-ng camera service).
    Reads the latest frame from oak0 camera via shared camera frame file.
    """
    def generate():
        while True:
            frame_bytes = get_oak0_frame_bytes()
            if frame_bytes is not None:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error streaming oak0 frame: {e}")
            time.sleep(1/15)  # 15 FPS to match oak1 camera and reduce latency

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plot_data')
def plot_data():
    """
    Get waypoint plot data for D3.js visualization.
    Returns waypoints, robot position, and planned path.
    """
    # Try to read waypoint data from CSV with status
    waypoint_data = load_waypoint_data()

    # Update waypoint statuses from navigation state
    for wp in waypoint_data:
        wp['status'] = get_waypoint_status(wp['index'])

    # Get current robot pose
    robot_data = None
    pose = get_latest_pose()
    if pose is not None:
        robot_data = {
            'x': pose.x,
            'y': pose.y,
            'heading': pose.yaw
        }

    # Get navigation state
    nav_state = get_navigation_state()

    return jsonify({
        'waypoints': waypoint_data,
        'robot': robot_data,
        'current_index': nav_state['current_waypoint_index'],
        'total': nav_state['total_waypoints']
    })

@app.route('/detection_data')
def detection_data():
    """Get detection scatter plot data from detectionPlot.py"""
    try:
        with open('/tmp/amiga_detections.json', 'r') as f:
            detections = json.load(f)
        return jsonify({'detections': detections})
    except Exception:
        return jsonify({'detections': []})

@app.route('/robot_status')
def robot_status():
    """Get current robot status for display"""
    pose = get_latest_pose()
    nav_state = get_navigation_state()

    # Check if navigation subprocess is actually running
    subprocess_running = state.is_navigation_running()

    # Determine navigation status: subprocess must be running OR state file says running
    # But if subprocess is NOT running and state says running, that's stale - clear it
    if not subprocess_running and nav_state['navigation_running']:
        # Stale state detected - clear it
        from utils.navigation_state import clear_navigation_state
        clear_navigation_state()
        nav_state = get_navigation_state()  # Reload cleared state

    # Final determination: navigation is running if subprocess exists
    nav_running = subprocess_running

    status = {
        'navigation_running': nav_running,
        'track_status': nav_state['track_status'],
        'current_waypoint': nav_state['current_waypoint_index'],
        'total_waypoints': nav_state['total_waypoints'],
        'filter_converged': False,  # Default if no pose available
        'vision_active': nav_state['vision_active'],
        'pose': None
    }

    if pose is not None:
        import math
        status['filter_converged'] = pose.converged
        status['pose'] = {
            'x': pose.x,
            'y': pose.y,
            'heading_deg': math.degrees(pose.yaw)
        }

    return jsonify(status)

@app.route('/camera_diagnostics')
def camera_diagnostics():
    """Get camera feed diagnostics and gRPC health status"""
    diagnostics = {
        'oak1_feed': {
            'source': 'detectionPlot.py (DepthAI)',
            'cache_file': '/tmp/amiga_camera_frame.jpg',
            'status': 'active' if Path('/tmp/amiga_camera_frame.jpg').exists() else 'no_frames'
        },
        'oak0_feed': {
            'source': 'oak0 camera service (gRPC)',
            'cache_file': '/tmp/amiga_oak0_frame.jpg',
            'status': 'active' if Path('/tmp/amiga_oak0_frame.jpg').exists() else 'no_frames',
            'subscription_config': {
                'every_n': 5,
                'host': 'localhost',
                'port': 50010
            }
        }
    }

    # Check frame freshness (if file exists, check modification time)
    oak0_cache = Path('/tmp/amiga_oak0_frame.jpg')
    if oak0_cache.exists():
        age_seconds = time.time() - oak0_cache.stat().st_mtime
        diagnostics['oak0_feed']['frame_age_seconds'] = round(age_seconds, 2)
        if age_seconds > 2.0:
            diagnostics['oak0_feed']['status'] = 'stale'
            diagnostics['oak0_feed']['warning'] = 'Frames not updating (possible gRPC overload or service down)'

    oak1_cache = Path('/tmp/amiga_camera_frame.jpg')
    if oak1_cache.exists():
        age_seconds = time.time() - oak1_cache.stat().st_mtime
        diagnostics['oak1_feed']['frame_age_seconds'] = round(age_seconds, 2)
        if age_seconds > 2.0:
            diagnostics['oak1_feed']['status'] = 'stale'

    return jsonify(diagnostics)

# ==================== Socket.IO Events ====================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to navigation GUI'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_navigation')
def handle_start_navigation():
    """Start the navigation system by running run.sh"""
    if state.is_navigation_running():
        emit('error', {'message': 'Navigation already running'})
        return

    try:
        # Run run.sh in the parent directory
        run_script = Path(__file__).resolve().parents[1] / 'run.sh'

        if not run_script.exists():
            emit('error', {'message': f'run.sh not found at {run_script}'})
            return

        # Start navigation process with unbuffered output
        state.navigation_process = subprocess.Popen(
            ['bash', str(run_script)],
            cwd=run_script.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            bufsize=1,  # Line buffered
            universal_newlines=True,
            preexec_fn=os.setsid  # Create new process group for clean shutdown
        )

        # Start output reader thread
        import threading
        def read_output():
            """Read navigation output and emit to clients"""
            try:
                for line in iter(state.navigation_process.stdout.readline, ''):
                    if line:
                        socketio.emit('nav_log', {'message': line.rstrip()})
            except Exception as e:
                socketio.emit('nav_log', {'message': f'Error reading output: {e}'})

        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()

        emit('success', {'message': 'Navigation started'})
        print(f"Started navigation process (PID: {state.navigation_process.pid})")

    except Exception as e:
        emit('error', {'message': f'Failed to start navigation: {str(e)}'})
        print(f"Error starting navigation: {e}")

@socketio.on('stop_navigation')
def handle_stop_navigation():
    """Stop the navigation system"""
    if not state.is_navigation_running():
        emit('error', {'message': 'Navigation not running'})
        return

    try:
        # Send SIGTERM to process group
        os.killpg(os.getpgid(state.navigation_process.pid), signal.SIGTERM)
        state.navigation_process.wait(timeout=5)
        state.navigation_process = None

        # Clear navigation state file
        from utils.navigation_state import clear_navigation_state
        clear_navigation_state()

        emit('success', {'message': 'Navigation stopped'})
        print("Navigation process stopped")

    except Exception as e:
        # Force kill if graceful shutdown fails
        try:
            os.killpg(os.getpgid(state.navigation_process.pid), signal.SIGKILL)
            state.navigation_process = None
            from utils.navigation_state import clear_navigation_state
            clear_navigation_state()
            emit('warning', {'message': 'Navigation force killed'})
        except:
            emit('error', {'message': f'Failed to stop navigation: {str(e)}'})

@socketio.on('emergency_stop')
def handle_emergency_stop():
    """Emergency stop - immediately kill navigation"""
    if state.is_navigation_running():
        try:
            os.killpg(os.getpgid(state.navigation_process.pid), signal.SIGKILL)
            state.navigation_process = None
        except:
            pass

    # Clear navigation state file
    from utils.navigation_state import clear_navigation_state
    clear_navigation_state()

    emit('success', {'message': 'EMERGENCY STOP ACTIVATED'})
    print("EMERGENCY STOP")

# ==================== Helper Functions ====================

def load_waypoint_data():
    """Load waypoint data from CSV for plotting"""
    waypoints = []

    # Try to read from surveyed waypoints
    waypoint_file = Path(__file__).resolve().parents[1] / 'surveyed-waypoints' / 'physicsLabBack2Lanes.csv'

    if waypoint_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(waypoint_file)

            # Assume CSV has dx, dy columns (ENU coordinates)
            # Note: CSV has dx=X (east), dy=Y (north) in world frame
            for idx, row in df.iterrows():
                waypoints.append({
                    'x': float(row.get('dx', 0)),  # East
                    'y': float(row.get('dy', 0)),  # North
                    'index': int(idx),
                    'status': 'pending'  # Will be updated by navigation system
                })
            print(f"Loaded {len(waypoints)} waypoints from CSV")
        except Exception as e:
            print(f"Error loading waypoints: {e}")
    else:
        print(f"Waypoint file not found: {waypoint_file}")

    return waypoints

async def filter_pose_updater():
    """Subscribe to filter state and continuously update pose cache"""
    try:
        # Load filter service config from parent directory
        filter_config_path = Path(__file__).resolve().parents[1] / 'configs' / 'config.json'

        # Create filter client
        from farm_ng.core.event_service_pb2 import EventServiceConfigList, SubscribeRequest
        config_list = proto_from_json_file(filter_config_path, EventServiceConfigList())

        # Find filter service config
        filter_config = None
        for config in config_list.configs:
            if config.name == "filter":
                filter_config = config
                break

        if filter_config is None:
            print("⚠️  Filter service not found in config, pose updates disabled")
            return

        # Create subscription request manually (since it's not in config)
        from farm_ng.core.uri_pb2 import Uri
        subscription = SubscribeRequest(
            uri=Uri(path="/state", query="service_name=filter"),
            every_n=1
        )

        # Subscribe to filter state
        client = EventClient(filter_config)
        print(f"✓ Subscribed to filter service at {filter_config.host}:{filter_config.port}")

        async for event, message in client.subscribe(subscription, decode=True):
            # Update pose cache with filter state
            pose = Pose3F64.from_proto(message.pose)
            x = float(pose.a_from_b.translation[0])
            y = float(pose.a_from_b.translation[1])
            yaw = float(pose.a_from_b.rotation.log()[-1])
            converged = bool(getattr(message, "has_converged", False))
            set_latest_pose(x, y, yaw, converged)

    except Exception as e:
        print(f"⚠️  Filter pose updater error: {e}")
        import traceback
        traceback.print_exc()


async def oak0_camera_updater():
    """Subscribe to oak0 camera service and continuously update camera frame cache with auto-reconnect"""

    # Load oak0 camera service config (only once at startup)
    oak0_config_path = Path(__file__).resolve().parents[2] / 'camera_client' / 'service_config.json'

    if not oak0_config_path.exists():
        print("⚠️  oak0 camera service config not found at", oak0_config_path)
        print("    Camera feed 2 disabled")
        return

    oak0_config = proto_from_json_file(oak0_config_path, EventServiceConfig())

    if oak0_config is None:
        print("⚠️  oak0 camera service config could not be loaded, camera feed 2 disabled")
        return

    from farm_ng.core.event_service_pb2 import SubscribeRequest
    from farm_ng.core.uri_pb2 import Uri

    # Create subscription config (only once)
    subscription = SubscribeRequest(
        uri=Uri(path="/rgb", query="service_name=oak0"),
        every_n=5  # Only process every 5th frame to reduce latency
    )

    # Auto-reconnect loop
    retry_delay = 5.0  # Start with 5 second retry
    max_retry_delay = 30.0

    while True:
        try:
            # Subscribe to oak0 RGB stream
            client = EventClient(oak0_config)
            print(f"✓ Connecting to oak0 camera service at {oak0_config.host}:{oak0_config.port}")

            # Diagnostics for monitoring subscription health
            frame_count = 0
            last_report_time = time.time()
            last_frame_time = None  # Will be set after first frame
            connected = False

            # Watchdog to detect stuck subscriptions
            async def watchdog():
                """Monitor for stuck subscription (no frames for 30 seconds)"""
                nonlocal last_frame_time
                # Wait for first frame
                while last_frame_time is None:
                    await asyncio.sleep(1)

                # Now monitor for stuck subscription
                while True:
                    await asyncio.sleep(10)
                    if time.time() - last_frame_time > 30:
                        print("⚠️  oak0 subscription stuck (no frames for 30s), reconnecting...")
                        raise TimeoutError("Subscription stuck - no frames received")

            # Start watchdog task
            watchdog_task = asyncio.create_task(watchdog())

            print(f"🔍 Attempting to subscribe to oak0 camera stream (every_n={subscription.every_n})...")
            print("⏳ Waiting for first frame from oak0 camera (60s timeout)...")

            # Create subscription with connection timeout
            connection_timeout = 60.0
            connection_start = time.time()
            first_frame_received = False

            subscription_stream = client.subscribe(subscription, decode=True)

            # Process frames
            async for event, message in subscription_stream:
                last_frame_time = time.time()  # Update watchdog timer
                try:
                    # Check connection timeout on first frame
                    if not first_frame_received:
                        if time.time() - connection_start > connection_timeout:
                            raise ConnectionError("Timeout waiting for first frame from oak0 camera")
                        print("✓ oak0 camera connected successfully - first frame received")
                        first_frame_received = True
                        connected = True
                        retry_delay = 5.0  # Reset retry delay

                    # Track frame timing for latency diagnostics
                    current_time = time.time()
                    frame_count += 1

                    # Get frame timestamp from event
                    if event.timestamps:
                        frame_timestamp = event.timestamps[0].stamp.seconds + event.timestamps[0].stamp.nanos / 1e9

                        # Calculate latency (time between frame capture and processing)
                        latency_ms = (current_time - frame_timestamp) * 1000

                        # Report diagnostics every 5 seconds
                        if current_time - last_report_time > 5.0:
                            fps = frame_count / (current_time - last_report_time)
                            print(f"📊 oak0 camera: {fps:.1f} FPS, latency: {latency_ms:.0f}ms")
                            frame_count = 0
                            last_report_time = current_time

                            # Warn if latency is high
                            if latency_ms > 500:
                                print(f"⚠️  High latency detected ({latency_ms:.0f}ms) - consider increasing every_n")

                    # Decode image data from camera message
                    image = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)

                    if image is not None:
                        # Update oak0 frame cache (overwrites previous frame)
                        set_oak0_frame(image)

                except Exception as decode_error:
                    # Silently skip frame decode errors to avoid spamming logs
                    pass

        except Exception as e:
            # Cancel watchdog task if it exists
            if 'watchdog_task' in locals():
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass

            print(f"⚠️  oak0 camera connection lost: {e}")
            print(f"    Retrying in {retry_delay:.0f} seconds...")
            await asyncio.sleep(retry_delay)

            # Exponential backoff
            retry_delay = min(retry_delay * 1.5, max_retry_delay)
            continue

def background_status_updater():
    """Background thread to emit status updates to all clients"""
    while True:
        try:
            nav_state = get_navigation_state()

            # Check if navigation subprocess is actually running
            subprocess_running = state.is_navigation_running()

            # Detect stale state: if state file says running but no subprocess exists
            if not subprocess_running and nav_state['navigation_running']:
                # Clear stale state
                from utils.navigation_state import clear_navigation_state
                clear_navigation_state()
                nav_state = get_navigation_state()  # Reload cleared state

            # Final determination: navigation is running if subprocess exists
            nav_running = subprocess_running

            status = {
                'navigation_running': nav_running,
                'track_status': nav_state['track_status'],
                'current_waypoint': nav_state['current_waypoint_index'],
                'total_waypoints': nav_state['total_waypoints'],
                'filter_converged': False,  # Default if no pose available
                'vision_active': nav_state['vision_active']
            }

            # Get robot pose
            pose = get_latest_pose()
            if pose is not None:
                import math
                status['filter_converged'] = pose.converged
                status['pose'] = {
                    'x': pose.x,
                    'y': pose.y,
                    'heading_deg': math.degrees(pose.yaw)
                }

            socketio.emit('status_update', status)

        except Exception as e:
            print(f"Error in status updater: {e}")

        time.sleep(0.5)  # Update twice per second

# ==================== Main ====================

def run_async_filter_updater():
    """Run the async filter updater in its own event loop"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(filter_pose_updater())


def run_async_oak0_camera_updater():
    """Run the async oak0 camera updater in its own event loop"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(oak0_camera_updater())


if __name__ == '__main__':
    # Start background filter pose updater
    filter_thread = threading.Thread(target=run_async_filter_updater, daemon=True)
    filter_thread.start()

    # Start background oak0 camera updater
    # NOTE: gRPC BlockingIOError warnings are harmless (Python 3.8 asyncio issue)
    # The camera feed will work despite these errors
    try:
        oak0_config_path = Path(__file__).resolve().parents[2] / 'camera_client' / 'service_config.json'
        if oak0_config_path.exists():
            # Suppress asyncio error logging for gRPC (known Python 3.8 issue)
            import logging
            logging.getLogger('asyncio').setLevel(logging.CRITICAL)

            oak0_camera_thread = threading.Thread(target=run_async_oak0_camera_updater, daemon=True)
            oak0_camera_thread.start()
            print("✓ Started oak0 camera feed thread")
            print("  (Note: gRPC BlockingIOError warnings in logs are harmless)")
        else:
            print("⚠️  oak0 camera service config not found - Camera Feed 2 disabled")
    except Exception as e:
        print(f"⚠️  Could not start oak0 camera thread: {e}")

    # Start background status updater
    status_thread = threading.Thread(target=background_status_updater, daemon=True)
    status_thread.start()

    print("\n" + "="*70)
    print("AMIGA WAYPOINT NAVIGATION - WEB GUI")
    print("="*70)
    print(f"Starting Flask server on http://0.0.0.0:5000")
    print("Open in browser to monitor and control navigation")
    print("="*70 + "\n")

    # Run Flask with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
