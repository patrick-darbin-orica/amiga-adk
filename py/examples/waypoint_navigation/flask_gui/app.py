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
from utils.oak2_camera_cache import get_oak2_frame_bytes, set_oak2_frame, is_inference_active

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
    Stream oak2 mono camera feed (downward-facing alignment camera).
    Reads the latest frame from oak2 camera via shared camera frame file.
    Uses mono (left camera) for better performance (1 channel vs 3 for RGB).
    """
    def generate():
        while True:
            frame_bytes = get_oak2_frame_bytes()
            if frame_bytes is not None:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error streaming oak2 frame: {e}")
            time.sleep(1/10)  # 10 FPS for better performance

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
        'oak2_feed': {
            'source': 'oak2 alignment service (DepthAI)',
            'cache_file': '/tmp/amiga_oak2_frame.jpg',
            'status': 'active' if Path('/tmp/amiga_oak2_frame.jpg').exists() else 'no_frames',
            'subscription_config': {
                'every_n': 5,
                'host': 'localhost',
                'port': 50010
            }
        }
    }

    # Check frame freshness (if file exists, check modification time)
    oak2_cache = Path('/tmp/amiga_oak2_frame.jpg')
    if oak2_cache.exists():
        age_seconds = time.time() - oak2_cache.stat().st_mtime
        diagnostics['oak2_feed']['frame_age_seconds'] = round(age_seconds, 2)
        if age_seconds > 2.0:
            diagnostics['oak2_feed']['status'] = 'stale'
            diagnostics['oak2_feed']['warning'] = 'Frames not updating (alignment service may be down)'

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

    # Try to read from surveyed waypoints (configurable via env var)
    # Default to kktcBack2Lanes.csv to match run.sh
    csv_filename = os.environ.get('WAYPOINT_CSV', 'kktcBack2Lanes.csv')
    waypoint_file = Path(__file__).resolve().parents[1] / 'surveyed-waypoints' / csv_filename

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
    print("[oak0] Starting oak0 camera updater thread...", flush=True)

    # Load oak0 camera service config (only once at startup)
    oak0_config_path = Path(__file__).resolve().parents[2] / 'camera_client' / 'service_config.json'

    if not oak0_config_path.exists():
        print(f"⚠️  [oak0] Service config not found at {oak0_config_path}", flush=True)
        print("    Camera feed 2 disabled", flush=True)
        return

    oak0_config = proto_from_json_file(oak0_config_path, EventServiceConfig())

    if oak0_config is None:
        print("⚠️  [oak0] Service config could not be loaded, camera feed 2 disabled", flush=True)
        return

    print(f"✓ [oak0] Loaded config: {oak0_config.host}:{oak0_config.port}", flush=True)

    from farm_ng.core.event_service_pb2 import SubscribeRequest
    from farm_ng.core.uri_pb2 import Uri

    # Create subscription config (only once)
    # Using /rgb for color data (can use /left for mono if better performance needed)
    # Note: service_name is "oak/0" not "oak0" (farm-ng uses oak/0 format)
    subscription = SubscribeRequest(
        uri=Uri(path="/rgb", query="service_name=oak/0"),
        every_n=1  # Process every frame for smoother video
    )
    print("✓ [oak0] Created subscription: path=/rgb, service_name=oak/0, every_n=1", flush=True)

    # Auto-reconnect loop
    retry_delay = 5.0  # Start with 5 second retry
    max_retry_delay = 30.0

    while True:
        try:
            # Subscribe to oak0 RGB stream
            print(f"[oak0] Creating EventClient for {oak0_config.host}:{oak0_config.port}...")
            client = EventClient(oak0_config)
            print(f"✓ [oak0] EventClient created, connecting to camera service...")

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

            # Process frames (with frame skipping for performance)
            frame_skip_counter = 0
            PROCESS_EVERY_N_FRAMES = 3  # Only process every 3rd frame (~10 FPS)

            async for event, message in subscription_stream:
                last_frame_time = time.time()  # Update watchdog timer
                frame_skip_counter += 1

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

                    # Skip frames to reduce CPU load (only process every Nth frame)
                    if frame_skip_counter % PROCESS_EVERY_N_FRAMES != 0:
                        continue  # Skip this frame, don't decode/encode/write

                    # Report diagnostics every 5 seconds
                    if current_time - last_report_time > 5.0:
                        fps = frame_count / (current_time - last_report_time)
                        # Check if cache file exists and is fresh
                        import os
                        cache_exists = os.path.exists('/tmp/amiga_oak0_frame.jpg')
                        cache_age = 0
                        if cache_exists:
                            cache_age = current_time - os.path.getmtime('/tmp/amiga_oak0_frame.jpg')
                        effective_fps = fps / PROCESS_EVERY_N_FRAMES
                        print(f"📊 oak0 camera (gRPC): {fps:.1f} FPS received, {effective_fps:.1f} FPS processed, cache: {'OK' if cache_age < 2 else 'STALE'}")
                        frame_count = 0
                        last_report_time = current_time

                    # Check if inference is active - if so, don't overwrite processed frames
                    if is_inference_active():
                        # Inference is running, skip writing raw frames to avoid conflict
                        # Log once when we first detect inference is active
                        if frame_count == 1 or (frame_count % 50 == 0):
                            print(f"[oak0] Inference active - skipping raw frame writes (frame {frame_count})")
                        continue

                    # Decode image data from camera message
                    image = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)

                    if image is not None:
                        # Update oak0 frame cache (overwrites previous frame)
                        set_oak0_frame(image)
                    else:
                        # Log first decode failure
                        if frame_count == 1:
                            print("⚠️  oak0: Failed to decode first frame")

                except Exception as decode_error:
                    # Log first frame error, then suppress to avoid spam
                    if frame_count == 1:
                        print(f"⚠️  oak0 frame processing error: {decode_error}")
                        import traceback
                        traceback.print_exc()

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
    import sys
    # CRITICAL FIX: Set selector event loop for Python 3.8 gRPC compatibility
    # This prevents BlockingIOError in threading + asyncio + gRPC
    if sys.platform != 'win32':
        import selectors
        selector = selectors.SelectSelector()
        loop = asyncio.SelectorEventLoop(selector)
    else:
        loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)
    loop.run_until_complete(filter_pose_updater())


def run_async_oak0_camera_updater():
    """Run the async oak0 camera updater in its own event loop"""
    import asyncio
    import sys
    try:
        # CRITICAL FIX: Set selector event loop for Python 3.8 gRPC compatibility
        # This prevents BlockingIOError in threading + asyncio + gRPC
        if sys.platform != 'win32':
            import selectors
            selector = selectors.SelectSelector()
            loop = asyncio.SelectorEventLoop(selector)
        else:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
        print("✓ oak0 camera thread event loop created (SelectorEventLoop for gRPC compatibility)", flush=True)
        print("[oak0] About to start oak0_camera_updater()...", flush=True)
        loop.run_until_complete(oak0_camera_updater())
    except Exception as e:
        print(f"⚠️  CRITICAL: oak0 camera thread crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Start background filter pose updater
    filter_thread = threading.Thread(target=run_async_filter_updater, daemon=True)
    filter_thread.start()

    # Start background oak0 camera updater
    # NOTE: Using SelectorEventLoop to prevent Python 3.8 gRPC BlockingIOError issues
    # NOTE: oak0 gRPC camera subscription disabled - using oak2 alignment service instead
    # The oak2 alignment service writes frames directly to /tmp/amiga_oak2_frame.jpg
    print("ℹ️  oak0 gRPC camera subscription disabled (using oak2 alignment service for Camera Feed 2)")

    # Uncomment below to re-enable oak0 gRPC subscription if needed:
    # try:
    #     oak0_config_path = Path(__file__).resolve().parents[2] / 'camera_client' / 'service_config.json'
    #     if oak0_config_path.exists():
    #         # Suppress gRPC asyncio warnings (now using SelectorEventLoop to prevent them)
    #         import logging
    #         logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    #
    #         oak0_camera_thread = threading.Thread(target=run_async_oak0_camera_updater, daemon=True)
    #         oak0_camera_thread.start()
    #         print("✓ Started oak0 camera feed thread (using SelectorEventLoop for gRPC compatibility)")
    #     else:
    #         print("⚠️  oak0 camera service config not found - Camera Feed 2 disabled")
    # except Exception as e:
    #     print(f"⚠️  Could not start oak0 camera thread: {e}")
    #     import traceback
    #     traceback.print_exc()

    # Start background status updater
    status_thread = threading.Thread(target=background_status_updater, daemon=True)
    status_thread.start()

    # Get Tailscale IP for remote access
    import subprocess
    try:
        tailscale_ip = subprocess.check_output(['tailscale', 'ip', '-4'], text=True).strip()
    except Exception:
        tailscale_ip = None

    print("\n" + "="*70)
    print("AMIGA WAYPOINT NAVIGATION - WEB GUI")
    print("="*70)
    print("Starting Flask server on http://0.0.0.0:5000")
    print("")
    print("Access URLs:")
    print("  Local:       http://localhost:5000")
    if tailscale_ip:
        print(f"  Tailscale:   http://{tailscale_ip}:5000")
    print("")
    print("Open in browser to monitor and control navigation")
    print("="*70 + "\n")

    # Run Flask with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
