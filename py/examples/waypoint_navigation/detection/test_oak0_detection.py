#!/usr/bin/env python3
"""
Test script: YOLO detection on oak0 gRPC camera stream
Subscribes to oak0 RGB stream via farm-ng gRPC and runs YOLO inference
"""

import asyncio
import cv2
import numpy as np
from pathlib import Path
import time
import sys

# Add parent directory to import detection classes
sys.path.append(str(Path(__file__).resolve().parent))

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri

# Import YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False
    sys.exit(1)

# Model configuration
# TODO: Switch back to best.engine after rebuild completes (currently rebuilding for correct classes)
MODEL_PATH = Path(__file__).parent / "best.engine"  # Temporarily use .pt until engine rebuild finishes
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


async def test_oak0_detection():
    """Subscribe to oak0 camera and run YOLO detection"""
    
    # Load model
    print(f"\n{'='*70}")
    print("TESTING YOLO DETECTION ON OAK0 GRPC STREAM")
    print(f"{'='*70}")
    print(f"Model: {MODEL_PATH}")
    print(f"Conf:  {CONF_THRESHOLD}")
    print(f"{'='*70}\n")
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    model = YOLO(str(MODEL_PATH))
    print(f"✓ Loaded YOLO model with {len(model.names)} classes")
    print(f"  Classes: {list(model.names.values())}\n")
    
    # Load oak0 camera service config
    config_path = Path(__file__).resolve().parents[2] / 'camera_client' / 'service_config.json'
    
    if not config_path.exists():
        print(f"❌ oak0 service config not found: {config_path}")
        return
    
    config = proto_from_json_file(config_path, EventServiceConfig())
    print(f"✓ Loaded oak0 service config: {config.host}:{config.port}")
    
    # Create subscription to oak0 RGB stream
    subscription = SubscribeRequest(
        uri=Uri(path="/rgb", query="service_name=oak/0"),
        every_n=1  # Process every frame
    )
    print(f"✓ Created subscription: /rgb on service_name=oak/0\n")
    
    # Subscribe and process frames
    client = EventClient(config)
    print("🚀 Starting detection loop...")
    print("   Press Ctrl+C to exit\n")
    
    frame_count = 0
    fps_start_time = time.time()
    total_detections = 0
    
    try:
        async for event, message in client.subscribe(subscription, decode=True):
            frame_count += 1
            
            # Decode image from gRPC message
            image = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"⚠️  Frame {frame_count}: Failed to decode")
                continue
            
            # Run YOLO inference (ONLY class 0 = Collar)
            # IMPORTANT: Specify imgsz=640 to match TensorRT engine input size
            inference_start = time.time()
            results = model.predict(
                image,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=[0],  # Only detect class 0 (Collar)
                verbose=False,
                imgsz=640
            )
            inference_time = time.time() - inference_start

            # Process detections
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])

                    # Double-check: only keep class 0 (Collar)
                    if cls != 0:
                        continue

                    conf = float(boxes.conf[i])
                    xyxy = boxes.xyxy[i].cpu().numpy()

                    detections.append({
                        'class': 'Collar',
                        'confidence': conf,
                        'bbox': xyxy
                    })
            
            total_detections += len(detections)
            
            # Draw detections on image
            annotated_frame = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox'].astype(int)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display
            cv2.namedWindow("oak0 Detection Test", cv2.WINDOW_NORMAL)
            cv2.imshow("oak0 Detection Test", annotated_frame)
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = frame_count / elapsed
                avg_inference = inference_time * 1000  # ms
                print(f"[{frame_count:4d}] FPS: {fps:.1f} | Inference: {avg_inference:.0f}ms | Detections: {len(detections)} | Total: {total_detections}")
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n✓ User requested exit")
                break
                
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    
    # Final stats
    elapsed = time.time() - fps_start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n{'='*70}")
    print("DETECTION TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Frames processed: {frame_count}")
    print(f"Total runtime:    {elapsed:.1f}s")
    print(f"Average FPS:      {fps:.1f}")
    print(f"Total detections: {total_detections}")
    print(f"{'='*70}\n")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(test_oak0_detection())
