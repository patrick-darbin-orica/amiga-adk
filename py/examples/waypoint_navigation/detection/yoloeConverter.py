from ultralytics import YOLO

# point to your trained weights (e.g., runs/detect/train*/weights/best.pt)
m = YOLO("/mnt/managed_home/farm-ng-user-patrick-orica/farm-ng-amiga/py/examples/waypoint_navigation/detection/yolov8nCones.pt")

onnx_path = m.export(
    format="onnx",
    imgsz=640,      # your training size
    opset=12,       # good for OpenVINO/RVC2
    simplify=True,
    dynamic=False   # usually best for blobconversion
)
print("Exported:", onnx_path)
