#!/usr/bin/env python3
"""Convert YOLOv8 .pt model to ONNX format."""

from ultralytics import YOLOE

# Load the YOLOe-11s model
model = YOLOE("best1.pt")

# Export to ONNX format
model.export(format="onnx", imgsz=640, simplify=True)

print("✓ Conversion complete: best.onnx")
    