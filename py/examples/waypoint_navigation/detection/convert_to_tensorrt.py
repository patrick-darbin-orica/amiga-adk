#!/usr/bin/env python3
"""Convert YOLO .pt model to TensorRT engine for GPU acceleration on Jetson."""

from ultralytics import YOLO
from pathlib import Path
import sys

# Model to convert
model_path = Path(__file__).parent / "yolo26n-pose.pt"

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)

print(f"{'='*70}")
print("YOLO → TensorRT Conversion")
print(f"{'='*70}")
print(f"Input:  {model_path.name}")
print(f"Output: {model_path.stem}.engine")
print(f"{'='*70}\n")

# Load model
print(f"Loading model...")
model = YOLO(str(model_path))

print(f"✓ Model loaded")
print(f"  Classes: {list(model.names.values())}")
print(f"  Number of classes: {len(model.names)}\n")

# Export to TensorRT engine
print(f"Exporting to TensorRT engine...")
print(f"⚠️  This will take 5-10 minutes on first run")
print(f"   (TensorRT compiles/optimizes for this specific GPU)\n")

try:
    export_path = model.export(
        format="engine",        # TensorRT engine
        imgsz=640,             # Must match inference size
        half=True,             # FP16 precision for Jetson
        device=0,              # GPU device 0
        workspace=4,           # 4GB workspace
        verbose=True,          # Show progress
        simplify=True,         # Simplify ONNX before TRT conversion
    )

    print(f"\n{'='*70}")
    print(f"✓ Conversion complete!")
    print(f"{'='*70}")
    print(f"Output file: {export_path}")
    print(f"\nExpected speedup: 20-100x faster than .pt file")
    print(f"  - .pt file:     0.2-0.3 FPS (CPU)")
    print(f"  - .engine file: 5-30 FPS (GPU)")
    print(f"\nNow update your script to use: {Path(export_path).name}")
    print(f"{'='*70}\n")

except Exception as e:
    print(f"\n❌ Export failed: {e}")
    print(f"\nTroubleshooting:")
    print(f"  1. Make sure CUDA/TensorRT are properly installed")
    print(f"  2. Check GPU is available: nvidia-smi")
    print(f"  3. Try ONNX format instead if TensorRT fails")
    sys.exit(1)
