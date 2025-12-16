#!/usr/bin/env python3
"""Convert visual-prompted YOLOE model to TensorRT engine."""

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import numpy as np
import os
from pathlib import Path

# ========== CONFIGURATION ==========
# Add as many objects as you want:
training_data = [
    # img1.png
    {
        "image": "collarImages/img1.png",
        "box": [827, 912, 1124, 1090]
    },
    # img2.png
    {
        "image": "collarImages/img2.png",
        "box": [872, 409, 1100, 692]
    },
    {
        "image": "collarImages/img2.png",
        "box": [1241, 859, 1561, 1084]
    },
    # img3.png
    {
        "image": "collarImages/img3.png",
        "box": [788, 596, 1049, 903]
    },
    # img4.png
    {
        "image": "collarImages/img4.png",
        "box": [674, 778, 952, 1078]
    },
    {
        "image": "collarImages/img4.png",
        "box": [1115, 853, 1415, 1077]
    },
    # img5.png
    {
        "image": "collarImages/img5.png",
        "box": [1101, 440, 1352, 728]
    },
    {
        "image": "collarImages/img5.png",
        "box": [735, 409, 963, 663]
    },
    # img6.png
    {
        "image": "collarImages/img6.png",
        "box": [844, 704, 1118, 1010]
    },
    # img7.png
    {
        "image": "collarImages/img7.png",
        "box": [1059, 581, 1172, 720]
    },
    {
        "image": "collarImages/img7.png",
        "box": [337, 525, 438, 653]
    },
    {
        "image": "collarImages/img7.png",
        "box": [933, 123, 980, 171]
    },
    # img8.png
    {
        "image": "collarImages/img8.png",
        "box": [970, 294, 1037, 372]
    },
    {
        "image": "collarImages/img8.png",
        "box": [914, 78, 953, 116]
    },
    {
        "image": "collarImages/img8.png",
        "box": [527, 272, 585, 352]
    },
    # img9.png
    {
        "image": "collarImages/img9.png",
        "box": [1144, 627, 1278, 786]
    },
    {
        "image": "collarImages/img9.png",
        "box": [745, 142, 782, 192]
    },
    {
        "image": "collarImages/img9.png",
        "box": [214, 163, 253, 207]
    },
    {
        "image": "collarImages/img9.png",
        "box": [1051, 54, 1087, 94]
    },
    # img10.png
    {
        "image": "collarImages/img10.png",
        "box": [1154, 537, 1262, 684]
    },
    {
        "image": "collarImages/img10.png",
        "box": [601, 361, 683, 459]
    },
    {
        "image": "collarImages/img10.png",
        "box": [1603, 239, 1662, 311]
    },
]
# ===================================

print("="*70)
print("VISUAL PROMPT YOLOE → TensorRT Conversion")
print("="*70)
print(f"Base model: yoloe-11s-seg.pt")
print(f"Output: collarDetectionv3.engine")
print(f"Visual prompts: {len(training_data)} bounding boxes")
print("="*70)

model = YOLOE("yoloe-11s-seg.pt")

# Collect all bboxes and class IDs
all_bboxes = []
all_class_ids = []

for i, data in enumerate(training_data):
    all_bboxes.append(data["box"])
    all_class_ids.append(0)  # All boxes are the same class (ID 0)

visual_prompts = {
    'bboxes': np.array(all_bboxes),
    'cls': np.array(all_class_ids)
}

print("\n✓ Loading visual prompts into model...")

# Run prediction with visual prompts to embed them in the model
model.predict(
    training_data[0]["image"],
    refer_image=training_data[0]["image"],
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    conf=0.1
)

print("✓ Visual prompts loaded")
print("\nExporting to TensorRT engine...")
print("⚠️  This will take 5-10 minutes (TensorRT optimization)")

try:
    # Export with embedded prompts to TensorRT
    export_path = model.export(
        format="engine",
        imgsz=640,
        half=True,      # FP16 for Jetson
        device=0,       # GPU
        workspace=4,    # 4GB workspace
        simplify=True
    )

    # Rename to collarDetectionv3.engine
    final_path = Path(__file__).parent / "collarDetectionv3.engine"
    if Path(export_path).exists():
        os.rename(export_path, final_path)
        print(f"\n{'='*70}")
        print("✓ Conversion complete!")
        print(f"{'='*70}")
        print(f"Output: {final_path}")
        print(f"\nExpected performance:")
        print(f"  - ONNX CPU: 0.2 FPS")
        print(f"  - TensorRT: 10-30 FPS (50-150x faster!)")
        print(f"\nUsage:")
        print(f"  python test_oak0_alignment.py --model-path collarDetectionv3.engine")
        print("="*70)
    else:
        print(f"❌ Export failed - output file not found")

except Exception as e:
    print(f"\n❌ Export failed: {e}")
    print(f"\nTroubleshooting:")
    print(f"  1. Visual prompting may not support TensorRT export")
    print(f"  2. Try using a detection model instead of segmentation")
    print(f"  3. Consider training a proper custom model")
