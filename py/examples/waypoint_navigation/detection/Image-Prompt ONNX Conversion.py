from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import numpy as np
import os

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

# Run prediction with visual prompts to embed them in the model
model.predict(
    training_data[0]["image"],
    refer_image=training_data[0]["image"],
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    conf=0.1
)

# Export with embedded prompts
model.export(format="onnx", imgsz=640)

# Rename the exported model
os.rename("yoloe-11s-seg.onnx", "collarDetectionv3.onnx")

print("Training complete!")
print("Object mapping:")
for i, data in enumerate(training_data):
    print(f"  ID {i}: {data['image']}")
