import sys
import os
import cv2
import torch
from ultralytics import YOLO
from config import SAVED_MODEL_PATH, OUTPUT_FOLDER

# Ensure Python finds 'run_script'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load trained YOLOv8 model
model = YOLO(SAVED_MODEL_PATH)  # Load trained model

def detect_fashion_items(image_path):
    """Detects and extracts fashion items from an image using YOLOv8."""
    results = model(image_path, save=True, conf=0.5)

    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)

    detected_items = []
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        class_id = int(box.cls[0])  
        label = model.names[class_id]  

        # Crop detected item
        cropped_item = image[y1:y2, x1:x2]
        save_path = os.path.join(OUTPUT_FOLDER, f"{label}_{i}.jpg")
        cv2.imwrite(save_path, cropped_item)
        detected_items.append((save_path, label))

    print(f"✅ Extracted {len(detected_items)} fashion items.")
    return detected_items  # Return list of detected items
