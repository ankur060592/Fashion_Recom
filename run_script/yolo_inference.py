import sys
import os
import cv2
import torch
from ultralytics import YOLO
from config import SAVED_MODEL_PATH, OUTPUT_FOLDER

# Ensure Python finds 'run_script'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load trained YOLO model
model = YOLO(SAVED_MODEL_PATH)

def detect_fashion_items(image_path):
    """Detects fashion items, draws bounding boxes on the original image, and saves the updated image."""
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Run YOLO inference
    results = model(image_path, conf=0.4)  

    # Load the original image
    image = cv2.imread(image_path)

    detected_labels = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        class_id = int(box.cls[0])  
        label = model.names[class_id]  
        
        # Draw bounding box on the original image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detected_labels.append(label)

    # Save the updated image with all detected items
    unified_image_path = os.path.join(OUTPUT_FOLDER, "outfit_combined.jpg")
    cv2.imwrite(unified_image_path, image)

    print(f"✅ Saved outfit image: {unified_image_path}")

    return unified_image_path, detected_labels  # Return image path & detected clothing labels
