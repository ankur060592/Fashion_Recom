import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from config import FASHION_SAVED_MODEL_PATH, OUTPUT_FOLDER

# Ensure Python finds 'run_script'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load trained YOLO model
model = YOLO(FASHION_SAVED_MODEL_PATH)


def map_to_correct_label(labels):
    """Map similar labels to the most correct one."""
    label_priority = {
        "top": ["t-shirt", "sweatshirt", "shirt"],
        "jacket": ["coat"],
        "sleeve": ["sleeve"],
    }

    final_labels = set()
    for label in labels:
        found = False
        for correct_label, similar_labels in label_priority.items():
            if label in similar_labels:
                final_labels.add(correct_label)
                found = True
                break
        if not found:
            final_labels.add(label)

    return list(final_labels)


def detect_fashion_items(input_data):
    """Detects fashion items from either image path or video frame, draws bounding boxes, and returns unique labels."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if isinstance(input_data, str):  # Case 1: Image Path
        image = cv2.imread(input_data)
    elif isinstance(input_data, np.ndarray):  # Case 2: Frame from Webcam
        image = input_data
    else:
        raise ValueError("Invalid input data type. Expected file path or numpy array.")

    # Run YOLO inference
    results = model(image, conf=0.4)
    detected_labels = set()  # Use a set to store unique labels

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]

        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        detected_labels.add(label)  # Add label to the set

    # Save the updated image if it's a numpy array (Webcam frame)
    if isinstance(input_data, np.ndarray):
        unified_image_path = os.path.join(OUTPUT_FOLDER, "outfit_combined.jpg")
        cv2.imwrite(unified_image_path, image)
        return unified_image_path, list(detected_labels)

    # If it's an image path, return detected labels only
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_labels = map_to_correct_label(detected_labels)
    return image, list(detected_labels)


def process_frame(frame):
    """
    Processes a frame through the YOLO model. Draws bounding boxes and returns a boolean indicating
    whether a person was detected in the frame (with confidence above threshold).
    """
    label_detected_flag = False
    detected_labels = set()
    frame = cv2.resize(frame, (640, 640))
    # Run YOLO inference
    results = model(frame, conf=0.5, stream=True)
    # Loop over results from YOLO
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get class class index
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Check if label detected and meets confidence threshold
            if label:
                label_detected_flag = True
                # Draw bounding box and class name on frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                # Add label to set of detected labels
                detected_labels.add(label)
    detected_labels = map_to_correct_label(detected_labels)
    return label_detected_flag, frame, detected_labels


def update_consecutive(detected, cdc):
    """Update consecutive detection counter based on whether a person was detected."""
    consecutive_detection_count = cdc
    if detected:
        consecutive_detection_count += 1
    else:
        consecutive_detection_count = 0
    return consecutive_detection_count
