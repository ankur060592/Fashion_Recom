import os

# Automatically detect the root directory
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Dataset paths (relative to ROOT_PATH)
DATA_FOLDER = os.path.join(ROOT_PATH, "data", "colorful_fashion_dataset_for_object_detection")
IMAGE_PATH = os.path.join(DATA_FOLDER, "JPEGImages")
ANNOTATION_PATH = os.path.join(DATA_FOLDER, "Annotations_txt")

# Best saved YOLO model
SAVED_MODEL_PATH = os.path.join(ROOT_PATH, "runs", "detect", "train4", "weights", "best.pt")

# YOLO output folder
OUTPUT_FOLDER = os.path.join(ROOT_PATH, "output", "detected_items")

# LM Studio API Settings
API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "2vasabi_-_qwen2-vl-llava150k-lora"    #"LLaVA-1.5-13B"


# Test image path (ensure the test_ai folder exists)
TEST_IMAGE_PATH = os.path.join(ROOT_PATH, "data", "test_ai", "6452.jpg")

# Temp path (create a writable temp directory inside the project)
TEMP_PATH = os.path.join(ROOT_PATH, "temp")
os.makedirs(TEMP_PATH, exist_ok=True)  # Ensure the temp directory exists
