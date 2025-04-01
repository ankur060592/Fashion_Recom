import os

# Automatically detect the root directory
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Dataset paths (relative to ROOT_PATH)
DATA_FOLDER = os.path.join(
    ROOT_PATH, "data", "colorful_fashion_dataset_for_object_detection"
)
IMAGE_PATH = os.path.join(DATA_FOLDER, "JPEGImages")
ANNOTATION_PATH = os.path.join(DATA_FOLDER, "Annotations_txt")

# Best saved YOLO model
SAVED_MODEL_PATH = os.path.join(
    ROOT_PATH, "runs", "detect", "train5", "weights", "best.pt"
)

# Best saved YOLO model
FASHION_SAVED_MODEL_PATH = os.path.join(
    ROOT_PATH, "runs", "detect", "train7", "weights", "best.pt"
)

# YOLO output folder
OUTPUT_FOLDER = os.path.join(ROOT_PATH, "output", "detected_items")

# Gemini model name
GEMINI_MODEL_NAME = "gemini-2.0-flash"


# Temp path (create a writable temp directory inside the project)
TEMP_PATH = os.path.join(ROOT_PATH, "temp")
os.makedirs(TEMP_PATH, exist_ok=True)  # Ensure the temp directory exists
ASSEST_PATH = os.path.join(ROOT_PATH, "streamlit", "assets")
