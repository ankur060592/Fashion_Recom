import os

# Get the root directory of the project
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# === Dataset Configuration === #
DATA_FOLDER = os.path.join(
    ROOT_PATH, "data", "colorful_fashion_dataset_for_object_detection"
)
IMAGE_PATH = os.path.join(DATA_FOLDER, "JPEGImages")
ANNOTATION_PATH = os.path.join(DATA_FOLDER, "Annotations_txt")

# === YOLO Model Paths === #
# You can switch between versions as needed
YOLO_TRAINED_MODELS_PATH = os.path.join(ROOT_PATH, "runs", "detect")
YOLO_MODEL_V1 = os.path.join(YOLO_TRAINED_MODELS_PATH, "train5", "weights", "best.pt")
YOLO_MODEL_FASHION = os.path.join(
    YOLO_TRAINED_MODELS_PATH, "train7", "weights", "best.pt"
)

# Set active model path (change here to switch model)
FASHION_SAVED_MODEL_PATH = YOLO_MODEL_FASHION

# === Output and Temp Paths === #
OUTPUT_FOLDER = os.path.join(ROOT_PATH, "output", "detected_items")
TEMP_PATH = os.path.join(ROOT_PATH, "temp")
os.makedirs(TEMP_PATH, exist_ok=True)

# === Assets (Streamlit Images, Logos) === #
ASSET_PATH = os.path.join(ROOT_PATH, "streamlit", "assets")

# === LLM Config === #
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# === Logging Configuration === #
LOG_DIR = os.path.join(ROOT_PATH, "log_files")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "fashion_lens.log")
