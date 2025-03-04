# config.py

# Input and output directories 
ROOT_PATH = "C:/Work/GenAI/Fashion_Recom"

#dataset path
DATA_FOLDER = "C:/Work/GenAI/Fashion_Recom/data/colorful_fashion_dataset_for_object_detection"
IMAGE_PATH = 'C:/Work/GenAI/Fashion_Recom/data/colorful_fashion_dataset_for_object_detection/JPEGImages/'
ANNOTATION_PATH  = 'C:/Work/GenAI/Fashion_Recom/data/colorful_fashion_dataset_for_object_detection/Annotations_txt/'

#best saved Model
SAVED_MODEL_PATH = "C:/Work/GenAI/Fashion_Recom/runs/detect/train4/weights/best.pt"

#YOLO output folder
OUTPUT_FOLDER = "C:/Work/GenAI/Fashion_Recom/output/detected_items"

# LM Studio API Settings
API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "LLaVA-1.5-13B"

TEST_IMAGE_PATH = "C:/Work/GenAI/Fashion_Recom/data/test_ai/6452.jpg"

# Change temp_path to a user-writable directory
TEMP_PATH = "C:/Users/ankur_qyibdp/Documents/FashionRecomTemp"