import os
import sys

from ultralytics import YOLO

from config import ROOT_PATH

# Ensure Python finds 'run_script'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    # model = YOLO("yolov8n.pt")  # Load YOLOv8'n' model used nano for faster training,
    #  we can change n,l,x for different models
    model = YOLO("yolo11n.pt")  # Latest YOLOv11 model
    data_yml_path = os.path.join(ROOT_PATH, "data.yaml")
    model.train(
        data=data_yml_path, epochs=50, device=0, imgsz=640
    )  # for CPU remove device
