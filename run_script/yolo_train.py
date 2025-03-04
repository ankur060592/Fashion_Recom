from ultralytics import YOLO
import os
from config import ROOT_PATH
if __name__ == "__main__":

    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    data_yml_path = os.path.join(ROOT_PATH, "data.yaml")
    model.train(data=data_yml_path, epochs=50, device=0, batch=8)# for CPU remove device

