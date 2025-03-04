# AI Fashion Recommendation System

This repository contains an AI-powered fashion recommendation system that detects and analyzes fashion items in images using **YOLOv8** and **LLaVA (Large Language and Vision Assistant)**. The project integrates **LM Studio** for Generative AI-based fashion recommendations and **Streamlit** for the user interface.

## Features

- Detects fashion items in images using **YOLOv8**.
- Provides AI-powered fashion recommendations using **LLaVA v1.5 via LM Studio**.
- User-friendly **Streamlit UI** for interaction.
- Supports **GPU and CPU-based** processing.

## Prerequisites

Ensure you have the following installed:

- Python 3.10
- CUDA (if using GPU for YOLOv8) - Optional

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ankur060592/Fashion_Recom.git
cd Fashion_recom
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install LM Studio

1. Download and install **LM Studio** from [LM Studio's official site](https://lmstudio.ai/).
2. Follow the instructions to set up an **API server** for local inference.
3. Enable the API Server (Set it to http://localhost:1234).
4. Ensure **LLaVA v1.5** is running in LM Studio.

## Running the Project

### 1. Prepare YOLO Dataset

Ensure your dataset is located at:
```
Fashion_recom/data/colorful_fashion_dataset_for_object_detection/JPEGImages/
```

Run the dataset preparation script (if required):
```bash
python run_script/prepare_yolo_dataset.py
```

### 2. Train & Run YOLOv8 Model

#### **Using Pretrained Model**
Since you have already trained YOLOv8, ensure your best model is available at:
```
runs/detect/train4/weights/best.pt
```

#### **(Optional) Train YOLOv8 from Scratch**
**With GPU:**
```bash
python run_script/yolo_train.py
```

**Without GPU (CPU Mode):**
Modify `yolo_train.py` to use CPU:
```python
model = YOLO("yolov8n.pt")
data_yml_path = os.path.join(ROOT_PATH, "data.yaml")
model.train(data=data_yml_path, epochs=50, batch=8)
```
Then, run:
```bash
python run_script/yolo_train.py
```

### 3. Run the Streamlit App

Once the model is ready, launch the UI:
```bash
streamlit run streamlit/app.py
```

## Project Structure
```
Fashion_recom/
│── streamlit/app.py  # Streamlit UI
│── run_script/fashion_analysis.py  # YOLOv8 + LM Studio API logic
│── config.py  # Configuration (dataset paths, YOLO model path, output folder, LM Studio API settings, test image path)
│── output/  # Detected cropped images
│── data/colorful_fashion_dataset_for_object_detection/JPEGImages/  # Input images
```

## Project Flow

1️⃣ **User uploads an image** via Streamlit UI.
2️⃣ **YOLOv8 detects fashion items** (shirt, pants, hats, etc.).
3️⃣ **Detected items are cropped and saved** in `output/`.
4️⃣ **Each detected item is sent to Generative AI** (LLaVA/GPT-4 Vision via LM Studio) for analysis.
5️⃣ **AI provides fashion insights and recommendations**.
6️⃣ **Streamlit displays results** (original image, boxed image, and AI-generated suggestions).

## Dataset Used
- [Colorful Fashion Dataset for Object Detection (Kaggle)](https://www.kaggle.com/datasets)

## Acknowledgments

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [LLaVA - Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)
- [LM Studio](https://lmstudio.ai/)

