# AI Fashion Recommendation System

This repository contains an AI-powered fashion recommendation system that detects and analyzes fashion items in images using **YOLOv11** and **Gemini API**. The project integrates **Streamlit** for a user-friendly interface.

## Features

- Detects fashion items in images using **YOLOv11**.
- Provides AI-powered fashion recommendations using **Gemini API**.
- User-friendly **Streamlit UI** for interaction.
- Supports **GPU and CPU-based** processing.

## Dashboard

![TA Fashion Lens AI](https://github.com/ankur060592/Fashion_Recom/blob/update_readme/streamlit/result_sample/UI_sample.gif)

### Project Structure

```
Fashion_recom/
│── streamlit/app.py                 # Streamlit UI
│── run_script/fashion_analysis.py    # Gemini API + Prompt
│── run_script/yolo_inference.py     # YOLOv11
│── config.py                         # Configuration
│── output/                           # Detected boxed images
│── data/colorful_fashion_dataset_for_object_detection/JPEGImages/  # Input images
│── .env                             # Environment variables
```

### Project Flow

1. **User uploads an image** via Streamlit UI.
2. **YOLOv11 detects fashion items** (shirt, pants, hats, etc.).
3. **Detected items are boxed and saved** in `output/`.
4. **Each detected item is sent to Generative AI** (Gemini API) for analysis.
5. **AI provides fashion insights and recommendations**.
6. **Streamlit displays results** (original image, boxed image, and AI-generated suggestions).

## Quick Start Guide

### To Directly Use the Application (UI)

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ankur060592/Fashion_Recom.git
   cd Fashion_recom
   ```

2. **Set Up Environment Variables**

   - Create a `.env` file in the root directory of the project.
   - Add your API keys to the `.env` file:

     ```env
     GEMINI_API_KEY=your_gemini_api_key
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**

   ```bash
   streamlit run streamlit/app.py
   ```

5. **Use the Application**

   - Open the provided URL in your browser.
   - Upload an image to get fashion recommendations.

## Docker Setup

To build and run the Docker image, follow these steps:

### Prerequisites

- Ensure Docker is installed on your system. You can download and install Docker from [here](https://www.docker.com/get-started).
- (Optional) If you have Make installed, you can use the Makefile to simplify the build and run process.

### Building the Docker Image

#### Using Make

If you have Make installed on your system, you can use the following commands to build and run the Docker image:

```bash
make docker-build
make docker-run
 ```

#### Using Terminal

   ```bash
docker build -t your-image-name .
docker run -p 8501:8501 your-image-name
   ```
#### If you get the API issue
   ```bash
docker run -p 8501:8501 -e GEMINI_API_KEY=your_api_key_here your-image-name
   ```

## For Developers

### Prerequisites

- Python 3.10
- CUDA (if using GPU for YOLOv11) - Optional

### Prepare YOLO Dataset

Ensure your dataset is located at:
```
Fashion_recom/data/colorful_fashion_dataset_for_object_detection/JPEGImages/
```

Run the dataset preparation script (if required):
```bash
python run_script/prepare_yolo_dataset.py
```

### Train & Run YOLOv11 Model

#### **Using Pretrained Model**

Ensure your best model is updated in the config (current best model is at this location):
```
runs/detect/train7/weights/best.pt
```

#### **(Optional) Train YOLOv11 from Scratch**

**With GPU:**
```bash
python run_script/yolo_train.py
```

**Without GPU (CPU Mode):**
Modify `yolo_train.py` to use CPU:
```python
model = YOLO("yolo11n.pt")
data_yml_path = os.path.join(ROOT_PATH, "data.yaml")
# Best Model was trained using Fashion_data using below yml
# To use this fashion data, download the data from Kaggle (3.7GB)
# Structure the data as per the fashion_data YML file in the root folder
(optional) data_yml_path = os.path.join(ROOT_PATH, "fashion_data.yaml")
model.train(data=data_yml_path, epochs=50, batch=8)
```
Then, run:
```bash
python run_script/yolo_train.py
```

## Dataset Used

- [Colorful Fashion Dataset for Object Detection (Kaggle)](https://www.kaggle.com/datasets)
- [FashionPedia Fashion Dataset for Object Detection (Kaggle)](https://www.kaggle.com/datasets)

## Acknowledgments

- [YOLOv11 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Gemini API](https://ai.google.dev/)

## Next Steps

- Add a shopping element to "Complete the Look."
- Enhance the interactivity of the live webcam feature.
- Error Handling