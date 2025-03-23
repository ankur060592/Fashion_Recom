# AI Fashion Recommendation System

This repository contains an AI-powered fashion recommendation system that detects and analyzes fashion items in images using **YOLOv11** and **Gemini API**. The project integrates **Streamlit** for the user interface.

## Features

- Detects fashion items in images using **YOLOv11**.
- Provides AI-powered fashion recommendations using **Gemini API**.
- User-friendly **Streamlit UI** for interaction.
- Supports **GPU and CPU-based** processing.

## Result
![TA Fashion Lens AI] https://github.com/ankur060592/Fashion_Recom/blob/update_readme/streamlit/result_sample/UI_sample.gif

## Prerequisites

Ensure you have the following installed:

- Python 3.10
- CUDA (if using GPU for YOLOv11) - Optional

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

### 2. Train & Run YOLOv11 Model

#### **Using Pretrained Model**
Since you have already trained YOLOv11, ensure your best model is available at:
```
runs/detect/train5/weights/best.pt
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
#### **(Optional) For better training fashion_data.yaml()
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
│── run_script/fashion_analysis.py  # Gemini API + Prompt
│── run_script/yolo_inference.py  # YOLOv11
│── config.py  # Configuration (dataset paths, YOLO model path, output folder, Gemini API settings, temp image path etc)
│── output/  # Detected boxed images
│── data/colorful_fashion_dataset_for_object_detection/JPEGImages/  # Input images
│── env  # Gemini API key
```

## Project Flow

1️⃣ **User uploads an image** via Streamlit UI.
2️⃣ **YOLOv11 detects fashion items** (shirt, pants, hats, etc.).
3️⃣ **Detected items are boxed and saved** in `output/`.
4️⃣ **Each detected item is sent to Generative AI** (Gemini API) for analysis.
5️⃣ **AI provides fashion insights and recommendations**.
6️⃣ **Streamlit displays results** (original image, boxed image, and AI-generated suggestions).

## Dataset Used
- [Colorful Fashion Dataset for Object Detection (Kaggle)](https://www.kaggle.com/datasets)
- [FashionPedia Fashion Dataset for Object Detection (Kaggle)](https://www.kaggle.com/datasets)

## Acknowledgments

- [YOLOv11 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Gemini API](https://ai.google.dev/)

## Next Steps
- Consolidate all components into a single repository.  
- Investigate the integration of Ujwal's referenced graph-based approach with the "Complete the Look" feature. If it proves unsuitable, consider adding a shopping feature using the Google API.  
- Eliminate background noise from images to focus solely on the main subject, preferably utilizing the existing YOLO model.  
- Improve the UI by incorporating design enhancements from previously shared references and updating the Streamlit interface.  
- Optimize AI prompts to generate more accurate and insightful recommendations.  

