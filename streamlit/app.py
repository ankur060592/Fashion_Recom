import sys
import os

# Ensure Python finds 'run_script'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from run_script.yolo_inference import detect_fashion_items
from run_script.fashion_analysis import analyze_fashion_item
from config import TEMP_PATH
st.title("👗 AI-Powered Fashion Advisor")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save the uploaded file
    # Ensure output directory exists
    os.makedirs(TEMP_PATH, exist_ok=True)
    image_path = os.path.join(TEMP_PATH, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    # Detect fashion items
    detected_items = detect_fashion_items(image_path)

    st.subheader("🖼 Detected Fashion Items & Recommendations")
    for item_path, label in detected_items:
        insight = analyze_fashion_item(item_path, label)
        st.image(item_path, caption=f"Detected: {label}", width=200)
        st.write("**Fashion Analysis:**", insight)
