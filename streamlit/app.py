import sys
import os

# Ensure Python finds the 'run_script' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import numpy as np
from config import TEMP_PATH

# Delay imports to avoid circular dependencies
def load_yolo():
    from run_script.yolo_inference import detect_fashion_items
    return detect_fashion_items

def load_ai():
    from run_script.fashion_analysis import analyze_outfit
    return analyze_outfit

# Set Streamlit page layout
st.set_page_config(layout="wide")

# Center-align Title
st.markdown("<h1 style='text-align: center;'>👗 AI-Powered Fashion Advisor</h1>", unsafe_allow_html=True)

# Create layout with three equal columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📤 Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    os.makedirs(TEMP_PATH, exist_ok=True)
    image_path = os.path.join(TEMP_PATH, uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run YOLO detection
    detect_fashion_items = load_yolo()
    boxed_image, detected_labels = detect_fashion_items(image_path)

    with col2:
        st.subheader("🖼 Detected Fashion Items")
        st.image(boxed_image, caption="Detected Outfit", use_container_width=True)

    # Run AI analysis and recommendations
    analyze_outfit = load_ai()
    outfit_recommendation = analyze_outfit(boxed_image, detected_labels)

    with col3:
        st.subheader("📌 Outfit Insights")
        st.write(outfit_recommendation.split("Recommendation:")[0])

        st.subheader("🎯 Fashion Recommendations")
        st.write("Recommendation:" + outfit_recommendation.split("Recommendation:")[-1])
