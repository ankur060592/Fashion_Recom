import os
import sys

import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TEMP_PATH
from run_script.fashion_analysis import analyze_outfit
from run_script.yolo_inference import detect_fashion_items

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="👗 AI Fashion Advisor", page_icon="👠")

# Sidebar - Upload Image

ta_logo = Image.open("assets/tiger_logo.jpeg")
ta_logo.resize((300,300))
google_next_logo = Image.open("assets/google_next_logo_2.png")
gcp_logo = Image.open("assets/gcp_logo.webp")
gemini_logo = Image.open("assets/gemini_logo.png")

st.sidebar.image(ta_logo)
st.sidebar.markdown("#")
st.sidebar.markdown("Get personalized fashion recommendations using the power of Generative AI.")
st.sidebar.markdown("#")
st.sidebar.markdown("📤 Upload Your Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)
image_path = None  # User must upload an image
st.sidebar.markdown("#")
st.sidebar.header("Developed for")
st.sidebar.image(google_next_logo, use_container_width=True)
st.sidebar.header("Powered by")
st.sidebar.image([gcp_logo.resize((75,75)), gemini_logo.resize((150,75))])

if uploaded_file:
    os.makedirs(TEMP_PATH, exist_ok=True)
    image_path = os.path.join(TEMP_PATH, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.image(image_path, caption="Uploaded Image", use_container_width=True)

    # Layout: Two sections
    st.markdown(
        "<h1 style='text-align: center;'>🎨 AI Fashion Analysis</h1>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 1])  # Equal width for better balance

    # Left Pane - Detection Output
    with col1:
        st.subheader("👀 Detected Outfit")
        detected_labels = []
        if image_path:
            output_image, detected_labels = detect_fashion_items(image_path)
            st.image(
                output_image, caption="Detected Fashion Items", use_container_width=True
            )
            st.markdown(
                f"**👗 Detected Look:** {', '.join(detected_labels)}",
                unsafe_allow_html=True,
            )

    # Right Pane - Personas Selection
    with col2:
        st.subheader("💡 Fashion AI Insights")

        # Initialize session state for persona
        if "persona" not in st.session_state:
            st.session_state.persona = None

        # Buttons to select persona
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("🔥 Style Roast/Compliment"):
            st.session_state.persona = "Style Roast/Compliment"
        if col_btn2.button("✨ Complete the Look"):
            st.session_state.persona = "Complete the Look"
        if col_btn1.button("🎭 Dress the Occasion"):
            st.session_state.persona = "Dress the Occasion"
        if col_btn2.button("💬 Ask Me Anything"):
            st.session_state.persona = "Ask Me Anything (Fashion Edition)"

        # Input fields appear based on selected persona
        user_input = ""
        if st.session_state.persona == "Dress the Occasion":
            user_input = st.text_input("🎭 Enter the Occasion:")
        elif st.session_state.persona == "Ask Me Anything (Fashion Edition)":
            user_input = st.text_area("💬 Ask your fashion-related question:")

        # AI Analysis Button
        if st.session_state.persona:
            with st.spinner("🧵 Analyzing Fashion... Please wait!"):
                if (
                    st.session_state.persona
                    in ["Dress the Occasion", "Ask Me Anything (Fashion Edition)"]
                    and not user_input
                ):
                    st.warning("⚠️ Please enter input for this persona.")
                else:
                    result = analyze_outfit(
                        image_path,
                        detected_labels,
                        st.session_state.persona,
                        user_input,
                    )
                    st.success("✅ Analysis Complete!")
                    st.markdown("### AI Fashion Analysis 🎭")
                    st.write(result)
