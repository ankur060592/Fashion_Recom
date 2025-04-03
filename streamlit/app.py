import os
import sys

from PIL import Image

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

import cv2

from config import ASSEST_PATH, TEMP_PATH
from run_script.fashion_analysis import analyze_outfit
from run_script.yolo_inference import (
    detect_fashion_items,
    process_frame,
    update_consecutive,
)


def start_webcam_mode():
    """Handles live webcam video with YOLO detection and saves the best frame for AI analysis."""
    st.session_state.best_frame_captured = (
        False  # Reset flag for capturing the best frame
    )
    cap = cv2.VideoCapture(0)
    consecutive_detection_count = 0
    detection_wait_time = 5  # Number of seconds to wait for stable detection
    start_time = time.time()

    stframe = st.empty()  # For displaying the live webcam feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with YOLO detection
        detected, frame_with_boxes, detected_labels = process_frame(frame)

        # Display the live webcam feed (resize only for display purposes)
        stframe.image(frame_with_boxes, channels="BGR", use_container_width=True)

        # Update consecutive detection count
        consecutive_detection_count = update_consecutive(
            detected, consecutive_detection_count
        )

        # Check if detection is stable for a few seconds
        if (time.time() - start_time >= detection_wait_time) and detected_labels:
            # Save the original frame with boxes (not resized)
            best_frame_path = os.path.join(TEMP_PATH, "best_frame.jpg")
            cv2.imwrite(
                best_frame_path, frame_with_boxes
            )  # Save frame_with_boxes instead of resized version

            # Store the detected labels and best frame path
            st.session_state.best_frame_captured = True
            st.session_state.detected_labels = list(detected_labels)
            st.session_state.image_path = best_frame_path
            # Clear the live webcam feed display
            stframe.empty()
            break

    cap.release()
    cv2.destroyAllWindows()


def persona_buttons():
    st.subheader("💡 Fashion AI Insights")

    if "persona" not in st.session_state:
        st.session_state.persona = None

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("🔥 Style Roast/Compliment"):
        st.session_state.persona = "Style Roast/Compliment"
    if col_btn2.button("✨ Complete the Look"):
        st.session_state.persona = "Complete the Look"
    if col_btn1.button("🎭 Dress the Occasion"):
        st.session_state.persona = "Dress the Occasion"
    if col_btn2.button("💬 Ask Me Anything"):
        st.session_state.persona = "Ask Me Anything (Fashion Edition)"

    user_input = ""
    if st.session_state.persona == "Dress the Occasion":
        user_input = st.text_input("🎭 Enter the Occasion:")
    elif st.session_state.persona == "Ask Me Anything (Fashion Edition)":
        user_input = st.text_area("💬 Ask your fashion-related question:")

    return st.session_state.persona, user_input


def display_ai_analysis(image_path, detected_labels, persona, user_input):
    if persona:
        with st.spinner("🧵 Analyzing Fashion... Please wait!"):
            if persona == "Dress the Occasion" and not user_input:
                st.warning(
                    "⚠️ Please enter an occasion (e.g., 'Office Meeting', 'Casual Outing', 'Formal Dinner',)."
                )
                return
            elif persona == "Ask Me Anything (Fashion Edition)" and not user_input:
                st.warning("⚠️ Please enter a fashion-related question.")
                return
            else:
                result = analyze_outfit(
                    image_path, detected_labels, persona, user_input
                )
                st.success("✅ Analysis Complete!")
                st.markdown("### AI Fashion Analysis 🎭")
                st.write(result)


def display_detected_outfit(image_path):
    if st.session_state.input_mode == "Upload Image":
        if st.session_state.image_path:
            output_image, detected_labels = detect_fashion_items(image_path)
            st.image(
                output_image,
                caption="Detected Fashion Items",
                use_container_width=False,
                width=400,
            )
            st.markdown(
                f"**👗 Detected Look:** {', '.join(detected_labels)}",
                unsafe_allow_html=True,
            )
            # Store detected labels in session state
            st.session_state.detected_labels = detected_labels
            return detected_labels
    elif st.session_state.input_mode == "Live Webcam":
        st.sidebar.empty()
        if (
            "best_frame_captured" not in st.session_state
            or not st.session_state.best_frame_captured
        ):
            start_webcam_mode()
        if st.session_state.best_frame_captured and "image_path" in st.session_state:
            st.image(
                st.session_state.image_path,
                caption="Detected Fashion Items",
                use_container_width=False,
                width=400,
            )
            st.markdown(
                f"**👗 Detected Look:** {', '.join(st.session_state.detected_labels)}",
                unsafe_allow_html=True,
            )
            return st.session_state.detected_labels


def main():
    st.set_page_config(layout="wide", page_title="TA Fashion Lens AI", page_icon="👠")
    ta_logo = Image.open(os.path.join(ASSEST_PATH, "tiger_logo.jpg"))
    google_next_logo = Image.open(os.path.join(ASSEST_PATH, "google_next_logo_2.png"))
    gcp_logo = Image.open(os.path.join(ASSEST_PATH, "gcp_logo.webp"))
    gemini_logo = Image.open(os.path.join(ASSEST_PATH, "gemini_logo.png"))

    # Initialize session state variables
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Upload Image"
    if "image_path" not in st.session_state:
        st.session_state.image_path = None
    if "best_frame_captured" not in st.session_state:
        st.session_state.best_frame_captured = False
    if "detected_labels" not in st.session_state:
        st.session_state.detected_labels = []
    if "ai_analysis_output" not in st.session_state:
        st.session_state.ai_analysis_output = ""
    if "persona" not in st.session_state:
        st.session_state.persona = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Reset relevant session state variables when input mode changes or a new image is uploaded
    if st.session_state.input_mode != st.session_state.get(
        "previous_input_mode", None
    ) or (
        st.session_state.input_mode == "Upload Image" and st.session_state.image_path
    ):
        if "persona" not in st.session_state or not st.session_state.persona:
            st.session_state.persona = (
                None  # Keep existing persona instead of resetting every time
            )
            st.session_state.user_input = st.session_state.get("user_input", "")
            st.session_state.ai_analysis_output = ""

    # Update the previous input mode
    st.session_state.previous_input_mode = st.session_state.input_mode

    # Custom CSS to control the sidebar width
    st.markdown(
        """
        <style>
        .css-1d391kg {max-width: 250px; padding-top: 20px;}
        .css-1lcbmhc {max-width: 250px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image(ta_logo.resize((250, 80)))
        st.radio("Select Input Mode", ("Upload Image", "Live Webcam"), key="input_mode")

        if st.session_state.input_mode == "Upload Image":
            st.title("📤 Upload Your Image")
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

            if uploaded_file:
                os.makedirs(TEMP_PATH, exist_ok=True)
                image_path = os.path.join(TEMP_PATH, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.image_path = image_path

        st.header("Developed for")
        st.image(google_next_logo, use_container_width=True)
        st.header("Powered by")
        st.image([gcp_logo.resize((75, 75)), gemini_logo.resize((200, 75))])

    st.markdown(
        "<h1 style='text-align: center;'>👗 TA Fashion Lens AI</h1>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 1])

    if st.session_state.input_mode == "Upload Image" and st.session_state.image_path:
        with col1:
            detected_labels = display_detected_outfit(st.session_state.image_path)
        with col2:
            persona, user_input = persona_buttons()
            if detected_labels and persona:
                display_ai_analysis(
                    st.session_state.image_path,
                    detected_labels,
                    persona,
                    user_input,
                )

    elif st.session_state.input_mode == "Live Webcam":
        with col1:
            detected_labels = display_detected_outfit(None)
        with col2:
            persona, user_input = persona_buttons()
            if detected_labels and st.session_state.persona:
                display_ai_analysis(
                    st.session_state.image_path,
                    detected_labels,
                    st.session_state.persona,
                    user_input,
                )


if __name__ == "__main__":
    main()
