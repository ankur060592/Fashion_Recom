import os
import sys

from PIL import Image

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

    stframe = st.empty()  # For displaying the live webcam feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with YOLO detection
        detected, frame_with_boxes, detected_labels = process_frame(frame)

        # Resize the frame to match the Upload Image display size
        frame_with_boxes_resized = cv2.resize(frame_with_boxes, (400, 300))

        # Display live webcam feed with YOLO detection
        stframe.image(frame_with_boxes_resized, channels="BGR", width=400)

        # Update consecutive detection count
        consecutive_detection_count = update_consecutive(
            detected, consecutive_detection_count
        )

        # Check if detection is stable for a few frames (e.g., 5 consecutive frames)
        if consecutive_detection_count >= 5 and detected_labels:
            # Save the best frame as an image for AI analysis
            best_frame_path = os.path.join(TEMP_PATH, "best_frame.jpg")
            cv2.imwrite(best_frame_path, frame)

            # Store the detected labels and best frame path
            st.session_state.best_frame_captured = True
            st.session_state.detected_labels = list(detected_labels)
            st.session_state.image_path = best_frame_path

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
            if (
                persona in ["Dress the Occasion", "Ask Me Anything (Fashion Edition)"]
                and not user_input
            ):
                st.warning("⚠️ Please enter input for this persona.")
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
            st.image(output_image, caption="Detected Fashion Items", width=400)
            st.markdown(
                f"**👗 Detected Look:** {', '.join(detected_labels)}",
                unsafe_allow_html=True,
            )
            return detected_labels
    elif st.session_state.input_mode == "Live Webcam":
        st.sidebar.empty()  # Clear the sidebar
        if (
            "best_frame_captured" not in st.session_state
            or not st.session_state.best_frame_captured
        ):
            start_webcam_mode()
        if st.session_state.best_frame_captured and "image_path" in st.session_state:
            st.image(
                st.session_state.image_path, caption="Detected Fashion Items", width=300
            )
            st.markdown(
                f"**👗 Detected Look:** {', '.join(st.session_state.detected_labels)}",
                unsafe_allow_html=True,
            )
            return st.session_state.detected_labels


def main():
    st.set_page_config(layout="wide", page_title="👗 AI Fashion Advisor", page_icon="👠")

    ta_logo = Image.open(os.path.join(ASSEST_PATH, "tiger_logo.jpg"))
    google_next_logo = Image.open(os.path.join(ASSEST_PATH, "google_next_logo_2.png"))
    gcp_logo = Image.open(os.path.join(ASSEST_PATH, "gcp_logo.webp"))
    gemini_logo = Image.open(os.path.join(ASSEST_PATH, "gemini_logo.png"))

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Upload Image"

    with st.sidebar:
        st.image(ta_logo.resize((300, 100)))
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
        "<h1 style='text-align: center;'>🎨 AI Fashion Analysis</h1>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 1])

    if (
        st.session_state.input_mode == "Upload Image"
        and "image_path" in st.session_state
    ):
        with col1:
            detected_labels = display_detected_outfit(st.session_state.image_path)
        with col2:
            persona, user_input = persona_buttons()
            if detected_labels:
                display_ai_analysis(
                    st.session_state.image_path, detected_labels, persona, user_input
                )

    elif st.session_state.input_mode == "Live Webcam":
        with col1:
            detected_labels = display_detected_outfit(None)
        with col2:
            persona, user_input = persona_buttons()
            if detected_labels:
                display_ai_analysis(
                    st.session_state.image_path, detected_labels, persona, user_input
                )


if __name__ == "__main__":
    main()
