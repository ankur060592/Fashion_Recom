import os
import sys

import cv2

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TEMP_PATH
from run_script.fashion_analysis import analyze_outfit
from run_script.yolo_inference import (
    detect_fashion_items,
    process_frame,
    update_consecutive,
)

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="👗 AI Fashion Advisor", page_icon="👠")


def persona_buttons():
    """Display persona selection buttons and return the selected persona."""
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
    return st.session_state.persona


def display_ai_analysis(image_path, detected_labels):
    """Display AI Personas analysis and user interactions."""
    if (
        st.session_state.best_frame_captured
    ):  # Only show this if the best frame is captured
        st.subheader("💡 Fashion AI Insights")

        persona = persona_buttons()
        user_input = ""

        if persona == "Dress the Occasion":
            user_input = st.text_input("🎭 Enter the Occasion:")
        elif persona == "Ask Me Anything (Fashion Edition)":
            user_input = st.text_area("💬 Ask your fashion-related question:")

        if persona:
            with st.spinner("🧵 Analyzing Fashion... Please wait!"):
                if (
                    persona
                    in ["Dress the Occasion", "Ask Me Anything (Fashion Edition)"]
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


def display_detected_outfit():
    """Display the detected outfit for Image Upload or Live Webcam mode."""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👀 Detected Outfit")

        if st.session_state.input_mode == "Upload Image":
            if st.session_state.image_path:
                output_image, detected_labels = detect_fashion_items(
                    st.session_state.image_path
                )
                st.image(
                    output_image,
                    caption="Detected Fashion Items",
                    use_container_width=True,
                )
                st.markdown(
                    f"**👗 Detected Look:** {', '.join(detected_labels)}",
                    unsafe_allow_html=True,
                )
                st.session_state.detected_labels = detected_labels

        elif st.session_state.input_mode == "Live Webcam":
            st.sidebar.empty()  # Clear previous sidebar content
            start_webcam_mode()

    with col2:
        if st.session_state.best_frame_captured:
            display_ai_analysis(
                st.session_state.image_path, st.session_state.detected_labels
            )
        else:
            st.info(
                "📸 Waiting to capture the best frame... Keep your outfit in view for a few seconds."
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

        # Display live webcam feed with YOLO detection
        stframe.image(frame_with_boxes, channels="BGR", use_container_width=True)

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


def main():
    st.sidebar.title("🎥 Select Mode")
    st.sidebar.radio(
        "Choose Input Mode", ("Upload Image", "Live Webcam"), key="input_mode"
    )

    if st.session_state.input_mode == "Upload Image":
        st.sidebar.title("📤 Upload Your Image")
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            os.makedirs(TEMP_PATH, exist_ok=True)
            image_path = os.path.join(TEMP_PATH, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.image(
                image_path, caption="Uploaded Image", use_container_width=True
            )
            st.session_state.image_path = image_path
            st.session_state.best_frame_captured = True
            st.session_state.detected_labels = []

            st.markdown(
                "<h1 style='text-align: center;'>🎨 AI Fashion Analysis</h1>",
                unsafe_allow_html=True,
            )
            display_detected_outfit()

    elif st.session_state.input_mode == "Live Webcam":
        st.session_state.image_path = None  # Clear previous image
        st.session_state.best_frame_captured = False
        st.session_state.detected_labels = []

        st.markdown(
            "<h1 style='text-align: center;'>🎨 AI Fashion Analysis</h1>",
            unsafe_allow_html=True,
        )
        display_detected_outfit()


if __name__ == "__main__":
    main()
