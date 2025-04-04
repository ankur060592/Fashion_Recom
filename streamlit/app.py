import base64
import os
import sys

from PIL import Image

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from threading import Thread

import cv2
from gtts import gTTS

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

    # CSS for fixed-size buttons
    st.markdown(
        """
        <style>
        .fixed-button {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 150px; /* Fixed width for buttons */
            height: 50px; /* Fixed height for buttons */
            font-size: 16px;
            border-radius: 8px;
            background-color: #444;
            color: white;
            border: none;
            cursor: pointer;
            transition: background 0.3s;
        }
        .fixed-button:hover {
            background-color: #666;
        }
        .wider-button {
            width: 300px; /* Wider width for the last row */
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Create columns for the buttons in three rows
    col1, col2 = st.columns(2)  # First row with 2 columns
    col3, col4 = st.columns(2)  # Second row with 2 columns
    col5 = st.columns(1)[0]  # Third row with 1 wider column

    # First row: Roast, Compliment
    with col1:
        if st.button(
            "🔥 Roast Me",
            key="Roast",
            help="Get a fashion roast!",
            use_container_width=True,
        ):
            st.session_state.persona = "Roast"

    with col2:
        if st.button(
            "💬 Compliment",
            key="Compliment",
            help="Receive a compliment!",
            use_container_width=True,
        ):
            st.session_state.persona = "Compliment"

    # Second row: Complete the Look, Dress for an Occasion
    with col3:
        if st.button(
            "✨ Complete the Look",
            key="Complete the Look",
            help="Complete your outfit!",
            use_container_width=True,
        ):
            st.session_state.persona = "Complete the Look"

    with col4:
        if st.button(
            "👔 Dress me for an Occasion",
            key="Dress the Occasion",
            help="Dress for any occasion!",
            use_container_width=True,
        ):
            st.session_state.persona = "Dress the Occasion"

    # Third row: Wider button for Ask Me Anything
    with col5:
        if st.button(
            "❓ Ask Me Anything",
            key="Ask Me Anything",
            help="Ask anything fashion-related!",
            use_container_width=True,
        ):
            st.session_state.persona = "Ask Me Anything"

    user_input = ""
    if st.session_state.persona == "Dress the Occasion":
        user_input = st.text_input("🎭 Enter the Occasion:")
    elif st.session_state.persona == "Ask Me Anything":
        user_input = st.text_area("💬 Ask your fashion-related question:")

    return st.session_state.persona, user_input


def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def text_to_speech(text, lang="en", slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow)
    audio_path = os.path.join(TEMP_PATH, "response.mp3")
    tts.save(audio_path)
    return audio_path


def display_ai_analysis(image_path, detected_labels, persona, user_input):
    prompt = ""
    audio_path = None

    if persona:
        with st.spinner("🧵 Analyzing Fashion... Please wait!"):
            if persona == "Dress for an Occasion" and not user_input:
                st.warning(
                    "⚠️ Please enter an occasion (e.g., 'Office Meeting', 'Casual Outing', 'Formal Dinner')."
                )
                return
            elif persona == "Ask Me Anything (Fashion Edition)" and not user_input:
                st.warning("⚠️ Please enter a fashion-related question.")
                return
            else:
                if persona == "Dress for an Occasion":
                    prompt = f"Suggest an outfit for the occasion: {user_input}"
                elif persona == "Ask Me Anything (Fashion Edition)":
                    prompt = user_input

                result = analyze_outfit(image_path, detected_labels, persona, prompt)

                # Convert the result to speech in a separate thread
                def generate_audio():
                    nonlocal audio_path
                    audio_path = text_to_speech(result)

                # Start audio generation in a separate thread
                audio_thread = Thread(target=generate_audio)
                audio_thread.start()

                # Wait for audio generation to complete
                audio_thread.join()

                st.success("✅ Analysis Complete!")
                st.markdown("### AI Fashion Analysis 🎭")
                st.write(result)
                st.audio(audio_path, format="audio/mp3", start_time=0)


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
    eye_lens_icon_path = os.path.join(ASSEST_PATH, "eye_lens_icon.png")
    # Set page configuration with custom icon
    st.set_page_config(
        layout="wide", page_title="Fashion Lens", page_icon=eye_lens_icon_path
    )
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
        # Display the custom icons with text
    # Display the title with the custom icon
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; align-items: center;'>
            <img src="data:image/png;base64,{get_image_base64(eye_lens_icon_path)}" style='height: 40px; margin-right: 10px;'/>
            <h1 style='margin: 0;'>f<span style='color: #FF69B4;'>A</span>sh<span style='color: #FF69B4;'>I</span>on lens</h1>
        </div>
        """,
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
