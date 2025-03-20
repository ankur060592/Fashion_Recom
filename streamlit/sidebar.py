import streamlit as st
from PIL import Image

def show_sidebar() -> None:
    ta_logo = Image.open("assets/tiger_logo.jpeg")
    ta_logo.resize((300,300))
    google_next_logo = Image.open("assets/google_next_logo_2.png")
    gcp_logo = Image.open("assets/gcp_logo.webp")
    gemini_logo = Image.open("assets/gemini_logo.png")
    st.sidebar.image(ta_logo)
    st.sidebar.markdown("#")
    st.sidebar.markdown("#")
    st.sidebar.markdown("Get personalized fashion recommendations using the power of Generative AI.")
    st.sidebar.markdown("#")
    st.sidebar.markdown("#")
    st.sidebar.markdown("#")
    st.sidebar.markdown("#")
    st.sidebar.header("Developed for")
    st.sidebar.image(google_next_logo, use_container_width=True)
    st.sidebar.markdown("#")
    st.sidebar.header("Powered by")
    st.sidebar.image([gcp_logo.resize((50,50)), gemini_logo.resize((100,50))])