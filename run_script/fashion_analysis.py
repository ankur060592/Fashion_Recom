import base64
import os

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL_NAME

# Load API key from environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


def analyze_fashion_style(detected_labels):
    """
    Generates either a style roast OR a compliment based on the detected fashion items.
    """
    STYLE_ROAST_PROMPT = """
        You are a bold and witty fashion critic with a sharp eye for style. 
        Your job is to analyze the given outfit and provide **either** a playful roast **or** a genuine compliment—**but not both**.
        Be expressive, engaging, and humorous. 

        ### Example Responses:
        **Roast:**  
        "Hmm… interesting choice. That oversized jacket makes you look like you borrowed it from an NBA player.
        Maybe try a more fitted style for balance?"

        **Compliment:**  
        "Wow, this is a look! The way the leather jacket complements your edgy vibe is pure fire. Rock on!"

        Now, **choose one**—either roast or compliment—and analyze the provided outfit details accordingly.
        """

    # Convert detected items into a descriptive format
    outfit_description = (
        ", ".join(detected_labels) if detected_labels else "A random casual outfit"
    )

    prompt = f"{STYLE_ROAST_PROMPT}\n\nOutfit Details: {outfit_description}"

    return prompt


def analyze_outfit(image_path, detected_labels, persona, user_input=None):
    """Generate AI fashion insights based on selected persona."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    if persona == "Style Roast/Compliment":
        prompt = analyze_fashion_style(detected_labels)
    elif persona == "Complete the Look":
        prompt = f"""Okay, let's accessorize! Analyze the outfit and
          suggest missing elements that would complete the look: {', '.join(detected_labels)}"""

    elif persona == "Dress the Occasion":
        if user_input:
            prompt = f"""Given the occasion '{user_input}', suggest how the outfit fits and
            what adjustments can be made: {', '.join(detected_labels)}"""
        else:
            return "Please enter an occasion to get recommendations."

    elif persona == "Ask Me Anything (Fashion Edition)":
        if user_input:
            prompt = f"Fashion Q&A mode activated! Answer this question: {user_input}"
        else:
            return "Please enter your fashion-related question."
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[
            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
            {"text": prompt},
        ],
    )

    return response.text if response else "Couldn't generate a response. Try again!"
