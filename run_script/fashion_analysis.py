import os
import base64
from google import genai
from dotenv import load_dotenv

from config import GEMINI_MODEL_NAME

# Load API key from environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

def analyze_outfit(image_path, detected_labels, persona, user_input=None):
    """Generate AI fashion insights based on selected persona."""
    """Sends detected fashion items and outfit details to Gemini API for structured analysis."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    if persona == "Style Roast/Compliment":
        prompt = f"Alright, let's get down to business! Roast or complement this outfit based on its style, color, and coordination: {', '.join(detected_labels)}"
    
    elif persona == "Complete the Look":
        prompt = f"Okay, let's accessorize! Analyze the outfit and suggest missing elements that would complete the look: {', '.join(detected_labels)}"
    
    elif persona == "Dress the Occasion":
        if user_input:
            prompt = f"Given the occasion '{user_input}', suggest how the outfit fits and what adjustments can be made: {', '.join(detected_labels)}"
        else:
            return "Please enter an occasion to get recommendations."
    
    elif persona == "Ask Me Anything (Fashion Edition)":
        if user_input:
            prompt = f"Fashion Q&A mode activated! Answer this question: {user_input}"
        else:
            return "Please enter your fashion-related question."
    
    else:
        return "Invalid persona selection."
    
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME, contents=[
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
    )
    
    return response.text if response else "Couldn't generate a response. Try again!"
