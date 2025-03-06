import os
import base64
from google import genai
from dotenv import load_dotenv

from config import GEMINI_MODEL_NAME

# Load API key from environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


def analyze_outfit(image_path, detected_labels):
    """Sends detected fashion items and outfit details to Gemini API for structured analysis."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    detected_items_text = ", ".join(detected_labels)
    
    prompt = f"""
    Analyze the provided outfit based on detected clothing items, colors, and patterns. Then, assess the overall style and recommend an alternative outfit for better coordination.
    
    **Detected Clothing Items:** {detected_items_text}
    **Detected Colors:** Identify the dominant colors of each item.
    **Detected Patterns (if any):** Identify patterns such as stripes, floral, polka dots.
    **Occasion (if applicable):** Suggest if this outfit is suited for casual, formal, business, party, or seasonal wear.
    
    **Output Structure:**
    1️⃣ **Outfit Breakdown & Color Analysis**
      - List each clothing item with its detected color and pattern.
      - Identify the overall color scheme (neutral, bold, pastel, monochrome, contrasting).
      - Describe the fashion style (casual, business casual, vintage, high fashion, minimal, streetwear, etc.).
    
    2️⃣ **Style Insights & Harmony Check**
      - Evaluate color coordination (well-balanced, contrasting, mismatched).
      - Assess pattern harmony (clashing/matching, busy/minimalistic).
      - Determine if the outfit is suitable for the occasion/season.
      - Suggest minor improvements (e.g., layering, accessory choices).
    
    3️⃣ **Alternative Outfit Recommendation**
      - Suggest an improved version of the current outfit while maintaining its essence.
      - Offer a seasonal or occasion-based alternative.
      - Recommend better color combinations.
      - Suggest fabric/material changes (e.g., linen for summer, wool for winter).
      - Include accessory recommendations (shoes, belts, jewelry, bags).
    """
    
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME, contents=[
                {"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                ]}
            ]
    )
        
    return response.text if response else "Error retrieving insights. Please try again."
