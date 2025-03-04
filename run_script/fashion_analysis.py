import os
import base64
import requests

from config import API_URL, MODEL_NAME

def analyze_outfit(image_path, detected_labels):
    """Sends detected fashion items and outfit details to LLaVA for structured analysis."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    # Constructing structured input for better analysis
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
      - Describe the fashion style (casual, business casual, vintage, high fashion, minimal, streetwear, etc.).
    
    2️⃣ **Style Insights & Harmony Check**
      - Evaluate color coordination (well-balanced, contrasting, mismatched).
      - Assess pattern harmony (clashing/matching, busy/minimalistic).
    
    3️⃣ **Alternative Outfit Recommendation**
      - Suggest an improved version of the current outfit while maintaining its essence.
      - Recommend better color combinations.
      - Include accessory recommendations (shoes, belts, jewelry, bags).
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a professional fashion stylist providing detailed outfit analysis and recommendations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "images": [base64_image]
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error retrieving insights. Please try again."
