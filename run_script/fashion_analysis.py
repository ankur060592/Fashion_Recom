import base64
import requests
from config import API_URL, MODEL_NAME

def analyze_outfit(image_path, detected_labels):
    """Sends the entire outfit image to LLaVA for a unified fashion recommendation."""
    
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Create a prompt that includes all detected clothing items
    prompt = (
        f"You are a professional fashion stylist. Analyze this outfit based on the detected clothing items: {', '.join(detected_labels)}. "
        f"Describe the overall style, color coordination, and suggest ways to improve the outfit. "
        f"Recommend styling tips and possible accessories to enhance the look."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a fashion AI expert providing outfit recommendations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 700,
        "images": [base64_image]
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "⚠️ Error retrieving outfit recommendation."
