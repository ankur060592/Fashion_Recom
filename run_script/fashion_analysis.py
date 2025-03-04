import base64
import requests
from config import API_URL, MODEL_NAME

def analyze_fashion_item(image_path, label):
    """Sends detected fashion items to LLaVA/GPT-4 Vision."""
    
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    prompt = f"Describe the {label} in this image. What are its colors, patterns, and style recommendations?"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Analyze this clothing item and provide fashion insights."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "images": [base64_image]
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error retrieving insights."
