import base64
import os

from dotenv import load_dotenv
from ebaysdk.finding import Connection as Finding
from google import genai

from config import GEMINI_MODEL_NAME

# Load API key from environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

EBAY_APP_ID = os.getenv("EBAY_APP_ID_new")
print(f"EBAY_APP_ID: {EBAY_APP_ID}")


def analyze_fashion_style(detected_labels):
    """
    Generates either a style roast OR a compliment based on the detected fashion items.
    """
    STYLE_ROAST_PROMPT = """
        You are a bold and witty fashion critic with a sharp eye for style.
        Your job is to analyze the given outfit and provide **either** a playful roast **or** a genuine compliment—**but not both**.
        Be expressive, engaging, and humorous. Keep it concise and actionable.

        ### Example Responses:
        **Roast:**
        "Hmm… interesting choice. That oversized jacket makes you look like you borrowed it from an NBA player.
        Maybe try a more fitted style for balance?"

        **Compliment:**
        "Wow, this is a look! The way the leather jacket complements your edgy vibe is pure fire. Rock on!"

        Now, **choose one**—either roast or compliment—and analyze the provided outfit details accordingly.
        """

    outfit_description = (
        ", ".join(detected_labels) if detected_labels else "A random casual outfit"
    )
    prompt = f"{STYLE_ROAST_PROMPT}\n\nOutfit Details: {outfit_description}"
    return prompt


def extract_suggested_items(response_text):
    """
    Extract suggested items from the AI response for the "Complete the Look" persona.
    """
    keywords = ["suggest:", "add:", "pair with:", "consider:", "try out:"]
    response_text = response_text.lower()

    suggested_items = []
    for line in response_text.split("\n"):
        if any(keyword in line.lower() for keyword in keywords):
            # Extract item names from the line, ensuring no bounding box details are included
            item = line.split(":")[-1].strip()
            if "box" not in item.lower():
                suggested_items.append(item)
    return suggested_items


def get_ebay_recommendations(suggested_items):
    """
    Retrieve product recommendations from eBay based on suggested items.
    """
    api = Finding(
        appid=EBAY_APP_ID, config_file=None, siteid="EBAY-US", compat_level=967
    )
    recommendations = []

    for item in suggested_items:
        response = api.execute("findItemsAdvanced", {"keywords": item})
        if response.reply.ack == "Success":
            for product in response.reply.searchResult.item:
                recommendations.append(
                    {
                        "name": product.title,
                        "price": product.sellingStatus.currentPrice.value,
                        "image_url": product.galleryURL,
                        "link": product.viewItemURL,
                    }
                )

    return recommendations


def analyze_outfit(image_path, detected_labels, persona, user_input=None):
    """Generate AI fashion insights based on selected persona."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    filtered_labels = filter_labels(detected_labels)

    if persona == "Style Roast/Compliment":
        return generate_ai_response(
            base64_image, analyze_fashion_style(filtered_labels)
        )

    elif persona == "Complete the Look":
        return complete_the_look(base64_image, filtered_labels)

    elif persona == "Dress the Occasion":
        if user_input:
            prompt = f"""Given the occasion '{user_input}', suggest how the outfit fits and
            what adjustments can be made: {', '.join(filtered_labels)}"""
            return generate_ai_response(base64_image, prompt)
        else:
            return "Please enter an occasion to get recommendations."

    elif persona == "Ask Me Anything (Fashion Edition)":
        if user_input:
            prompt = f"Fashion Q&A mode activated! Answer this question: {user_input}"
            return generate_ai_response(base64_image, prompt)
        else:
            return "Please enter your fashion-related question."


def filter_labels(detected_labels):
    """Filter out any labels that might contain bounding box details."""
    return [label for label in detected_labels if "box" not in label.lower()]


def generate_ai_response(base64_image, prompt):
    """Generate AI response using the Gemini API."""
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[
            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
            {"text": prompt},
        ],
    )
    return response.text if response else "Couldn't generate a response. Try again!"


def complete_the_look(base64_image, filtered_labels):
    """Handle the 'Complete the Look' persona."""
    prompt = f"""Analyze the outfit and suggest one or two missing elements that would complete the look.
            Use keywords like 'suggest:', 'add:', 'pair with:', 'consider:', 'try out:' in your response.
            For example: 'pair with: a silver necklace' or 'consider: a stylish belt'.
            Keep it concise and actionable: {', '.join(filtered_labels)}"""

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[
            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
            {"text": prompt},
        ],
    )

    if response:
        ai_response_text = response.text
        suggested_items = extract_suggested_items(ai_response_text)
        recommendations = get_ebay_recommendations(suggested_items)
        print(f"AI response: {ai_response_text}")
        print(f"Suggested items: {suggested_items}")
        print(f"Recommendations: {recommendations}")
        # Return the AI response followed by the shopping recommendations
        return f"{ai_response_text}\n\nShopping Recommendations:\n{recommendations}"
    else:
        return "Couldn't generate a response. Try again!"
