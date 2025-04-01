import base64
import os

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL_NAME

# Load API key from environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


def analyze_fashion_style(outfit_description):
    """
    Generates either a style roast OR a compliment based on the detected fashion items.
    """
    style_roast_prompt = """
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
    return f"{style_roast_prompt}\n\nOutfit Details: {outfit_description}"


def generate_common_prompt():
    """
    Generates a common prompt for fashion analysis.
    """
    return """You are a fashion expert with a keen eye for style.
    Analyze the outfit in this image using the **labels**, focusing on the **style, colors,
    and potential wearability** for a **specific occasion** or **weather condition**.

    **Important:** Do not include any bounding box details in your response. Focus solely on the fashion insights and recommendations.
    """


def analyze_outfit(image_path, detected_labels, persona, user_input=None):
    """Generate AI fashion insights based on selected persona."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Extract only the label names, ignoring bounding box data
    print("Detected Labels:", detected_labels)
    if not detected_labels:
        return "No fashion items detected. Please try again with a clearer image."
    outfit_description = (
        " ".join(detected_labels) if detected_labels else "A random casual outfit"
    )

    common_prompt = generate_common_prompt()

    if persona == "Style Roast/Compliment":
        prompt = analyze_fashion_style(outfit_description)
    elif persona == "Complete the Look":
        prompt = f"""
        {common_prompt}
        Suggest **one or two additional fashion elements** that would enhance the look.
        Ensure the suggestions are relevant to the **style, dressing sense, and visible accessories**.

        **Guidelines:**
        - Use keywords like 'suggest:', 'add:', 'pair with:', 'try out:'.
        - Provide **specific** and **descriptive** recommendations using keywords (e.g., "pair with: a classic leather wristwatch for a refined touch").
        - Consider **color coordination, accessory placement, and outfit balance**.

        Outfit Details: {outfit_description}
        """
    elif persona == "Dress the Occasion":
        if user_input:
            prompt = f"""
            {common_prompt}
            Given the occasion: '{user_input}', explain how the outfit fits and what adjustments can be made.

            **Example Response:**
            Photo of a man/woman in a garden, styled in a vintage, romantic outfit for a summer wedding.
            Consider soft pastels for dinner (use user_input).
            Suggest a flowing dress with floral embroidery and a delicate hat. The overall look should be elegant and whimsical.
            Keep it concise in **two to three lines**.

            Outfit Details: {outfit_description}
            """
        else:
            return "Please enter an occasion to get recommendations."
    elif persona == "Ask Me Anything (Fashion Edition)":
        if user_input:
            prompt = f"""
            {common_prompt}
            Fashion Q&A mode activated! Answer this question: {user_input}
            Keep it concise in **two to three lines** and actionable.

            Outfit Details: {outfit_description}
            """
        else:
            return "Please enter your fashion-related question."

    print("Prompt for Gemini:", prompt)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
                {"text": prompt},
            ],
        )
        print("Response from Gemini:", response)
        return response.text if response else "Couldn't generate a response. Try again!"
    except Exception as e:
        return f"An error occurred: {e}"
