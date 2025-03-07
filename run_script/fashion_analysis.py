import base64
import os
import random

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL_NAME

# Load API key from environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


def analyze_outfit(image_path, detected_labels, persona, user_input=None):
    """Generate AI fashion insights based on selected persona."""
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    if persona == "Style Roast/Compliment":
        overall_feedback = [
            """This outfit has potential, but the pieces don’t fully complement each other.
              A little refinement could take it from good to great!""",
            """You're making a statement with this look! But is it the statement you intended?
              Some small tweaks could make it shine.""",
            """It's bold, it's unique, and it definitely has personality! With a few adjustments,
              this could be an iconic look.""",
            """This is an interesting mix! A little fine-tuning could help everything
             flow together seamlessly.""",
        ]

        suggestion_feedback = [
            "Consider balancing the outfit by adjusting the contrast or streamlining some elements.",
            "A minor change in layering or color coordination could enhance the overall aesthetic.",
            "Accessories or a different texture could help bring out the best in this combination.",
            "Think about the harmony of the pieces—sometimes, less is more!",
        ]

        overall_comment = random.choice(overall_feedback)
        improvement_suggestion = random.choice(suggestion_feedback)

        prompt = (
            f"AI Fashion Analysis 🎭\n"
            f"Here are my thoughts on this look:\n\n"
            f"Overall Impression: {overall_comment}\n\n"
            f"Suggestions:\n- {improvement_suggestion}\n\n"
            f"With a few strategic adjustments, this outfit could go from 'almost there' to 'fashion-forward!'"
        )
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
    print(prompt)
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[
            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
            {"text": prompt},
        ],
    )

    return response.text if response else "Couldn't generate a response. Try again!"
