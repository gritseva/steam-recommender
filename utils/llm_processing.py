# File: utils/llm_processing.py

import json
import logging
import sys
import os
from typing import List, Union, Dict
import re

from models.transformer_model import load_transformer_model
from config.config import BASE_DIR

# Add project root to sys.path for imports if running standalone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def move_to_device(batch, device):
    return {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in batch.items()
    }


def extract_game_titles(user_message: str, context) -> List[str]:
    """
    Use the LLM to pull out raw game titles from the user's text.
    """
    if not isinstance(user_message, str):
        logger.error(
            f"extract_game_titles got a {type(user_message)}; skipping.")
        if isinstance(user_message, list):
            user_message = " ".join(str(x) for x in user_message)
        else:
            return []

    tokenizer = context.bot_data.get("tokenizer")
    model = context.bot_data.get("transformer_model")
    if not tokenizer or not model:
        logger.error("LLM model/tokenizer missing in context.bot_data.")
        return []

    prompt = (
        "[INST] Extract the video game titles from this message.\n"
        f"Message: \"{user_message}\"\n\n"
        "Reply with ONLY the game titles, separated by commas. No extra text.\n"
        "Example: The Witcher 3, Cyberpunk 2077 [/INST]"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = move_to_device(inputs, model.device) if not hasattr(
            inputs, 'to') else inputs.to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # If it echoed the [INST]… tags, chop them off
        if "[/INST]" in result:
            result = result.split("[/INST]", 1)[1].strip()

        # Split on commas and filter out any garbage
        titles = [t.strip() for t in result.split(",")]
        # Filter out non-title content and LLM fallback messages
        titles = [t for t in titles if t and not any(x in t.lower() for x in [
            'message:', 'example:', 'output:', 'format:', 'no video game titles'
        ]) and 'no video game titles' not in t.lower()]
        return titles

    except Exception:
        logger.exception("Error in extract_game_titles")
        return []


def infer_user_preferences_with_llm(user_message: str, context) -> Dict[str, Union[List[str], None]]:
    """
    Extract liked_games, genres, and excluded_tags via the LLM.
    """
    tokenizer = context.bot_data.get("tokenizer")
    model = context.bot_data.get("transformer_model")
    if not tokenizer or not model:
        logger.error("LLM model/tokenizer missing in context.bot_data.")
        return {"liked_games": [], "genres": [], "excluded_tags": []}

    prompt = (
        "[INST] Extract the user's game preferences from this message.\n\n"
        f"Message: \"{user_message}\"\n\n"
        "Output a JSON object with keys: liked_games, genres, excluded_tags.\n"
        "Example: {\"liked_games\": [\"Game A\"], \"genres\": [\"Action\"], \"excluded_tags\": []} [/INST]"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300
        )
        inputs = move_to_device(inputs, model.device) if not hasattr(
            inputs, 'to') else inputs.to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.pad_token_id
        )

        full = tokenizer.decode(output[0], skip_special_tokens=True)

        # 1) strip off the echo of the prompt
        if "[/INST]" in full:
            full = full.split("[/INST]", 1)[1].strip()

        # 2) now grab from the very first '{' through the very last '}'
        start = full.find("{")
        end = full.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in model output")
        json_blob = full[start:end]

        prefs = json.loads(json_blob)

    except Exception as e:
        logger.warning(f"LLM JSON parse failed ({e}), returning empty prefs")
        prefs = {}

    # If liked_games missing or empty, fall back to extract_game_titles
    if not prefs.get("liked_games"):
        prefs["liked_games"] = extract_game_titles(user_message, context)

    prefs.setdefault("genres", [])
    prefs.setdefault("excluded_tags", [])

    logger.info(f"Inferred preferences: {prefs}")
    return prefs


def parse_user_intent(user_message: str, context) -> str:
    """
    Classify user intent into one of our categories, using few-shot examples
    (including a “tell me more about” case) so we don’t need any hard-coded rules.
    """
    tokenizer = context.bot_data.get("tokenizer")
    model = context.bot_data.get("transformer_model")
    if not tokenizer or not model:
        logger.error("LLM model/tokenizer missing in context.bot_data.")
        return "unknown"

    categories = [
        "greeting", "recommend_games", "feedback", "additional_info", "out_of_context",
        "game_comparison", "opinion_request", "top_games_request", "video_search",
        "user_gaming_profile", "translation", "price_tracker", "gaming_news",
        "genre_exploration", "content_filter", "game_session_reminder"
    ]

    # Few‐shot examples to anchor “tell me more about …” into additional_info
    examples = [
        ("Hi!", "greeting"),
        ("Tell me a joke", "out_of_context"),
        ("What's the weather like today?", "out_of_context"),
        ("How do I cook pasta?", "out_of_context"),
        ("What's the capital of France?", "out_of_context"),
        ("Can you recommend some new RPGs?", "recommend_games"),
        ("I want game recommendations similar to Minecraft", "recommend_games"),
        ("My steam ID is 76561198129676583", "additional_info"),
        ("Here's my Steam profile: https://steamcommunity.com/id/example", "additional_info"),
        ("Compare Portal 2 and Half-Life.", "game_comparison"),
        ("What do you think about Stardew Valley?", "opinion_request"),
        ("What are the top RPG games?", "top_games_request"),
        ("Show me a video of Minecraft gameplay", "video_search"),
        ("Track the price of Cyberpunk 2077", "price_tracker"),
        ("Remind me to play Portal 2 at 8 PM", "game_session_reminder"),
        ("I don't want to see violent games", "content_filter"),
        ("Hola, quiero recomendaciones de juegos", "translation"),
        ("I love the recommendations you gave me!", "feedback"),
        ("The price tracking feature is really useful", "feedback"),
        ("Are you a real person?", "out_of_context"),
        ("Give me gaming news", "gaming_news"),
        ("Show me the best games in the adventure genre", "genre_exploration"),
        ("What's the weather like today?", "out_of_context")
    ]

    # Build the prompt
    prompt_lines = ["[INST] Classify this message into exactly one category.",
                    "Reply with ONLY the single category name, in lowercase.",
                    "\nExamples:"]
    for msg, cat in examples:
        prompt_lines.append(f"\"{msg}\" → {cat}")
    prompt_lines.append(
        f"\nNow classify this one:\nMessage: \"{user_message}\"")
    prompt_lines.append("\nCategories: " + ", ".join(categories))
    prompt_lines.append("[/INST]")
    prompt = "\n".join(prompt_lines)

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = move_to_device(inputs, model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(
            output[0], skip_special_tokens=True).strip().lower()

        # pick the category whose last occurrence is furthest right
        best_cat, best_idx = "unknown", -1
        for cat in categories:
            idx = result.rfind(cat)
            if idx > best_idx:
                best_idx, best_cat = idx, cat

        return best_cat

    except Exception:
        logger.exception("Error in parse_user_intent")
        return "unknown"


def test_llm_json_response(user_message: str):
    """
    Quick CLI test harness.
    """
    tokenizer, model = load_transformer_model()
    if not tokenizer or not model:
        print("❌ Model/tokenizer failed to load.")
        return

    prompt = (
        "[INST] Extract the user's game preferences from this message.\n\n"
        f"Message: \"{user_message}\"\n\n"
        "Output a JSON object with keys: liked_games, genres, excluded_tags.\n"
        "Example: {\"liked_games\": [\"Game A\"], \"genres\": [\"Action\"], \"excluded_tags\": []} [/INST]"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=300
    )
    inputs = move_to_device(inputs, model.device) if not hasattr(
        inputs, 'to') else inputs.to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.2,
        pad_token_id=tokenizer.pad_token_id
    )
    raw = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n--- RAW MODEL OUTPUT ---\n")
    print(raw)

    # clean off the prompt echo
    cleaned = raw
    if "[/INST]" in cleaned:
        cleaned = cleaned.split("[/INST]", 1)[1].strip()

    # slice from first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    snippet = cleaned[start:end] if start != -1 and end != -1 else cleaned

    print("====\nSNIPPET TO PARSE:\n", repr(snippet), "\n====")

    print("\n--- JSON PARSING RESULT ---\n")
    try:
        data = json.loads(snippet)
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"❌ JSON parse error: {e}\nExtracted snippet:\n{snippet}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LLM JSON output for user preference extraction."
    )
    parser.add_argument('--message', type=str, required=True,
                        help='User message to test.')
    args = parser.parse_args()
    test_llm_json_response(args.message)
