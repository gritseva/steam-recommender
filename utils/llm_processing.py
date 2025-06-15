# File: utils/llm_processing.py

import json
import re
import logging
from typing import List, Union, Dict
from models.transformer_model import load_transformer_model

logger = logging.getLogger(__name__)

# Lazy load the transformer model and tokenizer
_tokenizer, _model = load_transformer_model()
if _tokenizer is None or _model is None:
    logger.error(
        "Failed to load transformer model and tokenizer. LLM functions may not work as expected.")


def extract_game_titles(user_message: str) -> List[str]:
    """
    Use the LLM to extract game titles from the user's message.

    Args:
        user_message (str): The input message from the user.

    Returns:
        List[str]: A list of game titles extracted from the message.
    """
    prompt = (
        "[INST] Extract the video game titles from this message.\n"
        f"Message: \"{user_message}\"\n\n"
        "Reply with ONLY the game titles, separated by commas. No additional text.\n"
        "Example: The Witcher 3, Cyberpunk 2077 [/INST]"
    )
    try:
        inputs = _tokenizer(prompt, return_tensors="pt",
                            padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        response = _model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id
        )
        result = _tokenizer.decode(
            response[0], skip_special_tokens=True).strip()
        if "[INST]" in result:
            result = result.split("[/INST]")[-1].strip()
        titles = [title.strip() for title in result.split(",")]
        filtered_titles = [
            t for t in titles
            if t and not any(x in t.lower() for x in ['message:', 'example:', 'output:', 'format:'])
        ]
        logger.info(f"Extracted game titles: {filtered_titles}")
        return filtered_titles
    except Exception as e:
        logger.error(f"Error extracting game titles: {e}")
        return []


def infer_user_preferences_with_llm(user_message: str) -> Dict[str, Union[List[str], None]]:
    """
    Use the LLM to extract user game preferences from the user's message.
    Expects a JSON object with keys: liked_games, genres, and excluded_tags.

    Args:
        user_message (str): The user's input message.

    Returns:
        Dict[str, Union[List[str], None]]: A dictionary of inferred preferences.
    """
    prompt = (
        "[INST] Extract the user's game preferences from this message.\n\n"
        f"Message: \"{user_message}\"\n\n"
        "Output a JSON object with the keys: liked_games, genres, excluded_tags.\n"
        "Example: {\"liked_games\": [\"Game A\"], \"genres\": [\"Action\"], \"excluded_tags\": []} [/INST]"
    )
    try:
        inputs = _tokenizer(prompt, return_tensors="pt",
                            padding=True, truncation=True, max_length=300)
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        response = _model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.2,
            pad_token_id=_tokenizer.pad_token_id
        )
        result_text = _tokenizer.decode(
            response[0], skip_special_tokens=True).strip()
        try:
            preferences = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Raw output: {result_text}")
            preferences = {}
        # Fallback: if liked_games key is missing or empty, extract game titles manually if possible
        if "liked_games" not in preferences or not preferences["liked_games"]:
            if any(word in user_message.lower() for word in ["love", "enjoy", "like"]):
                preferences["liked_games"] = extract_game_titles(user_message)
            else:
                preferences["liked_games"] = []
        preferences.setdefault("genres", [])
        preferences.setdefault("excluded_tags", [])
        logger.info(f"Inferred user preferences: {preferences}")
        return preferences
    except Exception as e:
        logger.error(f"Error inferring user preferences: {e}")
        return {"liked_games": extract_game_titles(user_message), "genres": [], "excluded_tags": []}


def parse_user_intent(user_message: str) -> str:
    """
    Classify the user's message into one of the predefined intent categories.

    Args:
        user_message (str): The user's message.

    Returns:
        str: The detected intent category.
    """
    prompt = (
        "[INST] Classify this message into one category. Reply with ONLY the category name.\n"
        f"Message: \"{user_message}\"\n\n"
        "Categories: recommend_games, feedback, additional_info, out_of_context, "
        "game_comparison, opinion_request, top_games_request, video_search, user_gaming_profile, "
        "translation, price_tracker, gaming_news, genre_exploration, content_filter, game_session_reminder\n"
        "[/INST]"
    )
    try:
        inputs = _tokenizer(prompt, return_tensors="pt",
                            padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        response = _model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id
        )
        result = _tokenizer.decode(
            response[0], skip_special_tokens=True).strip().lower()
        valid_intents = [
            "recommend_games", "feedback", "additional_info", "out_of_context",
            "game_comparison", "opinion_request", "top_games_request", "video_search",
            "user_gaming_profile", "translation", "price_tracker", "gaming_news",
            "genre_exploration", "content_filter", "game_session_reminder"
        ]
        for intent in valid_intents:
            if intent in result:
                logger.info(f"Detected user intent: {intent}")
                return intent
        logger.warning(f"Could not determine intent from output: {result}")
        return "unknown"
    except Exception as e:
        logger.error(f"Error parsing user intent: {e}")
        return "unknown"
