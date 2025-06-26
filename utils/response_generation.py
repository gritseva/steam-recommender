# utils/response_generation.py
import re
import logging
import pandas as pd
from telegram import InlineKeyboardMarkup, InlineKeyboardButton

logger = logging.getLogger(__name__)


def move_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def generate_custom_response(raw_text: str, cleaning_patterns: list = None) -> str:
    """Cleans up raw generated text."""
    if cleaning_patterns is None:
        cleaning_patterns = [
            r'\[INST\].*?\[/INST\]',
            r'<s>|</s>',
        ]
    cleaned = raw_text
    for pattern in cleaning_patterns:
        cleaned = re.sub(pattern, ' ', cleaned,
                         flags=re.DOTALL | re.IGNORECASE).strip()

    if cleaned.startswith("You are a helpful game recommendation assistant"):
        return "I've found some games you might like! Check them out."

    if not cleaned or len(cleaned.split()) < 3:
        return "I've found some games you might like! Check them out."

    return cleaned


def generate_response(user_message: str, recommendations: pd.DataFrame, session, context, max_new_tokens: int = 200) -> str:
    """Generate a friendly recommendation response using the list of recommended games."""
    tokenizer = context.bot_data.get("tokenizer")
    model = context.bot_data.get("transformer_model")

    if not tokenizer or not model:
        logger.error("Transformer model unavailable in generate_response.")
        return "Based on what you told me, you might like these:\n\n" + "\n".join(
            [f"🎮 *{row['title']}*" for _, row in recommendations.iterrows()]
        )

    recommendation_list = "\n".join([
        f"- {row['title']}"
        for _, row in recommendations.head(3).iterrows()
    ])

    prompt = (
        "[INST] You are a friendly and enthusiastic game recommendation assistant. The user asked for games, and we found some. Briefly and conversationally introduce the recommendations. Mention one or two games by name. Keep it short (2-3 sentences).\n\n"
        f"User's request: \"{user_message}\"\n"
        "Games we found:\n"
        f"{recommendation_list}\n\n"
        "Your friendly response:[/INST]"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
        if hasattr(inputs, 'to'):
            inputs = inputs.to(model.device)
        else:
            inputs = move_to_device(inputs, model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        intro = generate_custom_response(generated_text)
        full_list = "\n\n" + "\n".join(
            [f"🎮 *{row['title']}*\n_{row.get('about_game', 'No description available.')[:150].strip()}..._\n" for _,
             row in recommendations.iterrows()]
        )
        return intro + full_list
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I found some great games for you! Here they are:"


def build_recommendation_keyboard(recommendations: pd.DataFrame) -> InlineKeyboardMarkup:
    """Builds an inline keyboard with Like/Dislike buttons for each recommendation."""
    keyboard = []
    for _, game in recommendations.head(5).iterrows():
        app_id = game['app_id']
        title = (game['title'][:20] + '..') if len(game['title']
                                                   ) > 22 else game['title']
        keyboard.append([
            InlineKeyboardButton(
                f"👍 Like {title}", callback_data=f"like:{app_id}"),
            InlineKeyboardButton(
                "👎 Dislike", callback_data=f"dislike:{app_id}")
        ])
    return InlineKeyboardMarkup(keyboard)
