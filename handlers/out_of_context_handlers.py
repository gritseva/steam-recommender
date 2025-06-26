# handlers/out_of_context_handlers.py
import logging
from telegram import Update
from telegram.ext import CallbackContext
from sessions.session_manager import get_user_session
from utils.response_generation import generate_custom_response
import random

logger = logging.getLogger(__name__)


def move_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


async def handle_out_of_context_response(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    session = get_user_session(update.message.chat_id)

    # Specific keyword checks for common out-of-context queries
    lower_message = user_message.lower()
    if any(word in lower_message for word in ["joke", "make me laugh", "funny"]):
        jokes = [
            "Why did the gamer bring a ladder to the bar? He heard the drinks were on the house!",
            "Why don't skeletons fight each other in games? They don't have the guts.",
            "What do you call a lazy kangaroo in a video game? Pouch potato!",
            "Why did the scarecrow win an award in the game? Because he was outstanding in his field!",
            "Why do programmers prefer dark mode? Because light attracts bugs!"
        ]
        await update.message.reply_text(random.choice(jokes))
        return

    if "how do i cook" in lower_message or "recipe for" in lower_message:
        await update.message.reply_text("While I'm a master of game strategies, my cooking skills are still at level 1! I'd recommend a cooking website for that. How about we find a great game to play instead?")
        return

    if "capital of" in lower_message:
        await update.message.reply_text("That sounds like a trivia question! My expertise is in the world of video games. Speaking of worlds, have you explored the world of The Witcher 3?")
        return

    if "are you a real person" in lower_message or "are you an ai" in lower_message:
        await update.message.reply_text("I'm an AI assistant, ready to help you with all your gaming needs! What can I help you find today?")
        return

    # Fallback to LLM for other out-of-context messages
    tokenizer = context.bot_data.get("tokenizer")
    transformer_model = context.bot_data.get("transformer_model")

    if not tokenizer or not transformer_model:
        await update.message.reply_text("I'm focused on gaming! Is there a game you'd like to talk about?")
        return

    device = transformer_model.device if hasattr(
        transformer_model, 'device') else "cpu"
    prompt = (
        "<s>[INST] The user has asked a question that is out of the gaming context:\n\n"
        f"User Message: \"{user_message}\"\n\n"
        "Briefly and politely redirect the conversation back to gaming topics. Do not answer the question. For example, say 'My expertise is in gaming, but I can help you find a new game to play!'[/INST]"
    )
    try:
        logger.info(
            "[OutOfContext] Using LLM fallback for out-of-context request.")
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True)
        inputs = move_to_device(inputs, device)

        response = transformer_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        result_text = tokenizer.decode(
            response[0], skip_special_tokens=True).strip()
        custom_response = generate_custom_response(result_text)
        await update.message.reply_text(custom_response)
    except Exception as e:
        logger.error(f"Out-of-context handler error: {e}")
        await update.message.reply_text(
            "I'm not sure about that, but I can definitely help you with games! What are you looking for?")
