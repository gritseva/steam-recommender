# handlers/greeting_handlers.py
import logging
from telegram import Update
from telegram.ext import CallbackContext
from utils.response_generation import generate_custom_response
from sessions.session_manager import get_user_session


def get_time_greeting():
    """Get appropriate time-based greeting."""
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    else:
        return "evening"


async def handle_greeting(update: Update, context: CallbackContext) -> None:
    """
    Handle basic greetings with a warm, gaming-focused response. Compatible with both command and message handler use.
    """
    user_message = update.message.text.lower().strip()
    session = get_user_session(update.message.chat_id)
    user_name = getattr(update.message.from_user,
                        'first_name', None) or "there"

    # Different greeting responses based on the type of greeting
    if any(word in user_message for word in ["hi", "hello", "hey", "sup", "yo"]):
        response = (
            f"Hey {user_name}! 👋 Welcome to your personal gaming assistant! "
            "I'm here to help you discover amazing games, track prices, and get personalized recommendations. "
            "What kind of games are you into, or would you like me to recommend something?"
        )
    elif any(word in user_message for word in ["good morning", "good afternoon", "good evening"]):
        response = (
            f"Good {get_time_greeting()} {user_name}! 🎮 "
            "Perfect time to explore some great games! "
            "What's on your gaming radar today?"
        )
    else:
        # Generic friendly response
        response = (
            f"Hello {user_name}! 🎮 "
            "I'm your gaming buddy - ready to help you find your next favorite game! "
            "What can I help you with today?"
        )

    await update.message.reply_text(response)
