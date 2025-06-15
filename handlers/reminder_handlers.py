import logging
from telegram import Update
from telegram.ext import CallbackContext
from utils.date_utils import extract_date_time
from sessions.session_manager import get_user_session
# if you want to use it for responses
from utils.response_generation import generate_custom_response

logger = logging.getLogger(__name__)


def handle_game_session_reminder(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/reminder' command to set a game session reminder.
    Extracts game title(s) and reminder time from the user's message.
    """
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id)

    # Extract game titles using the LLM extraction function.
    from utils.llm_processing import extract_game_titles
    game_titles = extract_game_titles(user_message)

    reminder_time = extract_date_time(user_message)
    if not reminder_time:
        update.message.reply_text(
            "I couldn't extract a valid reminder time. Please specify a valid time.")
        return

    if not game_titles:
        update.message.reply_text(
            "Please specify the game you want to set a reminder for.")
        return

    reminder = {'game': game_titles[0], 'time': reminder_time}
    session.reminders.append(reminder)
    response = f"Reminder set: You'll be reminded to play '{game_titles[0]}' at {reminder_time}."
    update.message.reply_text(response)
    # Use the transformer components loaded in bot_data to generate a custom response.
