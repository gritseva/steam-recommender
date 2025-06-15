import logging
from telegram import Update
from telegram.ext import CallbackContext
from utils.price_tracker import track_price_changes, match_titles_to_app_ids
from utils.llm_processing import extract_game_titles
from sessions.session_manager import get_user_session

logger = logging.getLogger(__name__)


def handle_price_tracker(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/price_tracker' command by extracting game titles from the user's message,
    matching them to app IDs, and tracking their price changes.
    """
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id)

    # Retrieve the shared game dataset
    games_complete_df = context.bot_data.get("games_complete_df")
    if games_complete_df is None or games_complete_df.empty:
        update.message.reply_text(
            "The game database is currently unavailable.")
        return

    game_titles = extract_game_titles(user_message)
    if not game_titles:
        update.message.reply_text(
            "I couldn't extract any valid game titles from your message. Please try again.")
        return

    app_ids = match_titles_to_app_ids(game_titles, games_complete_df)
    if not app_ids:
        update.message.reply_text(
            "I couldn't match any game titles to the database.")
        return

    try:
        price_data_list = track_price_changes(app_ids)
        response_lines = []
        for game_title, price_data in zip(game_titles, price_data_list):
            current_price = price_data.get('current_price', 'N/A')
            original_price = price_data.get('original_price', 'N/A')
            currency = price_data.get('currency', '')
            if current_price == 'N/A':
                line = f"{game_title}: Price information is not available."
            else:
                line = f"{game_title}: Current Price - {current_price:.2f} {currency}, Original Price - {original_price:.2f} {currency}"
            response_lines.append(line)
        response_message = (
            "I'm now tracking the prices for the following games:\n\n" +
            "\n".join(response_lines) +
            "\n\nI'll notify you if there are any changes."
        )
        update.message.reply_text(response_message)
    except Exception as e:
        logger.error(f"Error tracking prices: {e}")
        update.message.reply_text(
            "An error occurred while tracking prices. Please try again later.")
