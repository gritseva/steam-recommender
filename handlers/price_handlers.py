# handlers/price_handlers.py
import logging
from telegram import Update
from telegram.ext import CallbackContext
from utils.price_tracker import track_price_changes
from utils.game_utils import match_titles_to_app_ids
from utils.llm_processing import extract_game_titles
from sessions.session_manager import get_user_session

logger = logging.getLogger(__name__)


async def handle_price_tracker(update: Update, context: CallbackContext) -> None:
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
        logger.error("[PriceTracker] Game database is unavailable.")
        await update.message.reply_text(
            "The game database is currently unavailable. Please try again later.")
        return

    game_titles = extract_game_titles(user_message, context)
    logger.info(f"[PriceTracker] Extracted game titles: {game_titles}")
    if not game_titles:
        await update.message.reply_text(
            "I couldn't extract any valid game titles from your message. Please try again.")
        return

    app_ids = match_titles_to_app_ids(game_titles, games_complete_df)
    logger.info(f"[PriceTracker] Matched app IDs: {app_ids}")
    if not app_ids:
        await update.message.reply_text(
            f"I couldn't find a game called '{' '.join(game_titles)}' in the database.")
        return

    try:
        # The function now handles file paths correctly using defaults from config
        price_data_list = track_price_changes(app_ids)
        response_lines = []
        for i, app_id in enumerate(app_ids):
            game_title = game_titles[i]
            price_data = next(
                (item for item in price_data_list if item["appid"] == app_id), None)

            if price_data:
                current_price = price_data.get('current_price', 'N/A')
                original_price = price_data.get('original_price', 'N/A')

                if current_price == 'N/A' or 'unavailable' in current_price:
                    line = f"'{game_title}': Price information is not available."
                else:
                    line = f"'{game_title}': Current Price is {current_price} (Original: {original_price})"
                response_lines.append(line)

        if not response_lines:
            await update.message.reply_text("Could not fetch price information for the requested games.")
            return

        response_message = (
            "I'm now tracking the prices for the following games:\n\n" +
            "\n".join(response_lines) +
            "\n\nI'll notify you if there are any changes."
        )
        await update.message.reply_text(response_message)
    except Exception as e:
        logger.error(
            f"[PriceTracker] Error tracking prices: {e}", exc_info=True)
        await update.message.reply_text(
            "An error occurred while tracking prices. Please try again later.")
