# handlers/comparison_handlers.py
import logging
from telegram import Update
from telegram.ext import CallbackContext
from sessions.session_manager import get_user_session
from utils.llm_processing import extract_game_titles
from utils.response_generation import generate_custom_response
from utils.game_info_utils import get_game_info_by_title, compare_games

logger = logging.getLogger(__name__)


async def handle_game_comparison(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    session = get_user_session(update.message.chat_id)
    games_complete_df = context.bot_data.get("games_complete_df")

    if games_complete_df is None or games_complete_df.empty:
        await update.message.reply_text("Game data is not available. Please try again later.")
        return

    game_titles = extract_game_titles(user_message, context)
    if len(game_titles) < 2:
        await update.message.reply_text("Please mention two games to compare.")
        return

    # Use the first two extracted titles
    title1, title2 = game_titles[0], game_titles[1]

    game1_info = get_game_info_by_title(title1, games_complete_df)
    game2_info = get_game_info_by_title(title2, games_complete_df)

    missing_games = []
    if game1_info.empty:
        missing_games.append(title1)
    if game2_info.empty:
        missing_games.append(title2)

    if missing_games:
        await update.message.reply_text(
            f"Sorry, I couldn't find information for: {', '.join(missing_games)}.")
        return

    comparison = compare_games(game1_info, game2_info)
    # The response from compare_games is already formatted, no need for generate_custom_response
    await update.message.reply_text(comparison, parse_mode='Markdown')
