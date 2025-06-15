import logging
from telegram import Update
from telegram.ext import CallbackContext
from utils.youtube_utils import search_youtube_videos, extract_video_type
from utils.llm_processing import extract_game_titles
from utils.response_generation import generate_custom_response

logger = logging.getLogger(__name__)


def handle_video_search(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/video' command by searching for a relevant game video on YouTube.
    """
    user_message = update.message.text
    game_titles = extract_game_titles(user_message)
    if game_titles:
        game_title = game_titles[0]
        video_type = extract_video_type(user_message)
        search_query = f"{game_title} {video_type}" if video_type else game_title
        video_info = search_youtube_videos(search_query, max_results=1)
        if video_info:
            video_description = video_type.capitalize() if video_type else "Video"
            response = generate_custom_response(
                f"Here's a {video_description} for '{game_title}':\n{video_info['url']}")
            update.message.reply_text(response)
        else:
            video_description = video_type or "video"
            update.message.reply_text(generate_custom_response(
                f"Sorry, I couldn't find a {video_description} for '{game_title}'."))
    else:
        update.message.reply_text(generate_custom_response(
            "Please specify the game you'd like to see a video for."))
