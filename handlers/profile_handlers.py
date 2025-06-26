import logging
from telegram import Update
from telegram.ext import CallbackContext
from sessions.session_manager import get_user_session
from utils.llm_processing import infer_user_preferences_with_llm
from utils.translation import handle_translation
from utils.steam_utils import extract_steam_id, fetch_steam_profile, analyze_profile
from utils.game_info_utils import normalize_genre
import re

# TODO: add a function to handle the greeting and goodbye (unique ones)
# TODO: add a function to


async def handle_content_filter(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id)
    user_preferences = infer_user_preferences_with_llm(user_message, context)
    excluded_tags = user_preferences.get('excluded_tags', [])
    # Normalize excluded tags as genres
    excluded_tags = [normalize_genre(tag) for tag in excluded_tags]
    if not excluded_tags:
        await update.message.reply_text(
            "No specific content preferences detected. Let me know if you'd like to adjust them!")
        return
    session.set_excluded_tags(excluded_tags)
    await update.message.reply_text(
        f"Your content preferences have been updated. Excluded tags: {', '.join(excluded_tags)}.")


async def handle_additional_info(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id)
    logger = logging.getLogger(__name__)
    # Greeting detection
    greeting_pattern = re.compile(
        r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b", re.IGNORECASE)
    if greeting_pattern.search(user_message):
        await update.message.reply_text(
            "Hello! Welcome to SteamRecs. How can I help you today?")
        return
    steam_id = extract_steam_id(user_message)
    logger.info(f"[SteamProfile] Extracted Steam ID: {steam_id}")
    if steam_id:
        session.set_user_id(steam_id)
        try:
            profile_data = fetch_steam_profile(steam_id)
            logger.info(f"[SteamProfile] API response: {profile_data}")
        except Exception as e:
            logger.error(f"[SteamProfile] Error fetching profile: {e}")
            profile_data = {"error": str(e)}
        if profile_data and 'error' in profile_data:
            await update.message.reply_text(
                f"Couldn't retrieve your Steam profile data: {profile_data['error']}")
            return
        if profile_data:
            games_complete_df = context.bot_data.get("games_complete_df")
            user_preferences = analyze_profile(profile_data, games_complete_df)
            session.update_preferences(user_preferences)
            total_playtime = user_preferences.get('total_playtime', 0)
            most_played_games = user_preferences.get('most_played_games', [])
            favorite_genres = user_preferences.get('favorite_genres', [])
            summary = (
                f"Your Steam profile has been processed:\n"
                f"Total playtime: {total_playtime} hours\n"
                f"Most played games: {', '.join(most_played_games)}\n"
                f"Favorite genres: {', '.join(favorite_genres)}"
            )
            await update.message.reply_text(summary)
        else:
            await update.message.reply_text(
                "Couldn't retrieve your Steam profile data. Make sure your profile is public.")
    else:
        logger.info(
            f"[SteamProfile] No valid Steam ID found in message: {user_message}")
        await update.message.reply_text("Please provide a valid Steam ID.")


async def handle_unknown_intent(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Could you please clarify what you're looking for?")
