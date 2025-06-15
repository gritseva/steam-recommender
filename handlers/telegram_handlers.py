import logging
from telegram import Update, ChatAction
from telegram.ext import CallbackContext
from sessions.session_manager import get_user_session
from utils.llm_processing import extract_game_titles, infer_user_preferences_with_llm, parse_user_intent
from utils.translation import handle_translation
from utils.steam_utils import extract_steam_id, fetch_steam_profile, analyze_profile
from utils.response_generation import generate_custom_response, generate_response
from recommenders.collaborative import collaborative_filtering_with_fallback
from recommenders.content_based import filter_disliked_games
import pandas as pd

logger = logging.getLogger(__name__)


def handle_content_filter(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/filter' command to update content preferences by extracting excluded tags.
    """
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id)

    user_preferences = infer_user_preferences_with_llm(user_message)
    excluded_tags = user_preferences.get('excluded_tags', [])
    if not excluded_tags:
        update.message.reply_text(
            "No specific content preferences detected. Let me know if you'd like to adjust them!")
        return
    session.set_excluded_tags(excluded_tags)
    update.message.reply_text(
        f"Your content preferences have been updated. Excluded tags: {', '.join(excluded_tags)}.")


def handle_additional_info(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/additionalinfo' command to extract and store the user's Steam ID,
    then update their preferences based on their Steam profile.
    """
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id)

    steam_id = extract_steam_id(user_message)
    if steam_id:
        session.set_user_id(steam_id)
        profile_data = fetch_steam_profile(steam_id)
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
            update.message.reply_text(summary)
        else:
            update.message.reply_text(
                "Couldn't retrieve your Steam profile data. Make sure your profile is public.")
    else:
        update.message.reply_text("Please provide a valid Steam ID.")


def handle_out_of_context_response(update: Update, context: CallbackContext) -> None:
    """
    Handle messages that are not related to gaming by redirecting the conversation.
    """
    user_message = update.message.text
    session = get_user_session(update.message.chat_id)
    tokenizer = context.bot_data.get("tokenizer")
    transformer_model = context.bot_data.get("transformer_model")
    device = transformer_model.device if hasattr(
        transformer_model, 'device') else "cpu"

    prompt = (
        "<s>[INST] The user has asked a question that is out of gaming context:\n\n"
        f"User Message: \"{user_message}\"\n\n"
        "Redirect the conversation back to gaming topics.\n"
        "Response:[/INST]"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        response = transformer_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        result_text = tokenizer.decode(
            response[0], skip_special_tokens=True).strip()
        custom_response = generate_custom_response(result_text)
        update.message.reply_text(custom_response)
    except Exception as e:
        logger.error(f"Out-of-context handler error: {e}")
        update.message.reply_text(
            "I'm sorry, I couldn't process your request.")


def handle_unknown_intent(update: Update, context: CallbackContext) -> None:
    """
    Default handler for messages that don't match any known command.
    """
    update.message.reply_text(
        "Could you please clarify what you're looking for?")


def handle_game_comparison(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/compare' command to compare two games based on their details.
    """
    user_message = update.message.text
    session = get_user_session(update.message.chat_id)
    games_complete_df = context.bot_data.get("games_complete_df")
    if games_complete_df is None or games_complete_df.empty:
        update.message.reply_text("Game data is not available.")
        return
    game_titles = extract_game_titles(user_message)
    if len(game_titles) < 2:
        update.message.reply_text("Please mention two games to compare.")
        return
    # Assume helper functions are in utils.game_info_utils
    from utils.game_info_utils import get_game_info_by_title, compare_games
    game1_info = get_game_info_by_title(game_titles[0], games_complete_df)
    game2_info = get_game_info_by_title(game_titles[1], games_complete_df)
    if game1_info.empty or game2_info.empty:
        missing = []
        if game1_info.empty:
            missing.append(game_titles[0])
        if game2_info.empty:
            missing.append(game_titles[1])
        update.message.reply_text(
            f"Couldn't find information on {', '.join(missing)}.")
        return
    comparison = compare_games(game1_info, game2_info)
    response = generate_custom_response(comparison)
    update.message.reply_text(response, parse_mode='Markdown')


def handle_recommend_games(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/recommend' command to provide game recommendations.
    This function translates the message if necessary, extracts game titles,
    infers user preferences, filters games, and uses collaborative filtering with fallback.
    """
    logger = logging.getLogger(__name__)
    try:
        user_id = update.message.chat_id
        user_message = update.message.text
        update.message.reply_text("Processing your request. Please wait...")
        context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

        session = get_user_session(user_id)

        # Translate the message if not in English
        translated_message, detected_language = handle_translation(
            user_message, session)
        if not translated_message:
            update.message.reply_text(
                "Could not process your message. Please try again.")
            return

        game_titles = extract_game_titles(translated_message)
        if not game_titles:
            update.message.reply_text(
                "Could you specify which game you'd like recommendations similar to?")
            return

        try:
            user_preferences = infer_user_preferences_with_llm(user_message)
        except Exception as e:
            logger.error(f"Error inferring preferences: {e}")
            user_preferences = {
                "liked_games": game_titles, "excluded_tags": []}

        # Update session with liked games and excluded tags
        liked_games = user_preferences.get("liked_games", game_titles)
        session.update_likes(liked_games)
        excluded_tags = user_preferences.get("excluded_tags", [])
        session.set_excluded_tags(excluded_tags)

        # Retrieve shared resources from bot_data
        games_complete_df = context.bot_data.get("games_complete_df")
        ncf_model = context.bot_data.get("ncf_model")
        user_encoder = context.bot_data.get("user_encoder")
        game_encoder = context.bot_data.get("game_encoder")

        # Filter out disliked games
        filtered_games = filter_disliked_games(
            games_complete_df, session.disliked_games)
        # Apply content filtering based on excluded tags
        for tag in session.get_excluded_tags():
            filtered_games = filtered_games[~filtered_games['tags'].str.contains(
                tag, case=False, na=False)]
            filtered_games = filtered_games[~filtered_games['genres'].str.contains(
                tag, case=False, na=False)]
            if 'themes' in filtered_games.columns:
                filtered_games = filtered_games[~filtered_games['themes'].str.contains(
                    tag, case=False, na=False)]

        recommendations = collaborative_filtering_with_fallback(
            session.user_id or None,
            filtered_games,
            session,
            ncf_model=ncf_model,
            user_encoder=user_encoder,
            game_encoder=game_encoder,
            top_n=5
        )
        if recommendations.empty:
            update.message.reply_text(
                "Sorry, I couldn't find any recommendations based on your preferences.")
            return

        response = generate_response(user_message, recommendations, session)
        update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in recommendation handler: {e}")
        update.message.reply_text(
            "An unexpected error occurred. Please try again later.")
