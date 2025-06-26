import logging
import pandas as pd
import numpy as np
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import CallbackContext
from sessions.session_manager import get_user_session
from utils.llm_processing import extract_game_titles, infer_user_preferences_with_llm
from utils.translation import handle_translation
from utils.response_generation import generate_response
from recommenders.collaborative import collaborative_filtering_with_fallback
from recommenders.content_based import filter_disliked_games
from utils.game_info_utils import get_game_info_by_title, filter_games_by_genre, normalize_genre


# TODO:
# - add opinion request
# - add top games request
# - add translation request
# -

async def handle_recommend_games(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/recommend' command to provide game recommendations.
    This function translates the message if necessary, extracts game titles,
    infers user preferences, filters games, and uses collaborative filtering with fallback.
    """
    logger = logging.getLogger(__name__)
    try:
        user_id = update.message.chat_id
        user_message = update.message.text
        await update.message.reply_text("Processing your request. Please wait...")
        await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

        session = get_user_session(
            user_id, context.bot_data.get("vector_store"))

        # Translate the message if not in English
        translated_message, detected_language = handle_translation(
            user_message, session)
        if not translated_message:
            await update.message.reply_text(
                "Could not process your message. Please try again.")
            return

        game_titles = extract_game_titles(translated_message, context)
        if not game_titles:
            await update.message.reply_text(
                "Could you specify which game you'd like recommendations similar to?")
            return

        try:
            user_preferences = infer_user_preferences_with_llm(
                user_message, context)
        except Exception as e:
            logger.error(f"Error inferring preferences: {e}")
            user_preferences = {
                "liked_games": game_titles, "excluded_tags": []}

        # Normalize genres in user preferences
        genres = user_preferences.get("genres", [])
        user_preferences["genres"] = [normalize_genre(g) for g in genres]
        # Update session with liked games and excluded tags
        liked_games = user_preferences.get("liked_games", game_titles)
        session.update_likes(liked_games)
        excluded_tags = user_preferences.get("excluded_tags", [])
        # Normalize excluded tags as genres too
        user_preferences["excluded_tags"] = [
            normalize_genre(t) for t in excluded_tags]
        session.set_excluded_tags(user_preferences["excluded_tags"])

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
        # Apply genre filtering if genres are specified
        if user_preferences["genres"]:
            for genre in user_preferences["genres"]:
                filtered_games = filter_games_by_genre(filtered_games, genre)

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
            await update.message.reply_text(
                "Sorry, I couldn't find any recommendations based on your preferences.")
            return

        response = generate_response(
            user_message, recommendations, session, context)
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error in recommendation handler: {e}")
        await update.message.reply_text(
            "An unexpected error occurred. Please try again later.")


async def handle_opinion_request(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/opinion' command to provide an LLM-generated opinion on a game.
    """
    logger = logging.getLogger(__name__)
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id, context.bot_data.get("vector_store"))
    games_complete_df = context.bot_data.get("games_complete_df")
    tokenizer = context.bot_data.get("tokenizer")
    transformer_model = context.bot_data.get("transformer_model")
    device = transformer_model.device if hasattr(
        transformer_model, 'device') else "cpu"
    game_titles = extract_game_titles(user_message, context)
    if not game_titles:
        await update.message.reply_text(
            "Please specify the game you'd like my opinion on.")
        return
    game_title = game_titles[0]
    game_info = get_game_info_by_title(game_title, games_complete_df)
    if isinstance(game_info, pd.DataFrame) or game_info is None or game_info.empty:
        await update.message.reply_text(
            f"Sorry, I couldn't find information on '{game_title}'.")
        return
    # Prepare prompt for LLM
    title = game_info.get('title', '')
    genres = game_info.get('genres', '')
    rating = game_info.get('rating', '')
    release_date = game_info.get(
        'release_date', game_info.get('date_release', ''))
    positive_ratio = game_info.get('positive_ratio', '')
    user_reviews = game_info.get('user_reviews', '')
    description = game_info.get('about_game', game_info.get('description', ''))
    prompt = (
        f"[INST] Provide a brief, balanced opinion on the game '{title}'.\n"
        f"Game Information:\nTitle: {title}\nGenres: {genres}\nRating: {rating}\nRelease Date: {release_date}\n"
        f"Positive Reviews: {positive_ratio}%\nTotal Reviews: {user_reviews}\nDescription: {description}\n"
        "Discuss its main features, strengths, and any notable aspects. [/INST]"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, 'to')
                      else v for k, v in inputs.items()}
        output = transformer_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        # Clean up the response
        import re
        patterns = [r'\[INST\].*?\[/INST\]',
                    r'<s>|</s>', r'^\s*[-•]\s*', r'\n\s*\n']
        cleaned_response = result
        for pattern in patterns:
            cleaned_response = re.sub(
                pattern, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE).strip()
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
        await update.message.reply_text(
            cleaned_response or "I couldn't generate an opinion for this game.")
    except Exception as e:
        logger.error(f"Error generating opinion: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't generate an opinion at this time.")


async def handle_top_games_request(update: Update, context: CallbackContext) -> None:
    """
    Handle a request for top games of a specified genre.
    """
    user_message = update.message.text
    user_id = update.message.chat_id
    session = get_user_session(user_id, context.bot_data.get("vector_store"))
    games_complete_df = context.bot_data.get("games_complete_df")
    tokenizer = context.bot_data.get("tokenizer")
    transformer_model = context.bot_data.get("transformer_model")
    # Use LLM to infer genres from the user's message
    user_preferences = infer_user_preferences_with_llm(user_message, context)
    genres = user_preferences.get('genres', [])
    if not genres:
        await update.message.reply_text(
            "Please specify the genre you're interested in.")
        return
    genre = genres[0].strip().lower()
    # Use genre normalization and filtering
    filtered_games = filter_games_by_genre(games_complete_df, genre).copy()
    if filtered_games.empty:
        await update.message.reply_text(
            f"Sorry, I couldn't find any top games in the genre '{genre}'.")
        return
    # Score and sort games
    rating_order = {
        'Overwhelmingly Positive': 5,
        'Very Positive': 4,
        'Positive': 3,
        'Mostly Positive': 2,
        'Mixed': 1,
        'Mostly Negative': -1,
        'Negative': -2,
        'Very Negative': -3,
        'Overwhelmingly Negative': -4
    }
    filtered_games['rating_value'] = filtered_games['rating'].map(
        rating_order).fillna(0)
    filtered_games['positive_ratio'] = filtered_games['positive_ratio'].fillna(
        0)
    filtered_games['user_reviews'] = filtered_games['user_reviews'].fillna(0)
    filtered_games['score'] = (
        filtered_games['rating_value'] * 2 +
        filtered_games['positive_ratio'] * 0.1 +
        filtered_games['user_reviews'].apply(
            lambda x: 0 if pd.isna(x) else np.log1p(x))
    )
    top_games = filtered_games.sort_values(
        by='score', ascending=False).head(10)
    # Format the response
    top_games_list = "\n\n".join([
        f"{i}. *{row.title}*\nRating: {row.rating}, Positive Reviews: {row.positive_ratio:.0f}%"
        for i, row in enumerate(top_games.itertuples(), 1)
    ])
    response = f"Top 10 {genre.capitalize()} games:\n\n{top_games_list}"
    await update.message.reply_text(response, parse_mode='Markdown')
