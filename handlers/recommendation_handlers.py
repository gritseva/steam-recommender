import logging
import pandas as pd
import numpy as np
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import CallbackContext
from sessions.session_manager import get_user_session
from utils.llm_processing import extract_game_titles, infer_user_preferences_with_llm
from utils.translation import handle_translation
from utils.response_generation import generate_response, build_recommendation_keyboard
from recommenders.collaborative import collaborative_filtering_with_fallback
from recommenders.content_based import filter_disliked_games, get_advanced_similar_games
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
        logger.info(
            f"[RECOMMENDATION] Starting recommendation request for user {user_id}")
        logger.info(f"[RECOMMENDATION] User message: '{user_message}'")

        await update.message.reply_text("Processing your request. Please wait...")
        await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

        session = get_user_session(
            user_id, context.bot_data.get("vector_store"))
        logger.info(f"[RECOMMENDATION] Session retrieved for user {user_id}")

        # Translate the message if not in English
        logger.info(f"[RECOMMENDATION] Attempting translation...")
        translated_message, detected_language = handle_translation(
            user_message, session)
        if not translated_message:
            logger.error(
                f"[RECOMMENDATION] Translation failed for message: '{user_message}'")
            await update.message.reply_text(
                "Could not process your message. Please try again.")
            return
        logger.info(
            f"[RECOMMENDATION] Translation successful: '{translated_message}' (detected: {detected_language})")

        # Extract game titles
        logger.info(
            f"[RECOMMENDATION] Extracting game titles from translated message...")
        game_titles = extract_game_titles(translated_message, context)
        logger.info(f"[RECOMMENDATION] Extracted game titles: {game_titles}")
        if not game_titles:
            logger.warning(
                f"[RECOMMENDATION] No game titles extracted from message")
            await update.message.reply_text(
                "Could you specify which game you'd like recommendations similar to?")
            return

        # Infer user preferences
        logger.info(f"[RECOMMENDATION] Inferring user preferences...")
        try:
            user_preferences = infer_user_preferences_with_llm(
                user_message, context)
            logger.info(
                f"[RECOMMENDATION] Inferred preferences: {user_preferences}")
        except Exception as e:
            logger.error(f"[RECOMMENDATION] Error inferring preferences: {e}")
            user_preferences = {
                "liked_games": game_titles, "excluded_tags": []}
            logger.info(
                f"[RECOMMENDATION] Using fallback preferences: {user_preferences}")

        # Normalize genres in user preferences
        genres = user_preferences.get("genres", [])
        user_preferences["genres"] = [normalize_genre(g) for g in genres]
        logger.info(
            f"[RECOMMENDATION] Normalized genres: {user_preferences['genres']}")

        # Update session with liked games and excluded tags
        liked_games = user_preferences.get("liked_games", game_titles)
        logger.info(f"[RECOMMENDATION] Raw liked games: {liked_games}")

        # Sanity-check liked_games: verify each exists in games_complete_df
        logger.info(
            f"[RECOMMENDATION] Verifying game titles exist in database...")
        verified_liked_games = []
        unverified_titles = []
        for title in liked_games:
            game_info = get_game_info_by_title(
                title, context.bot_data.get("games_complete_df"))
            if not game_info.empty:
                verified_liked_games.append(game_info['title'])
                logger.info(
                    f"[RECOMMENDATION] Verified game: '{title}' -> '{game_info['title']}'")
            else:
                unverified_titles.append(title)
                logger.warning(
                    f"[RECOMMENDATION] Could not verify game: '{title}'")

        if not verified_liked_games:
            logger.error(
                f"[RECOMMENDATION] No valid games found: {unverified_titles}")
            await update.message.reply_text(f"Sorry, I couldn't find these games in my database: {', '.join(unverified_titles)}. Could you check the spelling?")
            return

        logger.info(
            f"[RECOMMENDATION] Verified liked games: {verified_liked_games}")
        session.update_likes(verified_liked_games)
        liked_games = verified_liked_games

        excluded_tags = user_preferences.get("excluded_tags", [])
        # Normalize excluded tags as genres too
        user_preferences["excluded_tags"] = [
            normalize_genre(t) for t in excluded_tags]
        session.set_excluded_tags(user_preferences["excluded_tags"])
        logger.info(
            f"[RECOMMENDATION] Excluded tags: {user_preferences['excluded_tags']}")

        # Retrieve shared resources from bot_data
        games_complete_df = context.bot_data.get("games_complete_df")
        ncf_model = context.bot_data.get("ncf_model")
        user_encoder = context.bot_data.get("user_encoder")
        game_encoder = context.bot_data.get("game_encoder")
        logger.info(
            f"[RECOMMENDATION] Retrieved shared resources - games: {len(games_complete_df)}, models: {ncf_model is not None}")

        # Filter out disliked games
        logger.info(
            f"[RECOMMENDATION] Filtering out disliked games: {session.disliked_games}")
        filtered_games = filter_disliked_games(
            games_complete_df, session.disliked_games)
        logger.info(
            f"[RECOMMENDATION] After disliked games filter: {len(filtered_games)} games")

        # Apply content filtering based on excluded tags
        for tag in session.get_excluded_tags():
            logger.info(f"[RECOMMENDATION] Filtering out tag: '{tag}'")
            filtered_games = filtered_games[~filtered_games['tags'].str.contains(
                tag, case=False, na=False)]
            filtered_games = filtered_games[~filtered_games['genres'].str.contains(
                tag, case=False, na=False)]
            if 'themes' in filtered_games.columns:
                filtered_games = filtered_games[~filtered_games['themes'].str.contains(
                    tag, case=False, na=False)]
        logger.info(
            f"[RECOMMENDATION] After excluded tags filter: {len(filtered_games)} games")

        # Apply genre filtering if genres are specified
        if user_preferences["genres"]:
            logger.info(
                f"[RECOMMENDATION] Applying genre filtering for: {user_preferences['genres']}")
            for genre in user_preferences["genres"]:
                filtered_games = filter_games_by_genre(filtered_games, genre)
            logger.info(
                f"[RECOMMENDATION] After genre filtering: {len(filtered_games)} games")

        # Multi-tiered fallback system
        logger.info(
            f"[RECOMMENDATION] Starting multi-tiered recommendation system...")
        recommendations = collaborative_filtering_with_fallback(
            session.user_id or None,
            filtered_games,
            session,
            ncf_model=ncf_model,
            user_encoder=user_encoder,
            game_encoder=game_encoder,
            top_n=5
        )
        logger.info(
            f"[RECOMMENDATION] Tier 1 (Collaborative + Content-based) results: {len(recommendations)} recommendations")

        # Tier 2: Try individual game vector search if empty
        if recommendations.empty and liked_games:
            logger.info(
                f"[RECOMMENDATION] Tier 1 failed. Starting Tier 2: Individual game vector search...")
            all_recs = []
            for title in liked_games:
                logger.info(
                    f"[RECOMMENDATION] Searching for games similar to: '{title}'")
                recs_df = get_advanced_similar_games(
                    user_query=title,
                    combined_df=filtered_games,
                    vector_store=session.vector_store,
                    genres=user_preferences.get("genres", []),
                    release_year_filter=user_preferences.get(
                        "release_year_filter"),
                    k=3
                )
                if not recs_df.empty:
                    logger.info(
                        f"[RECOMMENDATION] Found {len(recs_df)} games similar to '{title}': {recs_df['title'].tolist()}")
                    all_recs.append(recs_df)
                else:
                    logger.warning(
                        f"[RECOMMENDATION] No games found similar to '{title}'")

            if all_recs:
                recommendations = pd.concat(all_recs).drop_duplicates(
                    subset=['app_id']).head(5)
                logger.info(
                    f"[RECOMMENDATION] Tier 2 results: {len(recommendations)} unique recommendations")
                if not recommendations.empty:
                    logger.info(
                        f"[RECOMMENDATION] Tier 2 recommendations: {recommendations['title'].tolist()}")
            else:
                logger.warning(
                    f"[RECOMMENDATION] Tier 2 failed - no individual game results")

        # Tier 3: Genre-based fallback
        if recommendations.empty and liked_games:
            logger.info(
                f"[RECOMMENDATION] Tier 2 failed. Starting Tier 3: Genre-based fallback...")
            input_game_genres = set()
            input_app_ids = set()
            for title in liked_games:
                logger.info(
                    f"[RECOMMENDATION] Extracting genres for: '{title}'")
                game_info = get_game_info_by_title(title, games_complete_df)
                if not game_info.empty:
                    input_app_ids.add(game_info['app_id'])
                    if isinstance(game_info['genres'], list):
                        for g in game_info['genres']:
                            input_game_genres.add(normalize_genre(g))
                            logger.info(
                                f"[RECOMMENDATION] Added genre: '{g}' -> '{normalize_genre(g)}'")

            logger.info(
                f"[RECOMMENDATION] Extracted genres: {input_game_genres}")
            logger.info(
                f"[RECOMMENDATION] Input app_ids to exclude: {input_app_ids}")

            if input_game_genres:
                genre_recs = games_complete_df[
                    games_complete_df['genres'].apply(lambda g_list: any(g in input_game_genres for g in g_list)) &
                    ~games_complete_df['app_id'].isin(input_app_ids)
                ]
                logger.info(
                    f"[RECOMMENDATION] Found {len(genre_recs)} games in matching genres")
                if 'positive_ratio' in genre_recs.columns:
                    recommendations = genre_recs.sort_values(
                        by='positive_ratio', ascending=False).head(5)
                    logger.info(
                        f"[RECOMMENDATION] Tier 3 results (sorted by rating): {len(recommendations)} recommendations")
                else:
                    recommendations = genre_recs.head(5)
                    logger.info(
                        f"[RECOMMENDATION] Tier 3 results (no rating sort): {len(recommendations)} recommendations")

                if not recommendations.empty:
                    logger.info(
                        f"[RECOMMENDATION] Tier 3 recommendations: {recommendations['title'].tolist()}")
            else:
                logger.warning(
                    f"[RECOMMENDATION] Tier 3 failed - no genres extracted")

        # Final fallback: friendly message
        if recommendations.empty:
            logger.warning(
                f"[RECOMMENDATION] All tiers failed. Sending fallback message.")
            response = (
                "I've noted that you like "
                f"**{', '.join(liked_games)}**, which are great choices! "
                "However, I'm having trouble finding similar games right now. "
                "Could you tell me what aspects you enjoy most about them? (e.g., the atmosphere, puzzles, story)"
            )
            await update.message.reply_text(response, parse_mode='Markdown')
            return

        logger.info(
            f"[RECOMMENDATION] Final recommendations ({len(recommendations)}): {recommendations['title'].tolist()}")
        response = generate_response(
            user_message, recommendations, session, context)
        keyboard = build_recommendation_keyboard(recommendations)
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=keyboard)
        logger.info(
            f"[RECOMMENDATION] Successfully sent recommendations to user")

    except Exception as e:
        logger.error(f"[RECOMMENDATION] Error in recommendation handler: {e}")
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
