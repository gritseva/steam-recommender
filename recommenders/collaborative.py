import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
from recommenders.content_based import get_advanced_similar_games

logger = logging.getLogger(__name__)


def get_user_embedding(user_id: int, ncf_model, user_encoder) -> np.ndarray:
    """
    Retrieve the user embedding from the NCF model.

    Args:
        user_id (int): The user's identifier.
        ncf_model: Loaded Neural Collaborative Filtering model.
        user_encoder: Encoder to transform the user ID.

    Returns:
        np.ndarray: The user embedding, or None if an error occurs.
    """
    try:
        user_id_encoded = user_encoder.transform([user_id])
        user_embedding = ncf_model.get_layer(
            'user_embedding')(user_id_encoded).numpy()
        logger.info(f"Obtained embedding for user_id {user_id}.")
        return user_embedding
    except Exception as e:
        logger.error(f"Error encoding user ID {user_id}: {e}")
        return None


def collaborative_filtering(user_embedding: np.ndarray, ncf_model, game_encoder, filtered_games: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Generate game recommendations using collaborative filtering.

    Args:
        user_embedding (np.ndarray): The embedding vector for the user.
        ncf_model: The loaded NCF model.
        game_encoder: Encoder to map indices back to app IDs.
        filtered_games (pd.DataFrame): DataFrame with candidate games.
        top_n (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame containing recommended games.
    """
    try:
        item_embeddings = ncf_model.get_layer(
            'item_embedding').get_weights()[0]
        logger.debug(
            f"Item embeddings shape: {item_embeddings.shape}, User embedding shape: {user_embedding.shape}")
        similarity_scores = cosine_similarity(
            user_embedding.reshape(1, -1), item_embeddings).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        logger.debug(
            f"Top similarity scores: {similarity_scores[top_indices]}")
        recommended_app_ids = game_encoder.inverse_transform(top_indices)
        logger.debug(f"Top recommended app_ids: {recommended_app_ids}")
        recommendations = filtered_games[filtered_games['app_id'].isin(
            recommended_app_ids)]
        logger.info(
            f"Collaborative filtering produced {len(recommendations)} recommendations.")
        logger.debug(
            f"Collaborative recommendations titles: {recommendations['title'].tolist() if 'title' in recommendations else 'N/A'}")
        return recommendations
    except Exception as e:
        logger.error(f"Error during collaborative filtering: {e}")
        return pd.DataFrame()


def collaborative_filtering_with_fallback(user_id: int, filtered_games: pd.DataFrame, session, ncf_model, user_encoder, game_encoder, top_n: int = 5) -> pd.DataFrame:
    """
    Provide game recommendations using collaborative filtering with a fallback to content-based recommendations.

    Args:
        user_id (int): The user's identifier.
        filtered_games (pd.DataFrame): Candidate games DataFrame.
        session: The user's session containing preferences and liked games.
        ncf_model: The loaded NCF model.
        user_encoder: Encoder to transform user ID.
        game_encoder: Encoder to map model indices to app IDs.
        top_n (int): Number of recommendations desired.

    Returns:
        pd.DataFrame: DataFrame of top_n recommended games.
    """
    logger.info(
        f"[COLLABORATIVE] Starting collaborative filtering with fallback for user_id: {user_id}")
    logger.info(
        f"[COLLABORATIVE] Available filtered games: {len(filtered_games)}")
    logger.info(f"[COLLABORATIVE] Session liked games: {session.liked_games}")

    recommendations = pd.DataFrame()
    user_embedding = None

    # Step 1: Try collaborative filtering
    if user_id:
        logger.info(
            f"[COLLABORATIVE] User ID provided ({user_id}), attempting collaborative filtering...")
        user_embedding = get_user_embedding(user_id, ncf_model, user_encoder)
        if user_embedding is None:
            logger.warning(
                f"[COLLABORATIVE] No user embedding found for user_id {user_id}. Skipping collaborative filtering.")
        else:
            logger.info(
                f"[COLLABORATIVE] User embedding obtained successfully, shape: {user_embedding.shape}")
    else:
        logger.info(
            f"[COLLABORATIVE] No user_id provided, skipping collaborative filtering.")

    if user_embedding is not None:
        logger.info(
            f"[COLLABORATIVE] Running collaborative filtering algorithm...")
        recommendations = collaborative_filtering(
            user_embedding, ncf_model, game_encoder, filtered_games, top_n)
        logger.info(
            f"[COLLABORATIVE] Collaborative filtering returned {len(recommendations)} recommendations.")
        if not recommendations.empty:
            logger.info(
                f"[COLLABORATIVE] Collaborative recommendations: {recommendations['title'].tolist()}")
        else:
            logger.info(
                f"[COLLABORATIVE] Collaborative filtering produced empty results.")

    # Step 2: Fallback to content-based if needed
    if recommendations.empty or len(recommendations) < top_n:
        reason = 'empty' if recommendations.empty else f'insufficient results ({len(recommendations)} < {top_n})'
        logger.info(
            f"[COLLABORATIVE] Falling back to content-based recommendations. Reason: {reason}")

        liked_query = " ".join(session.liked_games) if isinstance(
            session.liked_games, (list, set)) else session.liked_games
        logger.info(f"[COLLABORATIVE] Content-based query: '{liked_query}'")
        logger.info(
            f"[COLLABORATIVE] Using genres: {session.user_preferences.get('genres', [])}")
        logger.info(
            f"[COLLABORATIVE] Using release year filter: {session.user_preferences.get('release_year_filter', None)}")

        recommendations = get_advanced_similar_games(
            user_query=liked_query,
            combined_df=filtered_games,
            vector_store=session.vector_store,
            genres=session.user_preferences.get("genres", []),
            release_year_filter=session.user_preferences.get(
                "release_year_filter"),
            k=top_n
        )
        logger.info(
            f"[COLLABORATIVE] Content-based fallback produced {len(recommendations)} recommendations.")
        if not recommendations.empty:
            logger.info(
                f"[COLLABORATIVE] Content-based recommendations: {recommendations['title'].tolist()}")
        else:
            logger.info(
                f"[COLLABORATIVE] Content-based fallback also produced empty results.")
    else:
        logger.info(
            f"[COLLABORATIVE] Collaborative filtering successful, no fallback needed.")

    final_recommendations = recommendations.head(top_n)
    logger.info(
        f"[COLLABORATIVE] Final recommendations ({len(final_recommendations)}): {final_recommendations['title'].tolist() if not final_recommendations.empty else 'None'}")

    return final_recommendations
