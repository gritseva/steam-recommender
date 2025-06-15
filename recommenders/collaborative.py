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
        # Retrieve item embeddings from the model
        item_embeddings = ncf_model.get_layer(
            'item_embedding').get_weights()[0]
        similarity_scores = cosine_similarity(
            user_embedding.reshape(1, -1), item_embeddings).flatten()
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommended_app_ids = game_encoder.inverse_transform(top_indices)
        recommendations = filtered_games[filtered_games['app_id'].isin(
            recommended_app_ids)]
        logger.info("Collaborative filtering produced recommendations.")
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
    recommendations = pd.DataFrame()
    user_embedding = None

    # Attempt to obtain user embedding if user_id is available
    if user_id:
        user_embedding = get_user_embedding(user_id, ncf_model, user_encoder)

    if user_embedding is not None:
        recommendations = collaborative_filtering(
            user_embedding, ncf_model, game_encoder, filtered_games, top_n)

    # Fallback: If collaborative filtering fails or returns insufficient recommendations,
    # use content-based recommendation based on the user's liked games.
    if recommendations.empty or len(recommendations) < top_n:
        liked_query = " ".join(session.liked_games) if isinstance(
            session.liked_games, (list, set)) else session.liked_games
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
            "Fell back to content-based recommendations due to insufficient collaborative filtering results.")

    return recommendations.head(top_n)
