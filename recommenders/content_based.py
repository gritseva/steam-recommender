# content based module

import logging
from rapidfuzz import fuzz
import pandas as pd
from typing import Union, List, Optional


# search in the combined_df for games that match the user preferences -> fallback collaborative filteting:
# get_advanced_similar_games - vector-based similarity from the full games catalog

logger = logging.getLogger(__name__)


def apply_genre_filter(df: pd.DataFrame, include_genres: List[str], exclude_genres: List[str]) -> pd.DataFrame:
    """
    Filter the DataFrame based on inclusion and exclusion of genres in the 'tags' column.

    Args:
        df (pd.DataFrame): The candidate games DataFrame.
        include_genres (List[str]): Genres to include.
        exclude_genres (List[str]): Genres to exclude.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if include_genres and 'tags' in df.columns:
        df = df[df['tags'].apply(lambda tags: any(tag.lower() in (
            t.lower() for t in tags) for tag in include_genres))]
    if exclude_genres and 'tags' in df.columns:
        df = df[~df['tags'].apply(lambda tags: any(tag.lower() in (
            t.lower() for t in tags) for tag in exclude_genres))]
    return df


def apply_price_filter(df: pd.DataFrame, min_price: float, max_price: float) -> pd.DataFrame:
    """
    Filter games by a price range.

    Args:
        df (pd.DataFrame): DataFrame with a 'price' column.
        min_price (float): Minimum acceptable price.
        max_price (float): Maximum acceptable price.

    Returns:
        pd.DataFrame: Price-filtered DataFrame.
    """
    if 'price' in df.columns:
        return df.loc[(df['price'] >= (min_price or 0)) & (df['price'] <= (max_price or float('inf')))]
    return df


def apply_platform_filter(df: pd.DataFrame, platforms: List[str]) -> pd.DataFrame:
    """
    Filter games based on supported platforms.

    Args:
        df (pd.DataFrame): DataFrame with platform columns.
        platforms (List[str]): List of platform identifiers (e.g., 'win', 'mac').

    Returns:
        pd.DataFrame: Platform-filtered DataFrame.
    """
    platform_columns = {'win': 'Windows', 'mac': 'Mac',
                        'linux': 'Linux', 'steam_deck': 'Steam Deck'}
    selected = [platform_columns[p]
                for p in platforms if p in platform_columns]
    if selected:
        return df[df[selected].any(axis=1)]
    return df


def content_based_filtering(user_preferences: dict, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multiple filters to the games DataFrame based on user preferences.

    Args:
        user_preferences (dict): Dictionary with keys such as include_genres, exclude_genres, min_price, max_price, platforms, etc.
        games_df (pd.DataFrame): The full games DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = games_df.copy()
    if 'date_release' in df.columns:
        df['date_release'] = pd.to_datetime(
            df['date_release'], errors='coerce')

    df = apply_genre_filter(df, user_preferences.get(
        "include_genres", []), user_preferences.get("exclude_genres", []))
    df = apply_price_filter(df, user_preferences.get(
        "min_price"), user_preferences.get("max_price"))
    df = apply_platform_filter(df, user_preferences.get("platforms", []))

    # Developer type filtering (if applicable)
    if "developer_type" in user_preferences and "developer" in df.columns:
        # Here you may integrate a function to classify developer type (e.g., AAA vs Indie)
        df['studio_type'] = df['developer'].apply(
            lambda x: "AAA" if x in user_preferences.get("aaa_studios", []) else "Indie")
        df = df[df['studio_type'] == user_preferences["developer_type"]]

    # Rating filtering
    if 'rating' in df.columns:
        if user_preferences.get("min_rating"):
            df = df[df['rating'] >= user_preferences["min_rating"]]
        if user_preferences.get("max_rating"):
            df = df[df['rating'] <= user_preferences["max_rating"]]

    # User reviews filtering
    if user_preferences.get("min_user_reviews") and 'user_reviews' in df.columns:
        df = df[df['user_reviews'] >= user_preferences["min_user_reviews"]]

    # Release year filtering
    if 'date_release' in df.columns:
        if user_preferences.get("start_year"):
            df = df[df['date_release'].dt.year >=
                    user_preferences["start_year"]]
        if user_preferences.get("end_year"):
            df = df[df['date_release'].dt.year <= user_preferences["end_year"]]

    # Playtime filtering
    if user_preferences.get("min_playtime") and 'average_playtime_forever' in df.columns:
        df = df[df['average_playtime_forever']
                >= user_preferences["min_playtime"]]

    return df.drop(columns=['studio_type'], errors='ignore')


def get_advanced_similar_games(user_query: Union[str, List[str]], combined_df: pd.DataFrame, vector_store, genres: Optional[List[str]] = None, release_year_filter: Optional[dict] = None, k: int = 5, similarity_threshold: int = 95, retrieval_multiplier: int = 5) -> pd.DataFrame:
    """
    Retrieve the top-k similar games to the user query using vector-based similarity search,
    with additional genre and release year filtering and fuzzy matching to remove duplicates.
    """
    logger.info(f"[CONTENT-BASED] Starting advanced similar games search")
    logger.info(f"[CONTENT-BASED] User query: {user_query}")
    logger.info(
        f"[CONTENT-BASED] Target k: {k}, retrieval_multiplier: {retrieval_multiplier}")
    logger.info(
        f"[CONTENT-BASED] Available games in combined_df: {len(combined_df)}")

    query_text = " ".join(user_query) if isinstance(
        user_query, list) else user_query
    logger.info(f"[CONTENT-BASED] Processed query text: '{query_text}'")

    # Create a set of lowercased input titles to filter out later
    input_titles_lower = set()
    if isinstance(user_query, list):
        input_titles_lower = {title.lower() for title in user_query}
        logger.info(
            f"[CONTENT-BASED] Input titles to filter out: {input_titles_lower}")

    try:
        # Step 1: Vector similarity search
        logger.info(
            f"[CONTENT-BASED] Performing vector similarity search with k={k * retrieval_multiplier}")
        results = vector_store.similarity_search(
            query=query_text, k=k * retrieval_multiplier)
        logger.info(
            f"[CONTENT-BASED] Vector store returned {len(results) if results else 0} results")

        if not results:
            logger.warning(
                "[CONTENT-BASED] No results from vector similarity search.")
            return pd.DataFrame()

        # Step 2: Extract app_ids from search results
        app_ids = [doc.metadata['app_id'] for doc in results]
        logger.info(
            f"[CONTENT-BASED] Extracted app_ids from vector results: {app_ids}")

        recommendations = combined_df[combined_df['app_id'].isin(
            app_ids)].copy()
        logger.info(
            f"[CONTENT-BASED] Initial recommendations after vector search: {len(recommendations)} games")
        if not recommendations.empty:
            logger.info(
                f"[CONTENT-BASED] Initial game titles: {recommendations['title'].tolist()}")

        # Step 3: Exclude irrelevant content (like DLCs or Mods)
        exclude_keywords = ['DLC', 'Bonus Content', 'Expansion', 'Mod']
        logger.info(
            f"[CONTENT-BASED] Filtering out keywords: {exclude_keywords}")
        recommendations = recommendations[~recommendations['title'].str.contains(
            '|'.join(exclude_keywords), case=False, na=False)]
        logger.info(
            f"[CONTENT-BASED] After DLC/Bonus/Expansion/Mod filtering: {len(recommendations)} games")
        if not recommendations.empty:
            logger.info(
                f"[CONTENT-BASED] Games after DLC filtering: {recommendations['title'].tolist()}")

        # Step 4: Filter by genres if provided
        if genres:
            logger.info(
                f"[CONTENT-BASED] Applying genre filtering for: {genres}")
            recommendations = recommendations[recommendations['tags'].apply(
                lambda tags: any(genre.lower() in (t.lower()
                                 for t in tags) for genre in genres)
            )]
            logger.info(
                f"[CONTENT-BASED] After genre filtering: {len(recommendations)} games")
            if not recommendations.empty:
                logger.info(
                    f"[CONTENT-BASED] Games after genre filtering: {recommendations['title'].tolist()}")

        # Step 5: Filter by release year if provided
        if release_year_filter:
            logger.info(
                f"[CONTENT-BASED] Applying release year filter: {release_year_filter}")
            recommendations['date_release'] = pd.to_datetime(
                recommendations['date_release'], errors='coerce')
            comparator, year = release_year_filter.get(
                "comparator"), release_year_filter.get("year")
            if comparator == "after":
                recommendations = recommendations[recommendations['date_release'].dt.year > year]
            elif comparator == "before":
                recommendations = recommendations[recommendations['date_release'].dt.year < year]
            elif comparator == "exact":
                recommendations = recommendations[recommendations['date_release'].dt.year == year]
            logger.info(
                f"[CONTENT-BASED] After release year filtering: {len(recommendations)} games")
            if not recommendations.empty:
                logger.info(
                    f"[CONTENT-BASED] Games after year filtering: {recommendations['title'].tolist()}")

        # Step 6: Filter out input games from recommendations
        if input_titles_lower:
            logger.info(
                f"[CONTENT-BASED] Filtering out input games: {input_titles_lower}")
            recommendations = recommendations[~recommendations['title'].str.lower().isin(
                input_titles_lower)]
            logger.info(
                f"[CONTENT-BASED] After filtering out input games: {len(recommendations)} games")
            if not recommendations.empty:
                logger.info(
                    f"[CONTENT-BASED] Games after input filtering: {recommendations['title'].tolist()}")

        # Step 7: Apply fuzzy matching to remove duplicate or very similar titles
        logger.info(
            f"[CONTENT-BASED] Applying fuzzy duplicate removal with threshold: {similarity_threshold}")
        unique_recs = []
        seen_titles = set()
        for _, row in recommendations.iterrows():
            title = row['title']
            if all(fuzz.partial_ratio(title, seen) < similarity_threshold for seen in seen_titles):
                unique_recs.append(row)
                seen_titles.add(title)
            if len(unique_recs) >= k:
                break

        logger.info(
            f"[CONTENT-BASED] After fuzzy deduplication: {len(unique_recs)} unique games")
        if unique_recs:
            logger.info(
                f"[CONTENT-BASED] Final unique recommendations: {[row['title'] for row in unique_recs]}")

        final_df = pd.DataFrame(unique_recs).head(k)
        logger.info(
            f"[CONTENT-BASED] Final result: {len(final_df)} recommendations")
        if not final_df.empty:
            logger.info(
                f"[CONTENT-BASED] Final recommendations: {final_df['title'].tolist()}")

        return final_df

    except Exception as e:
        logger.error(
            f"[CONTENT-BASED] Error in advanced similar games search: {e}")
        return pd.DataFrame()


def filter_disliked_games(games_df: pd.DataFrame, disliked_games: list) -> pd.DataFrame:
    """
    Remove games from the DataFrame whose titles or app_ids are in the disliked_games list.

    Args:
        games_df (pd.DataFrame): The complete games DataFrame.
        disliked_games (list): List of disliked game titles or app_ids.

    Returns:
        pd.DataFrame: DataFrame with disliked games removed.
    """
    if not disliked_games or games_df is None or games_df.empty:
        return games_df

    # Try to match by title (case-insensitive)
    filtered_df = games_df[~games_df['title'].str.lower().isin(
        [str(g).lower() for g in disliked_games])]

    # If app_id column exists and disliked_games contains numeric ids, also filter by app_id
    if 'app_id' in games_df.columns:
        disliked_ids = [g for g in disliked_games if str(g).isdigit()]
        if disliked_ids:
            filtered_df = filtered_df[~filtered_df['app_id'].astype(
                str).isin([str(g) for g in disliked_ids])]

    return filtered_df
