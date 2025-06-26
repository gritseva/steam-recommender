import pandas as pd
import logging
from fuzzywuzzy import process, fuzz


def match_titles_to_app_ids(titles: list, games_complete_df: pd.DataFrame) -> list:
    """
    Match game titles to their app IDs using fuzzy matching.

    Args:
        titles (list): List of game titles.
        games_complete_df (pd.DataFrame): DataFrame containing game information.

    Returns:
        list: List of matched app IDs.
    """
    logger = logging.getLogger(__name__)
    app_ids = []
    games_complete_df['title'] = games_complete_df['title'].astype(str)
    all_titles = games_complete_df['title'].tolist()
    for title in titles:
        match = process.extractOne(
            title, all_titles, scorer=fuzz.token_sort_ratio)
        logger.info(f"[AppIDMatch] Title: '{title}' | Fuzzy match: {match}")
        if match and match[1] > 80:
            app_id = games_complete_df.iloc[all_titles.index(
                match[0])]['app_id']
            logger.info(f"[AppIDMatch] Matched app ID for '{title}': {app_id}")
            app_ids.append(app_id)
        else:
            logger.info(
                f"[AppIDMatch] No good fuzzy match found for '{title}'")
    return app_ids
