# data/preprocess.py
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


def preprocess_games_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the games dataframe.
    - Converts the 'date_release' column to datetime.
    - Ensures 'app_id' is numeric and drops invalid rows.
    - Normalizes genre lists.
    """
    try:
        df['date_release'] = pd.to_datetime(
            df['date_release'], errors='coerce')
        df['app_id'] = pd.to_numeric(df['app_id'], errors='coerce')
        df = df.dropna(subset=['app_id']).astype({'app_id': int})

        # Normalize genres right after loading
        if 'genres' in df.columns:
            df['genres'] = df['genres'].apply(normalize_genres)

        logger.info("Preprocessed games dataframe successfully.")
    except Exception as e:
        logger.error(f"Error preprocessing dataframe: {e}")
    return df


def normalize_genres(genres):
    """
    Normalize genres from a comma-separated string, list, or Series into a list of lowercase, stripped genre strings.
    Handles various input formats gracefully.
    """
    if genres is None or (isinstance(genres, float) and pd.isna(genres)):
        return []
    if isinstance(genres, str):
        return [genre.strip().lower() for genre in genres.split(',') if genre and genre.strip()]
    if isinstance(genres, (list, pd.Series)):
        # Filter out None or NaN values before processing
        return [str(genre).strip().lower() for genre in genres if pd.notna(genre) and str(genre).strip()]
    return []


def clean_price(price_str):
    """
    Clean and convert a price string into a float.

    Args:
        price_str (str): The price string (may include currency symbols).

    Returns:
        float or None: The price as a float, or None if conversion fails.
    """
    if isinstance(price_str, str):
        cleaned = re.sub(r'[^\d.,]', '', price_str).replace(',', '.')
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Could not parse price: {price_str}")
            return None
    return price_str


def process_platforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert platform columns (e.g., 'Windows', 'Mac', 'Linux', 'Steam Deck') into boolean values.

    Args:
        df (pd.DataFrame): The games dataframe.

    Returns:
        pd.DataFrame: The dataframe with processed platform columns.
    """
    platform_columns = ['Windows', 'Mac', 'Linux', 'Steam Deck']
    for col in platform_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def clean_game_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove HTML tags from the 'about_game' and 'description' columns.

    Args:
        df (pd.DataFrame): The games dataframe.

    Returns:
        pd.DataFrame: The dataframe with cleaned text fields.
    """
    def remove_html(text):
        return re.sub(r'<.*?>', '', str(text)) if pd.notna(text) else ""

    if 'about_game' in df.columns:
        df['about_game'] = df['about_game'].apply(remove_html)
    if 'description' in df.columns:
        df['description'] = df['description'].apply(remove_html)
    logger.info("Cleaned HTML from game descriptions.")
    return df


def filter_games(games_df: pd.DataFrame, min_positive_ratio=70, min_user_reviews=100, min_rating='Positive', earliest_release_year=2000) -> pd.DataFrame:
    """
    Filter games based on a minimum positive review ratio, minimum user reviews, rating threshold,
    and earliest acceptable release year.

    Args:
        games_df (pd.DataFrame): The games dataframe.
        min_positive_ratio (int): Minimum percentage of positive reviews.
        min_user_reviews (int): Minimum number of user reviews.
        min_rating (str): The minimum rating category (e.g., 'Positive').
        earliest_release_year (int): The earliest release year to include.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    rating_order = {
        'Overwhelmingly Positive': 4,
        'Very Positive': 3,
        'Positive': 2,
        'Mostly Positive': 1,
        'Mixed': 0,
        'Mostly Negative': -1,
        'Negative': -2,
        'Very Negative': -3,
        'Overwhelmingly Negative': -4
    }
    try:
        df_copy = games_df.copy()
        df_copy['rating_value'] = df_copy['rating'].map(rating_order)
        min_rating_value = rating_order.get(min_rating, 0)

        # Ensure date_release is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date_release']):
            df_copy['date_release'] = pd.to_datetime(
                df_copy['date_release'], errors='coerce')

        filtered = df_copy[
            (df_copy['positive_ratio'] >= min_positive_ratio) &
            (df_copy['user_reviews'] >= min_user_reviews) &
            (df_copy['rating_value'] >= min_rating_value) &
            (df_copy['date_release'].dt.year >= earliest_release_year)
        ]
        logger.info(
            f"Filtered games: from {len(games_df)} rows down to {len(filtered)} rows.")
        return filtered.drop(columns=['rating_value'])
    except Exception as e:
        logger.error(f"Error filtering games: {e}")
        return games_df
