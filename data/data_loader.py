import pandas as pd
import json
import logging
import os
from config.config import GAME_CSV_PATH, COMBINED_DF_PATH

logger = logging.getLogger(__name__)


def load_games_csv() -> pd.DataFrame:
    """
    Load the main games dataset from a CSV file.

    Returns:
        pd.DataFrame: The games dataframe, or an empty DataFrame if loading fails.
    """
    try:
        df = pd.read_csv(GAME_CSV_PATH)
        logger.info(
            f"Loaded games CSV from {GAME_CSV_PATH} with {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading games CSV from {GAME_CSV_PATH}: {e}")
        return pd.DataFrame()


def load_combined_df() -> pd.DataFrame:
    """
    Load the combined dataframe from a Pickle file.

    Returns:
        pd.DataFrame: The combined dataframe, or an empty DataFrame if loading fails.
    """
    try:
        df = pd.read_pickle(COMBINED_DF_PATH)
        logger.info(
            f"Loaded combined dataframe from {COMBINED_DF_PATH} with {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(
            f"Error loading combined dataframe from {COMBINED_DF_PATH}: {e}")
        return pd.DataFrame()


def load_json_data(json_path: str) -> list:
    """
    Load game metadata from a JSON file where each line is a JSON object.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        list: A list of JSON objects (dictionaries), or an empty list on failure.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(data)} JSON entries from {json_path}.")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data from {json_path}: {e}")
        return []


def merge_game_data(games_df: pd.DataFrame, new_games_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge additional game data into the main dataset based on the 'app_id' key.

    Args:
        games_df (pd.DataFrame): The primary games dataframe.
        new_games_data (pd.DataFrame): Additional data to merge.

    Returns:
        pd.DataFrame: The merged dataframe with redundant columns removed.
    """
    try:
        merged_df = games_df.merge(
            new_games_data, on='app_id', how='left', suffixes=('', '_new'))
        redundant_columns = [
            col for col in merged_df.columns if col.endswith('_new')]
        merged_df = merged_df.drop(columns=redundant_columns, errors='ignore')
        logger.info("Merged additional game data successfully.")
        return merged_df
    except Exception as e:
        logger.error(f"Error merging game data: {e}")
        return games_df.copy()


def prepare_final_dataset(games_csv_path: str, metadata_json_path: str, new_games_csv_path: str) -> pd.DataFrame:
    """
    Load, merge, and preprocess multiple datasets to create the final structured DataFrame.

    Args:
        games_csv_path (str): Path to the main games CSV.
        metadata_json_path (str): Path to the games metadata JSON file.
        new_games_csv_path (str): Path to the additional games CSV.

    Returns:
        pd.DataFrame: The final merged and structured dataframe, or an empty DataFrame if any step fails.
    """
    try:
        # Load primary games data
        games_df = pd.read_csv(games_csv_path)
        logger.info(
            f"Loaded primary games CSV from {games_csv_path} with {len(games_df)} rows.")

        # Load additional game data and rename columns for consistency
        new_games_data = pd.read_csv(new_games_csv_path)
        new_games_data = new_games_data.rename(columns={
            'AppID': 'app_id',
            'Name': 'title',
            'Release date': 'date_release',
            'Estimated owners': 'estimated_owners',
            'Peak CCU': 'peak_ccu',
            'Required age': 'required_age',
            'Price': 'price',
            'DiscountDLC count': 'discount_dlc_count',
            'About the game': 'about_game',
            'Supported languages': 'supported_languages',
            'Metacritic score': 'metacritic_score',
            'User score': 'user_score',
            'Developers': 'developer',
            'Publishers': 'publisher',
            'Categories': 'categories',
            'Genres': 'genres',
            'Tags': 'tags'
        })
        # Ensure app_id is numeric and clean
        new_games_data['app_id'] = pd.to_numeric(
            new_games_data['app_id'], errors='coerce')
        new_games_data = new_games_data.dropna(
            subset=['app_id']).astype({'app_id': int})
        logger.info(
            f"Processed additional game data from {new_games_csv_path}.")

        # Merge additional data into primary dataframe
        games_df = merge_game_data(games_df, new_games_data)

        # Load metadata and merge description into games data
        games_metadata = load_json_data(metadata_json_path)
        games_metadata_df = pd.DataFrame(games_metadata)
        if 'app_id' in games_metadata_df.columns and 'description' in games_metadata_df.columns:
            games_complete_df = games_df.merge(
                games_metadata_df[['app_id', 'description']],
                on='app_id',
                how='left'
            )
            logger.info("Merged metadata into games dataframe successfully.")
        else:
            logger.warning(
                "Metadata missing 'app_id' or 'description'; skipping metadata merge.")
            games_complete_df = games_df.copy()
        return games_complete_df
    except Exception as e:
        logger.error(f"Error preparing final dataset: {e}")
        return pd.DataFrame()
