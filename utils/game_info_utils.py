import pandas as pd
from fuzzywuzzy import fuzz, process
import logging

# Genre synonym mapping


def normalize_genre(genre):
    genre_synonyms = {
        'rpg': 'role-playing',
        'role playing': 'role-playing',
        'roleplaying': 'role-playing',
        'fps': 'shooter',
        'first person shooter': 'shooter',
        'action adventure': 'action',
        'adventure': 'adventure',
        'sports': 'sports',
        'sim': 'simulation',
        'simulator': 'simulation',
        'strategy': 'strategy',
        'indie': 'indie',
        'puzzle': 'puzzle',
        'platformer': 'platformer',
        'horror': 'horror',
        'racing': 'racing',
        'casual': 'casual',
        'multiplayer': 'multiplayer',
        'co-op': 'co-op',
        'coop': 'co-op',
        'sandbox': 'sandbox',
        'survival': 'survival',
        'open world': 'open world',
        'family': 'family',
        'family friendly': 'family',
        'story rich': 'story rich',
        'roguelike': 'roguelike',
        'roguelite': 'roguelike',
        'turn based': 'turn-based',
        'turn-based': 'turn-based',
        'card': 'card',
        'deckbuilder': 'card',
        'deck builder': 'card',
        'visual novel': 'visual novel',
        'anime': 'anime',
        'fighting': 'fighting',
        'shooter': 'shooter',
        'third person shooter': 'shooter',
        'platform': 'platformer',
        'building': 'building',
        'city builder': 'building',
        'management': 'management',
        'music': 'music',
        'rhythm': 'music',
        'stealth': 'stealth',
        'tactical': 'tactical',
        'tower defense': 'tower defense',
        'vr': 'vr',
        'virtual reality': 'vr',
        'zombie': 'zombie',
        'historical': 'historical',
        'sci-fi': 'sci-fi',
        'science fiction': 'sci-fi',
        'space': 'space',
        'western': 'western',
        'mmo': 'mmo',
        'massively multiplayer': 'mmo',
        'free to play': 'free to play',
        'f2p': 'free to play',
    }
    g = genre.strip().lower()
    return genre_synonyms.get(g, g)


def get_game_info_by_title(game_title, games_df):
    """
    Retrieve a game's info row from games_df using fuzzy matching on the title.
    Returns a pandas Series (row) or an empty DataFrame if not found.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[GameLookup] Looking up game title: {game_title}")
    if not isinstance(game_title, str) or games_df is None or games_df.empty:
        logger.info(f"[GameLookup] Invalid input or empty DataFrame.")
        return pd.DataFrame()
    game_title_normalized = game_title.lower()
    game_titles_normalized = games_df['title'].str.lower().tolist()
    match = process.extractOne(
        game_title_normalized, game_titles_normalized, scorer=fuzz.token_sort_ratio)
    logger.info(f"[GameLookup] Fuzzy match result: {match}")
    if match and match[1] > 80:
        matched_row = games_df.iloc[game_titles_normalized.index(match[0])]
        logger.info(f"[GameLookup] Returning matched row for: {match[0]}")
        return matched_row
    logger.info(
        f"[GameLookup] No match found above threshold for: {game_title}")
    return pd.DataFrame()


def filter_games_by_genre(games_df, user_genre):
    """
    Filter games DataFrame by normalized genre.
    """
    norm_genre = normalize_genre(user_genre)
    logger = logging.getLogger(__name__)
    logger.info(
        f"[GenreFilter] Filtering for genre: {user_genre} (normalized: {norm_genre})")
    # Assume genres column is a list of genres (already normalized to lowercase)
    return games_df[games_df['genres'].apply(lambda genres: norm_genre in [normalize_genre(g) for g in genres] if isinstance(genres, list) else False)]


def compare_games(game1_info, game2_info):
    """
    Compare two games (pandas Series) and return a formatted string with their key info.
    """
    if not isinstance(game1_info, pd.Series) or game1_info.empty:
        return "Error: Unable to find information for the first game."
    if not isinstance(game2_info, pd.Series) or game2_info.empty:
        return "Error: Unable to find information for the second game."
    game1_title = game1_info.get('title', 'Unknown')
    game2_title = game2_info.get('title', 'Unknown')
    game1_genres = ', '.join(game1_info['genres']) if isinstance(
        game1_info.get('genres'), list) else game1_info.get('genres', '')
    game2_genres = ', '.join(game2_info['genres']) if isinstance(
        game2_info.get('genres'), list) else game2_info.get('genres', '')
    game1_rating = game1_info.get('rating', 'N/A')
    game2_rating = game2_info.get('rating', 'N/A')
    game1_release = game1_info.get(
        'release_date', game1_info.get('date_release', 'N/A'))
    game2_release = game2_info.get(
        'release_date', game2_info.get('date_release', 'N/A'))
    game1_description = str(game1_info.get('about_game', game1_info.get(
        'description', 'No description available')))[:500]
    game2_description = str(game2_info.get('about_game', game2_info.get(
        'description', 'No description available')))[:500]
    comparison_result = f"**Comparison between '{game1_title}' and '{game2_title}':**\n\n"
    comparison_result += f"- **Genres:**\n  - {game1_title}: {game1_genres}\n  - {game2_title}: {game2_genres}\n\n"
    comparison_result += f"- **Ratings:**\n  - {game1_title}: {game1_rating}\n  - {game2_title}: {game2_rating}\n\n"
    comparison_result += f"- **Release Dates:**\n  - {game1_title}: {game1_release}\n  - {game2_title}: {game2_release}\n\n"
    comparison_result += f"- **Descriptions:**\n  - {game1_title}: {game1_description}\n  - {game2_title}: {game2_description}\n\n"
    return comparison_result
