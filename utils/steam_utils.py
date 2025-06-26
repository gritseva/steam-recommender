# utils/steam_utils.py
import os
import re
import logging
import requests
import pandas as pd
from datetime import datetime


def extract_steam_id(user_message: str) -> int:
    """
    Extract a Steam ID from the user message.
    """
    match = re.search(r'\b(\d{5,})\b', user_message)
    return int(match.group(1)) if match else None


def store_steam_id(user_message: str, session) -> str:
    """
    Store the Steam ID in the session.
    """
    match = re.search(r'\b(\d{5,})\b', user_message)
    if match:
        session.user_id = int(match.group(1))
        return f"Thanks for providing your Steam ID ({session.user_id}). I'll use this to personalize recommendations!"
    return None


def fetch_steam_profile(steam_id: int) -> dict:
    """
    Fetch the Steam profile information for a given Steam ID.
    Returns a dict. If an error occurs, returns a dict with an 'error' key and a message.
    """
    api_key = os.getenv('STEAM_API_KEY')
    if not api_key:
        logging.error("Steam API key not found.")
        return {"error": "Steam API key not found. Please set the STEAM_API_KEY environment variable."}
    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={api_key}&steamid={steam_id}&include_appinfo=1&format=json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 403:
            logging.error(
                "Steam API returned 403 Forbidden. Likely invalid API key or private profile.")
            return {"error": "Steam API returned 403 Forbidden. Your profile may be private or the API key is invalid."}
        if response.status_code != 200:
            logging.error(f"Steam API error: {response.status_code}")
            return {"error": f"Steam API error: {response.status_code}"}
        data = response.json()
        if 'response' in data and ('games' not in data['response'] or not data['response']['games']):
            logging.warning("No games found or profile is private.")
            return {"error": "No games found in your Steam profile, or your profile is private."}
        return data
    except Exception as e:
        logging.error(f"Error fetching Steam profile: {e}")
        return {"error": f"Error fetching Steam profile: {e}"}


def analyze_profile(profile_data: dict, games_complete_df: pd.DataFrame) -> dict:
    """
    Analyze the user's Steam profile data and return a summary.
    """
    games = profile_data.get('response', {}).get('games', [])
    if not games:
        return {"error": "No games found in your Steam profile."}

    total_playtime = sum(game.get('playtime_forever', 0)
                         for game in games) // 60
    most_played_games = sorted(games, key=lambda x: x.get(
        'playtime_forever', 0), reverse=True)[:5]
    game_ids = [game['appid'] for game in most_played_games]
    game_details = games_complete_df[games_complete_df['app_id'].isin(
        game_ids)]

    genre_list = []
    for game in game_details.itertuples():
        genres = game.genres
        if isinstance(genres, list):
            genre_list.extend(genres)
        elif isinstance(genres, str):
            genre_list.extend(genres.split(','))
    favorite_genres = pd.Series(
        genre_list).value_counts().head(3).index.tolist()

    profile_summary = {
        'total_playtime': total_playtime,
        'most_played_games': [],
        'favorite_genres': favorite_genres
    }
    for game in most_played_games:
        app_id = game['appid']
        playtime_hours = game.get('playtime_forever', 0) // 60
        title_row = game_details[game_details['app_id'] == app_id]
        if not title_row.empty:
            title = title_row.iloc[0]['title']
            profile_summary['most_played_games'].append(
                f"{title} ({playtime_hours} hours)")
        else:
            profile_summary['most_played_games'].append(
                f"Unknown Game (AppID: {app_id}) ({playtime_hours} hours)")
    return profile_summary
