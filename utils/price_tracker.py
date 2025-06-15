import os
import re
import logging
import requests
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def get_request(url: str, parameters: dict = None) -> dict:
    """
    Perform an HTTP GET request and return the JSON response.
    """
    try:
        response = requests.get(url, params=parameters, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error during GET request to {url}: {e}")
        return {}


def parse_steam_price_request(appid: int) -> dict:
    """
    Retrieve current price information for a given Steam app.

    Args:
        appid (int): The Steam application ID.

    Returns:
        dict: A dictionary with price information and a timestamp.
    """
    url = "http://store.steampowered.com/api/appdetails/"
    parameters = {"appids": appid}
    json_data = get_request(url, parameters)

    if str(appid) in json_data and json_data[str(appid)]['success']:
        data = json_data[str(appid)]['data']
        if 'price_overview' in data and data['price_overview']:
            price_info = data['price_overview']
            current_price = price_info.get(
                'final_formatted', 'Price data unavailable')
            original_price = price_info.get(
                'initial_formatted', 'Price data unavailable')
            discount = price_info.get('discount_percent', 'No discount')
            currency = price_info.get('currency', 'EUR')
        else:
            current_price = original_price = discount = 'Price data unavailable'
            currency = 'EUR'
    else:
        current_price = original_price = discount = 'Price data unavailable'
        currency = 'EUR'

    return {
        'appid': appid,
        'current_price': current_price,
        'original_price': original_price,
        'discount_percent': discount,
        'currency': currency,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def save_price_data_to_csv(data: dict, filename: str = '../data/price_history.csv'):
    """
    Save or append price data to a CSV file.
    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = pd.DataFrame([data])
    try:
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(filename, index=False)
    except FileNotFoundError:
        df.to_csv(filename, index=False)
    except Exception as e:
        logger.error(f"Error saving price data to CSV: {e}")


def track_price_changes(app_list: list, download_path: str = '../data', filename: str = 'price_history.csv') -> list:
    """
    Track price changes for a list of Steam app IDs.

    Args:
        app_list (list): List of Steam app IDs.
        download_path (str): Directory to store the CSV file.
        filename (str): CSV file name.

    Returns:
        list: A list of price data dictionaries.
    """
    price_data_list = []
    for appid in app_list:
        price_data = parse_steam_price_request(appid)
        save_price_data_to_csv(
            price_data, filename=f"{download_path}/{filename}")
        logger.info(
            f"Tracked price for app {appid} at {price_data['date']}: {price_data['current_price']}")
        price_data_list.append(price_data)
    return price_data_list


def match_titles_to_app_ids(titles: list, games_complete_df: pd.DataFrame) -> list:
    """
    Match game titles to their app IDs using case-insensitive substring matching.

    Args:
        titles (list): List of game titles.
        games_complete_df (pd.DataFrame): DataFrame containing game information.

    Returns:
        list: List of matched app IDs.
    """
    app_ids = []
    games_complete_df['title'] = games_complete_df['title'].astype(str)
    for title in titles:
        match = games_complete_df[games_complete_df['title'].str.contains(
            title, case=False, na=False)]
        if not match.empty:
            app_ids.append(match.iloc[0]['app_id'])
    return app_ids
