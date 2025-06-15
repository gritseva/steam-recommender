# utils/youtube_utils.py
import logging
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os


def extract_video_type(user_message: str) -> str:
    video_types = ['trailer', 'gameplay', 'review',
                   'walkthrough', 'guide', 'preview', 'cutscene', 'speedrun']
    for video_type in video_types:
        if video_type in user_message.lower():
            return video_type
    return None


def parse_duration(duration: str) -> float:
    time_parts = re.findall(r'(\d+)([HMS])', duration)
    total_minutes = 0
    for value, unit in time_parts:
        value = int(value)
        if unit == 'H':
            total_minutes += value * 60
        elif unit == 'M':
            total_minutes += value
        elif unit == 'S':
            total_minutes += value / 60
    return total_minutes


def search_youtube_videos(query: str, max_results: int = 1) -> dict:
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError(
            "YouTube API key not found. Please set the YOUTUBE_API_KEY environment variable.")
    refined_query = query + " walkthrough full playthrough guide"
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.search().list(
            part='snippet',
            q=refined_query,
            maxResults=max_results,
            type='video',
            videoEmbeddable='true',
            videoSyndicated='true',
            safeSearch='moderate',
            order='relevance'
        )
        response = request.execute()
        if response['items']:
            item = response['items'][0]
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            url = f'https://www.youtube.com/watch?v={video_id}'
            video_details = youtube.videos().list(
                part="contentDetails", id=video_id).execute()
            duration = video_details['items'][0]['contentDetails']['duration']
            minutes = parse_duration(duration)
            if minutes < 10:
                return {'title': f"Short video found: {title}", 'url': url}
            return {'title': title, 'url': url}
        else:
            return None
    except HttpError as e:
        logging.error(f"YouTube API error: {e}")
        return None
