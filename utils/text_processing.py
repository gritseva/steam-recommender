# Text Processing module
import re


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string.

    Args:
        text (str): The input text.

    Returns:
        str: The text with HTML tags removed.
    """
    return re.sub(r'<.*?>', '', text)
