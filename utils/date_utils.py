# utils/date_utils.py
import dateparser
import logging


def extract_date_time(user_message: str) -> str:
    logger = logging.getLogger(__name__)
    logger.info(f"[Reminder] Extracting date/time from: {user_message}")
    parsed_date = dateparser.parse(user_message, settings={
                                   'PREFER_DATES_FROM': 'future'})
    logger.info(f"[Reminder] dateparser.parse result: {parsed_date}")
    if not parsed_date:
        # Try to extract any date/time substring using search_dates
        try:
            from dateparser.search import search_dates
            found = search_dates(user_message, settings={
                                 'PREFER_DATES_FROM': 'future'})
            logger.info(f"[Reminder] search_dates result: {found}")
            if found:
                parsed_date = found[0][1]
        except Exception as e:
            logger.error(f"[Reminder] search_dates error: {e}")
    if "prime time" in user_message.lower():
        if not parsed_date:
            parsed_date = dateparser.parse("9:00 PM")
            logger.info(f"[Reminder] Fallback to 'prime time': {parsed_date}")
    if "every evening" in user_message.lower() and parsed_date:
        parsed_date = parsed_date.replace(hour=21, minute=0)
        logger.info(f"[Reminder] Adjusted for 'every evening': {parsed_date}")
    if parsed_date:
        formatted = parsed_date.strftime('%Y-%m-%d %H:%M')
        logger.info(f"[Reminder] Final parsed/returned time: {formatted}")
        return formatted
    else:
        logger.info(f"[Reminder] No valid date/time found in message.")
        return None


def extract_reminder_time(user_message: str) -> str:
    parsed_date = dateparser.parse(user_message, settings={
                                   'PREFER_DATES_FROM': 'future'})
    return parsed_date.strftime('%Y-%m-%d %H:%M') if parsed_date else None
