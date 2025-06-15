# utils/date_utils.py
import dateparser


def extract_date_time(user_message: str) -> str:
    parsed_date = dateparser.parse(user_message, settings={
                                   'PREFER_DATES_FROM': 'future'})
    if "prime time" in user_message.lower():
        if not parsed_date:
            parsed_date = dateparser.parse("9:00 PM")
    if "every evening" in user_message.lower() and parsed_date:
        parsed_date = parsed_date.replace(hour=21, minute=0)
    return parsed_date.strftime('%Y-%m-%d %H:%M') if parsed_date else None


def extract_reminder_time(user_message: str) -> str:
    parsed_date = dateparser.parse(user_message, settings={
                                   'PREFER_DATES_FROM': 'future'})
    return parsed_date.strftime('%Y-%m-%d %H:%M') if parsed_date else None
