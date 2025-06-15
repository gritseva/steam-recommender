from googletrans import Translator
import logging

logger = logging.getLogger(__name__)

# Initialize the translator once
translator = Translator()


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text (str): The text for language detection.

    Returns:
        str: The detected language code (e.g., 'en', 'es').
    """
    try:
        detection = translator.detect(text)
        logger.info(f"Detected language: {detection.lang}")
        return detection.lang
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return 'en'  # Fallback to English


def translate_to_english(text: str) -> str:
    """
    Translate the given text to English.

    Args:
        text (str): The text to translate.

    Returns:
        str: The translated text in English.
    """
    try:
        translation = translator.translate(text, dest='en')
        logger.info("Translated text to English.")
        return translation.text
    except Exception as e:
        logger.error(f"Error translating text to English: {e}")
        return text


def translate_from_english(text: str, dest_language: str) -> str:
    """
    Translate English text to the specified destination language.

    Args:
        text (str): The text in English.
        dest_language (str): The target language code.

    Returns:
        str: The translated text in the destination language.
    """
    try:
        translation = translator.translate(text, dest=dest_language)
        logger.info(f"Translated text from English to {dest_language}.")
        return translation.text
    except Exception as e:
        logger.error(
            f"Error translating text from English to {dest_language}: {e}")
        return text


def handle_translation(user_message: str, session) -> tuple:
    """
    Translate the user message to English if necessary and return the translated text along with
    the detected language.

    Args:
        user_message (str): The original user message.
        session: The user session (reserved for future enhancements).

    Returns:
        tuple: (translated_text, detected_language)
    """
    try:
        detected_language = detect_language(user_message)
        if detected_language == 'en':
            return user_message, 'en'
        translated_text = translate_to_english(user_message)
        return translated_text, detected_language
    except Exception as e:
        logger.error(f"Error in handle_translation: {e}")
        return None, None
