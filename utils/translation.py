from deep_translator import GoogleTranslator
import logging
from langdetect import detect, LangDetectException
from telegram import Update
from telegram.ext import CallbackContext

logger = logging.getLogger(__name__)

# No need to initialize a translator instance; use class methods


def detect_language(text: str) -> str:
    """
    Detect the language of the given text using langdetect.
    Args:
        text (str): The text for language detection.
    Returns:
        str: The detected language code (e.g., 'en', 'es'), or 'unknown' if detection fails.
    """
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        logger.warning("Could not detect language for text: %s", text)
        return 'unknown'
    except Exception as e:
        logger.error(f"Error in detect_language: {e}")
        return 'unknown'


def translate_to_english(text: str) -> str:
    """
    Translate the given text to English using deep-translator.
    Args:
        text (str): The text to translate.
    Returns:
        str: The translated text in English.
    """
    try:
        translation = GoogleTranslator(
            source='auto', target='en').translate(text)
        logger.info("Translated text to English.")
        return translation
    except Exception as e:
        logger.error(f"Error translating text to English: {e}")
        return text


def translate_from_english(text: str, dest_language: str) -> str:
    """
    Translate English text to the specified destination language using deep-translator.
    Args:
        text (str): The text in English.
        dest_language (str): The target language code.
    Returns:
        str: The translated text in the destination language.
    """
    try:
        translation = GoogleTranslator(
            source='en', target=dest_language).translate(text)
        logger.info(f"Translated text from English to {dest_language}.")
        return translation
    except Exception as e:
        logger.error(
            f"Error translating text from English to {dest_language}: {e}")
        return text


def handle_translation(user_message: str, session) -> tuple:
    """
    Detect the language and translate the user message to English if necessary.
    Returns the translated text and the detected language code.
    Args:
        user_message (str): The original user message.
        session: The user session (reserved for future enhancements).
    Returns:
        tuple: (translated_text, detected_language)
    """
    try:
        detected_language = detect_language(user_message)
        if detected_language == 'en' or detected_language == 'unknown':
            return user_message, detected_language
        translated_text = translate_to_english(user_message)
        return translated_text, detected_language
    except Exception as e:
        logger.error(f"Error in handle_translation: {e}")
        return None, None


async def handle_translation_request(update: Update, context: CallbackContext) -> None:
    """
    Handle translation requests from users.
    """
    user_message = update.message.text
    user_id = update.message.chat_id

    try:
        # Detect language
        detected_language = detect_language(user_message)

        if detected_language == 'en':
            await update.message.reply_text(
                "Your message is already in English. No translation needed.")
            return

        # Translate to English
        translated_text = translate_to_english(user_message)

        if translated_text != user_message:
            response = (
                f"Detected language: {detected_language}\n"
                f"Translation: {translated_text}"
            )
            await update.message.reply_text(response)
        else:
            await update.message.reply_text(
                "I couldn't translate your message. Please try again.")

    except Exception as e:
        logger.error(f"Error in translation handler: {e}")
        await update.message.reply_text(
            "An error occurred during translation. Please try again.")
