# handlers/intent_router.py
import logging
from telegram import Update
from telegram.ext import CallbackContext
from utils.llm_processing import parse_user_intent
from utils import translation
from handlers import (
    price_handlers, video_handlers, feedback_handlers, reminder_handlers,
    recommendation_handlers, profile_handlers, comparison_handlers,
    out_of_context_handlers, greeting_handlers
)
from utils.llm_processing import extract_game_titles
from utils.date_utils import extract_date_time
from handlers.greeting_handlers import handle_greeting

logger = logging.getLogger(__name__)


# Map intent names to handler functions
INTENT_HANDLERS = {
    "greeting": handle_greeting,
    "recommend_games": recommendation_handlers.handle_recommend_games,
    "feedback": feedback_handlers.handle_feedback_response,
    "price_tracker": price_handlers.handle_price_tracker,
    "video_search": video_handlers.handle_video_search,
    "game_session_reminder": reminder_handlers.handle_game_session_reminder,
    "game_comparison": comparison_handlers.handle_game_comparison,
    "additional_info": profile_handlers.handle_additional_info,
    "content_filter": profile_handlers.handle_content_filter,
    "opinion_request": recommendation_handlers.handle_opinion_request,
    "out_of_context": out_of_context_handlers.handle_out_of_context_response,
    "top_games_request": recommendation_handlers.handle_top_games_request,
    # Re-route to the same handler
    "user_gaming_profile": profile_handlers.handle_additional_info,
    "translation": translation.handle_translation_request,
    # Route unknown to out_of_context
    "unknown": out_of_context_handlers.handle_out_of_context_response,
}


async def route_message(update: Update, context: CallbackContext) -> None:
    """
    Analyze user message intent and route to the appropriate handler.
    """
    if not update.message or not update.message.text:
        return

    user_message = update.message.text
    logger.info(
        f"Routing message from user {update.message.chat_id}: '{user_message}'")

    intent = parse_user_intent(user_message, context)
    logger.info(f"Detected intent: {intent}")

    # Keyword-based fallback: if intent is reminder but no time or game, treat as out_of_context
    if intent == "game_session_reminder":
        has_time = extract_date_time(user_message)
        has_game = bool(extract_game_titles(user_message, context))
        if not has_time and not has_game:
            logger.info(
                "[IntentRouter] Overriding intent to out_of_context due to lack of time/game in reminder.")
            intent = "out_of_context"

    handler = INTENT_HANDLERS.get(
        intent, out_of_context_handlers.handle_out_of_context_response)

    if handler:
        await handler(update, context)
    else:
        # This case should ideally not be reached if "unknown" is in INTENT_HANDLERS
        logger.warning(
            f"No handler found for intent '{intent}'. Using fallback.")
        await out_of_context_handlers.handle_out_of_context_response(update, context)
