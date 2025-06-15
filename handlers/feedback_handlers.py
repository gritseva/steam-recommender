# Feedback Handlers module
import logging
from telegram import Update
from telegram.ext import CallbackContext
# Ensure this module exists
from utils.response_generation import generate_custom_response
from sessions.session_manager import get_user_session

logger = logging.getLogger(__name__)


def handle_feedback_response(update: Update, context: CallbackContext) -> None:
    """
    Handle the '/feedback' command by acknowledging user feedback and prompting for additional details.
    """
    user_message = update.message.text
    session = get_user_session(update.message.chat_id)

    # Use the transformer components loaded in bot_data.
    tokenizer = context.bot_data.get("tokenizer")
    transformer_model = context.bot_data.get("transformer_model")
    device = transformer_model.device if hasattr(
        transformer_model, 'device') else "cpu"

    prompt = (
        "<s>[INST] The user has provided feedback on the recommendations:\n\n"
        f"User Message: \"{user_message}\"\n\n"
        "Please provide a brief, polite acknowledgment and ask for additional details to improve future recommendations.\n"
        "Response:[/INST]"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        response = transformer_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        result_text = tokenizer.decode(
            response[0], skip_special_tokens=True).strip()
        custom_response = generate_custom_response(result_text)
        update.message.reply_text(custom_response)
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        update.message.reply_text(
            "An error occurred while processing your feedback. Please try again later.")
