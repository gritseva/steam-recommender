import logging
import os
from config.config import TELEGRAM_API_TOKEN, GAME_CSV_PATH, COMBINED_DF_PATH, BASE_DIR
from data.data_loader import prepare_final_dataset
from data.preprocess import preprocess_games_df, clean_game_descriptions
from models.ncf_model import load_ncf_model, load_encoders
from models.transformer_model import load_transformer_model
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# If you have a logging utilities module, you can call its setup function.
# Otherwise, we'll configure basic logging.


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def main():
    # Initialize logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Game Recommender Bot...")

    # --- Load Data ---
    # Define additional file paths (adjust as needed or add to your config)
    metadata_json_path = os.path.join(BASE_DIR, "data", "games_metadata.json")
    new_games_csv_path = os.path.join(
        BASE_DIR, "data", "cleaned_games_developers.csv")

    # Prepare the final dataset (merging CSV, JSON metadata, and additional game data)
    games_complete_df = prepare_final_dataset(
        GAME_CSV_PATH, metadata_json_path, new_games_csv_path)
    if games_complete_df.empty:
        logger.error("Failed to load games data. Exiting.")
        return

    # Preprocess and clean the dataset
    games_complete_df = preprocess_games_df(games_complete_df)
    games_complete_df = clean_game_descriptions(games_complete_df)
    logger.info(f"Final games dataset contains {len(games_complete_df)} rows.")

    # --- Load Models ---
    ncf_model = load_ncf_model()
    user_encoder, game_encoder = load_encoders()
    tokenizer, transformer_model = load_transformer_model()

    if ncf_model is None or tokenizer is None or transformer_model is None:
        logger.error("One or more models failed to load. Exiting.")
        return

    # (Optional) Load your vector store here and assign it if needed
    # vector_store = ...
    # For now, we assume vector_store is handled in the session or other modules.

    # --- Initialize Telegram Bot ---
    updater = Updater(token=TELEGRAM_API_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Set shared resources in bot_data so handlers can access them
    dispatcher.bot_data["games_complete_df"] = games_complete_df
    dispatcher.bot_data["ncf_model"] = ncf_model
    dispatcher.bot_data["user_encoder"] = user_encoder
    dispatcher.bot_data["game_encoder"] = game_encoder
    dispatcher.bot_data["tokenizer"] = tokenizer
    dispatcher.bot_data["transformer_model"] = transformer_model
    # dispatcher.bot_data["vector_store"] = vector_store  # If available

    # --- Register Handlers ---
    # Import handlers from your handlers package
    from handlers import feedback_handlers, price_handlers, reminder_handlers, telegram_handlers, video_handlers

    # Command handlers
    dispatcher.add_handler(CommandHandler(
        "feedback", feedback_handlers.handle_feedback_response))
    dispatcher.add_handler(CommandHandler(
        "price_tracker", price_handlers.handle_price_tracker))
    dispatcher.add_handler(CommandHandler(
        "reminder", reminder_handlers.handle_game_session_reminder))
    dispatcher.add_handler(CommandHandler(
        "compare", telegram_handlers.handle_game_comparison))
    dispatcher.add_handler(CommandHandler(
        "recommend", telegram_handlers.handle_recommend_games))
    dispatcher.add_handler(CommandHandler(
        "video", video_handlers.handle_video_search))

    # Fallback for messages that do not match any command (optional)
    from handlers import telegram_handlers as default_handlers
    dispatcher.add_handler(MessageHandler(
        Filters.text & ~Filters.command, default_handlers.handle_unknown_intent))

    # --- Start the Bot ---
    updater.start_polling()
    logger.info("Bot started. Listening for incoming messages...")
    updater.idle()


if __name__ == '__main__':
    main()
