import logging
import os
from config.config import TELEGRAM_API_TOKEN, GAME_CSV_PATH, COMBINED_DF_PATH, BASE_DIR
from data.data_loader import prepare_final_dataset
from data.preprocess import preprocess_games_df, clean_game_descriptions
from models.ncf_model import load_ncf_model, load_encoders
from models.transformer_model import load_transformer_model
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

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
    metadata_json_path = os.path.join(BASE_DIR, "data", "games_metadata.json")
    new_games_csv_path = os.path.join(
        BASE_DIR, "data", "cleaned_games_developers.csv")
    games_complete_df = prepare_final_dataset(
        GAME_CSV_PATH, metadata_json_path, new_games_csv_path)
    if games_complete_df.empty:
        logger.error("Failed to load games data. Exiting.")
        return
    games_complete_df = preprocess_games_df(games_complete_df)
    games_complete_df = clean_game_descriptions(games_complete_df)
    logger.info(f"Final games dataset contains {len(games_complete_df)} rows.")

    # --- Load Models ---
    ncf_model = load_ncf_model()
    user_encoder, game_encoder = load_encoders()
    tokenizer, transformer_model = load_transformer_model()
    if transformer_model is not None:
        transformer_model.eval()  # Set model to evaluation mode (inference only)
    if ncf_model is None or tokenizer is None or transformer_model is None:
        logger.error("One or more models failed to load. Exiting.")
        return

    # --- Load Vector Store ---
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = os.path.join(
            BASE_DIR, "vector_store_minilm_extended")
        vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        logger.info("Chroma vector store loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Chroma vector store: {e}")
        vector_store = None

    # --- Initialize Telegram Bot (v20+) ---
    application = ApplicationBuilder().token(TELEGRAM_API_TOKEN).build()
    dispatcher = application  # For compatibility with handler registration

    # Set shared resources in bot_data so handlers can access them
    dispatcher.bot_data["games_complete_df"] = games_complete_df
    dispatcher.bot_data["ncf_model"] = ncf_model
    dispatcher.bot_data["user_encoder"] = user_encoder
    dispatcher.bot_data["game_encoder"] = game_encoder
    dispatcher.bot_data["tokenizer"] = tokenizer
    dispatcher.bot_data["transformer_model"] = transformer_model
    dispatcher.bot_data["vector_store"] = vector_store

    # --- Register Handlers ---
    from handlers import feedback_handlers, price_handlers, reminder_handlers, video_handlers
    from handlers import recommendation_handlers, profile_handlers, comparison_handlers, greeting_handlers

    dispatcher.add_handler(CommandHandler(
        "start", greeting_handlers.handle_greeting))
    dispatcher.add_handler(CommandHandler(
        "feedback", feedback_handlers.handle_feedback_response))
    dispatcher.add_handler(CommandHandler(
        "price_tracker", price_handlers.handle_price_tracker))
    dispatcher.add_handler(CommandHandler(
        "reminder", reminder_handlers.handle_game_session_reminder))
    dispatcher.add_handler(CommandHandler(
        "compare", comparison_handlers.handle_game_comparison))
    dispatcher.add_handler(CommandHandler(
        "recommend", recommendation_handlers.handle_recommend_games))
    dispatcher.add_handler(CommandHandler(
        "video", video_handlers.handle_video_search))
    dispatcher.add_handler(CommandHandler(
        "filter", profile_handlers.handle_content_filter))
    dispatcher.add_handler(CommandHandler(
        "additionalinfo", profile_handlers.handle_additional_info))

    from handlers.intent_router import route_message
    dispatcher.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, route_message))

    # --- Start the Bot ---
    application.run_polling()
    logger.info("Bot started. Listening for incoming messages...")


if __name__ == '__main__':
    main()
