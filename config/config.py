import os
import torch

# ==== BASE DIRECTORY & PATHS ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VECTOR_STORE_DIR = os.path.join(BASE_DIR, 'vector_store')

# ==== DATA PATHS ====
GAME_CSV_PATH = os.path.join(DATA_DIR, 'games.csv')
COMBINED_DF_PATH = os.path.join(DATA_DIR, 'combined_df.pkl')

# ==== MODEL PATHS ====
NCF_MODEL_PATH = os.path.join(MODELS_DIR, 'trained_model.keras')
USER_ENCODER_PATH = os.path.join(MODELS_DIR, 'user_encoder.pkl')
GAME_ENCODER_PATH = os.path.join(MODELS_DIR, 'game_encoder.pkl')
TRANSFORMER_CHECKPOINT = os.path.join(MODELS_DIR, 'saved_checkpoint')

# ==== API KEYS (from environment variables) ====
STEAM_API_KEY = os.getenv('STEAM_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN')

# ==== OTHER CONFIGURATIONS ====
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_N_RECOMMENDATIONS = 5  # Default recommendation count
MIN_POSITIVE_RATIO = 70  # Filtering threshold (%)
MIN_USER_REVIEWS = 100  # Minimum reviews required

# ==== DEVICE & LOGGING ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGING_LEVEL = "INFO"
