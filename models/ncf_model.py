import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from config.config import NCF_MODEL_PATH, USER_ENCODER_PATH, GAME_ENCODER_PATH
import logging

logger = logging.getLogger(__name__)


def load_ncf_model():
    """
    Load the pre-trained Neural Collaborative Filtering (NCF) model.

    This function clears any existing TensorFlow sessions to prevent memory leaks.

    Returns:
        model: The loaded TensorFlow Keras model, or None if loading fails.
    """
    try:
        tf.keras.backend.clear_session()  # Prevent memory leaks from previous models
        model = load_model(NCF_MODEL_PATH, compile=False)
        logger.info(f"NCF model loaded successfully from {NCF_MODEL_PATH}.")
        return model
    except Exception as e:
        logger.error(f"Error loading NCF model from {NCF_MODEL_PATH}: {e}")
        return None


def get_item_embeddings(model):
    """
    Retrieve the item embeddings from the NCF model.

    Args:
        model: The loaded NCF model.

    Returns:
        np.ndarray: The item embedding matrix, or None if retrieval fails.
    """
    try:
        # Assume the layer is named 'item_embedding'
        embeddings = model.get_layer('item_embedding').get_weights()[0]
        logger.info("Successfully retrieved item embeddings.")
        return embeddings
    except Exception as e:
        logger.error(f"Error retrieving item embeddings: {e}")
        return None


def load_encoders():
    """
    Load user and game encoders from pre-saved files.

    Returns:
        tuple: (user_encoder, game_encoder), or (None, None) if loading fails.
    """
    try:
        user_encoder = joblib.load(USER_ENCODER_PATH)
        game_encoder = joblib.load(GAME_ENCODER_PATH)
        logger.info("User and game encoders loaded successfully.")
        return user_encoder, game_encoder
    except Exception as e:
        logger.error(
            f"Error loading encoders from {USER_ENCODER_PATH} and {GAME_ENCODER_PATH}: {e}")
        return None, None
