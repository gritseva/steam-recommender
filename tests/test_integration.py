# File: tests/test_integration.py

import unittest
import os
from config.config import GAME_CSV_PATH, BASE_DIR
from data.data_loader import prepare_final_dataset
from data.preprocess import preprocess_games_df, clean_game_descriptions
from models.ncf_model import load_ncf_model, load_encoders
from models.transformer_model import load_transformer_model
from sessions.session_manager import get_user_session
from utils.llm_processing import extract_game_titles, infer_user_preferences_with_llm
from utils.translation import detect_language, translate_to_english


# python -m unittest discover -s tests


class IntegrationTest(unittest.TestCase):
    def test_data_loading_and_preprocessing(self):
        metadata_json_path = os.path.join(
            BASE_DIR, "data", "games_metadata.json")
        new_games_csv_path = os.path.join(
            BASE_DIR, "data", "cleaned_games_developers.csv")

        # Prepare final dataset (merging CSV, JSON, and additional CSV)
        df = prepare_final_dataset(
            GAME_CSV_PATH, metadata_json_path, new_games_csv_path)
        self.assertFalse(df.empty, "Final dataset should not be empty")

        # Preprocess and clean the dataset
        df = preprocess_games_df(df)
        df = clean_game_descriptions(df)
        self.assertIn("title", df.columns,
                      "DataFrame should contain 'title' column after processing")

    def test_model_loading(self):
        ncf_model = load_ncf_model()
        self.assertIsNotNone(
            ncf_model, "NCF model should be loaded successfully")

        user_encoder, game_encoder = load_encoders()
        self.assertIsNotNone(user_encoder, "User encoder should be loaded")
        self.assertIsNotNone(game_encoder, "Game encoder should be loaded")

        tokenizer, transformer_model = load_transformer_model()
        self.assertIsNotNone(tokenizer, "Tokenizer should be loaded")
        self.assertIsNotNone(
            transformer_model, "Transformer model should be loaded")

    def test_llm_processing(self):
        sample_message = "I really love The Witcher 3 and Cyberpunk 2077!"
        titles = extract_game_titles(sample_message)
        self.assertIsInstance(
            titles, list, "extract_game_titles should return a list")
        # Expecting at least one title extracted (depending on your LLM prompt responses)
        self.assertGreater(
            len(titles), 0, "At least one game title should be extracted")

        preferences = infer_user_preferences_with_llm(sample_message)
        self.assertIn("liked_games", preferences,
                      "Preferences output should contain 'liked_games' key")
        self.assertIsInstance(
            preferences["liked_games"], list, "'liked_games' should be a list")

    def test_translation(self):
        sample_message = "Hola, ¿cómo estás?"
        lang = detect_language(sample_message)
        self.assertEqual(lang, 'es', "Language should be detected as 'es'")
        translated = translate_to_english(sample_message)
        self.assertIsInstance(
            translated, str, "Translation should return a string")
        self.assertNotEqual(translated, sample_message,
                            "Translated text should differ from the original if not in English")

    def test_session_management(self):
        user_id = 12345
        session = get_user_session(user_id)
        session.update_likes(["The Witcher 3"])
        self.assertIn("The Witcher 3", session.liked_games,
                      "Session should record updated liked games")
        session.set_user_id(user_id)
        self.assertEqual(session.user_id, user_id,
                         "User ID should be set correctly in session")


if __name__ == '__main__':
    unittest.main()
