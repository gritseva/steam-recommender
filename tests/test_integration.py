# tests/test_integration.py
import unittest
import os
import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock

from config.config import GAME_CSV_PATH, BASE_DIR
from data.data_loader import prepare_final_dataset
from data.preprocess import preprocess_games_df, clean_game_descriptions
from models.ncf_model import load_ncf_model, load_encoders, get_item_embeddings
from models.transformer_model import load_transformer_model
from utils.llm_processing import extract_game_titles, infer_user_preferences_with_llm
from utils.translation import detect_language, translate_to_english


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load and prepare dataset once for all tests
        df = prepare_final_dataset(
            GAME_CSV_PATH,
            os.path.join(BASE_DIR, 'data', 'games_metadata.json'),
            os.path.join(BASE_DIR, 'data', 'cleaned_games_developers.csv')
        )
        df = preprocess_games_df(df)
        cls.df = clean_game_descriptions(df)

    def test_dataset_not_empty_and_has_required_columns(self):
        self.assertFalse(self.df.empty, "Final dataset should not be empty")
        for col in ('app_id', 'title', 'genres'):
            self.assertIn(col, self.df.columns,
                          f"Dataset should contain column '{col}'")

    def test_model_loading_and_embeddings(self):
        # NCF model and encoders
        ncf_model = load_ncf_model()
        self.assertIsNotNone(ncf_model, "NCF model should load successfully")
        user_enc, game_enc = load_encoders()
        self.assertIsNotNone(user_enc, "User encoder should load successfully")
        self.assertIsNotNone(game_enc, "Game encoder should load successfully")

        # Transformer model
        tokenizer, transformer_model = load_transformer_model()
        self.assertIsNotNone(tokenizer, "Tokenizer should load successfully")
        self.assertIsNotNone(
            transformer_model, "Transformer model should load successfully")

        # Item embeddings
        embeddings = get_item_embeddings(ncf_model)
        self.assertIsNotNone(
            embeddings, "Item embeddings should be retrievable")
        # Should be at least 2D: items x vector_size
        self.assertGreaterEqual(len(embeddings.shape), 2,
                                "Embeddings should be a 2D array")

    def test_llm_processing_with_mocks(self):
        # Mock tokenizer & model for LLM functions
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.device = 'cpu'

        # Simulate tokenizer call returning tensors
        mock_tokenizer.return_value = {'input_ids': [], 'attention_mask': []}
        # First decode call: titles, second: JSON prefs
        mock_tokenizer.decode.side_effect = [
            "Game X, Game Y",
            '{"liked_games": ["Game X", "Game Y"], "genres": ["genre"], "excluded_tags": ["tag"]}'
        ]
        mock_model.generate.return_value = [b'dummy']

        context = SimpleNamespace(bot_data={
            'tokenizer': mock_tokenizer,
            'transformer_model': mock_model
        })

        titles = extract_game_titles("msg", context)
        self.assertListEqual(titles, ["Game X", "Game Y"],
                             "extract_game_titles should split comma-separated titles")

        prefs = infer_user_preferences_with_llm("msg", context)
        self.assertListEqual(prefs['liked_games'], ["Game X", "Game Y"])
        self.assertListEqual(prefs['genres'], ["genre"])
        self.assertListEqual(prefs['excluded_tags'], ["tag"])

    def test_translation(self):
        sample = "Hola, ¿cómo estás?"
        lang = detect_language(sample)
        self.assertIn(lang, [
                      'auto', 'es'], "Language detection should identify Spanish as 'es' or 'auto'")
        translation = translate_to_english(sample)
        self.assertIsInstance(
            translation, str, "Translation should return a string")
        self.assertIn("how are you", translation.lower(),
                      "Translation should contain the English equivalent")


if __name__ == '__main__':
    unittest.main()
