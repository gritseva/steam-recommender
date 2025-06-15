# Test Models module
# tests/test_models.py

import unittest
from models.ncf_model import load_ncf_model, load_encoders, get_item_embeddings
from models.transformer_model import load_transformer_model


class TestModels(unittest.TestCase):
    def test_load_ncf_model(self):
        model = load_ncf_model()
        self.assertIsNotNone(model, "NCF model should load successfully")

    def test_load_encoders(self):
        user_encoder, game_encoder = load_encoders()
        self.assertIsNotNone(
            user_encoder, "User encoder should load successfully")
        self.assertIsNotNone(
            game_encoder, "Game encoder should load successfully")

    def test_get_item_embeddings(self):
        model = load_ncf_model()
        if model:
            embeddings = get_item_embeddings(model)
            self.assertIsNotNone(
                embeddings, "Item embeddings should be retrievable")

    def test_load_transformer_model(self):
        tokenizer, model = load_transformer_model()
        self.assertIsNotNone(tokenizer, "Tokenizer should load successfully")
        self.assertIsNotNone(
            model, "Transformer model should load successfully")


if __name__ == '__main__':
    unittest.main()
