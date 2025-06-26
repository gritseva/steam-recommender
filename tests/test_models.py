# Test Models module
# tests/test_models.py
import unittest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from models.ncf_model import load_ncf_model, load_encoders, get_item_embeddings
from models.transformer_model import load_transformer_model


class TestModels(unittest.TestCase):
    def test_load_ncf_model(self):
        """Model should load and expose expected layers."""
        model = load_ncf_model()
        self.assertIsNotNone(model, "NCF model should load successfully")
        # Should have a matrix factorization embedding layer for items
        self.assertTrue(hasattr(model, 'get_layer'),
                        "Model missing get_layer method")
        layer = model.get_layer('item_embedding')
        self.assertIsNotNone(
            layer, "Missing 'item_embedding' layer in NCF model")

    def test_load_encoders(self):
        """Encoders should be sklearn LabelEncoders with transform/inverse_transform."""
        user_enc, game_enc = load_encoders()
        for enc, name in [(user_enc, 'user'), (game_enc, 'game')]:
            self.assertIsInstance(
                enc, LabelEncoder, f"{name}_encoder should be LabelEncoder")
            # Test transform/inverse_transform round-trip
            sample = ['A', 'B', 'C']
            enc.fit(sample)
            transformed = enc.transform(sample)
            self.assertTrue(isinstance(transformed, np.ndarray) or hasattr(transformed, '__iter__'),
                            "Transform should return iterable of encoded labels")
            inverted = enc.inverse_transform(transformed)
            self.assertListEqual(list(inverted), sample,
                                 "inverse_transform should reverse transform exactly")

    def test_get_item_embeddings(self):
        """Item embeddings should be a 2D numpy array."""
        model = load_ncf_model()
        embeddings = get_item_embeddings(model)
        self.assertIsInstance(embeddings, np.ndarray,
                              "Embeddings should be a numpy array")
        self.assertEqual(embeddings.ndim, 2, "Embeddings array should be 2D")
        # Both dimensions should be positive
        self.assertGreater(
            embeddings.shape[0], 0, "Embeddings should have at least one item")
        self.assertGreater(
            embeddings.shape[1], 0, "Embeddings should have non-zero vector length")

    def test_load_transformer_model(self):
        """Transformer model loader should return a tokenizer and a model with expected APIs."""
        tokenizer, model = load_transformer_model()
        # Tokenizer contract
        self.assertIsNotNone(tokenizer, "Tokenizer should load successfully")
        self.assertIsInstance(tokenizer, PreTrainedTokenizerBase,
                              "Tokenizer should be a HuggingFace PreTrainedTokenizerBase")
        for method in ('encode', 'decode', 'batch_encode_plus'):
            self.assertTrue(hasattr(tokenizer, method),
                            f"Tokenizer missing method '{method}'")

        # Model contract
        self.assertIsNotNone(
            model, "Transformer model should load successfully")
        self.assertIsInstance(model, PreTrainedModel,
                              "Model should be a HuggingFace PreTrainedModel")
        # Must support generate() and have a device attribute
        self.assertTrue(hasattr(model, 'generate'),
                        "Model missing 'generate' method")
        self.assertTrue(hasattr(model, 'device'),
                        "Model missing 'device' attribute")


if __name__ == '__main__':
    unittest.main()
