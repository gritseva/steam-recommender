# Test Recommenders module
# tests/test_recommenders.py

import unittest
import pandas as pd
from recommenders.content_based import get_advanced_similar_games, apply_genre_filter
from recommenders.collaborative import collaborative_filtering_with_fallback, get_user_embedding
from unittest.mock import MagicMock
from rapidfuzz import fuzz


class DummyVectorStore:
    def similarity_search(self, query, k):
        # Return dummy objects with a metadata attribute containing an app_id
        def DummyDoc(app_id): return type(
            "DummyDoc", (), {"metadata": {"app_id": app_id}})
        return [DummyDoc(app_id) for app_id in range(1, k+1)]


class DummyEncoder:
    def transform(self, ids):
        # Return a dummy numpy array (simulate encoding)
        import numpy as np
        return np.array(ids).reshape(-1, 1)

    def inverse_transform(self, indices):
        # Simply return the indices as app_ids
        return indices


class DummyNCFModel:
    def get_layer(self, layer_name):
        class DummyLayer:
            def __call__(self, x):
                import numpy as np
                # Return a dummy embedding (vector of ones)
                return np.ones((len(x), 5))

            def get_weights(self):
                import numpy as np
                # Return dummy weights for item embedding: 10 items, vector length 5
                return [np.ones((10, 5))]
        return DummyLayer()


class TestRecommenders(unittest.TestCase):
    def setUp(self):
        # Create a dummy games DataFrame
        self.df = pd.DataFrame({
            "app_id": list(range(1, 11)),
            "title": [f"Game {i}" for i in range(1, 11)],
            "tags": [["action", "adventure"] for _ in range(10)],
            "date_release": pd.to_datetime(["2020-01-01"] * 10)
        })
        self.vector_store = DummyVectorStore()
        self.user_encoder = DummyEncoder()
        self.game_encoder = DummyEncoder()
        self.ncf_model = DummyNCFModel()

    def test_get_advanced_similar_games(self):
        recommendations = get_advanced_similar_games(
            user_query="Game",
            combined_df=self.df,
            vector_store=self.vector_store,
            genres=["action"],
            release_year_filter={"comparator": "after", "year": 2019},
            k=5
        )
        self.assertIsInstance(recommendations, pd.DataFrame,
                              "Should return a DataFrame")
        self.assertLessEqual(len(recommendations), 5,
                             "Should return at most 5 recommendations")

    def test_collaborative_filtering_with_fallback(self):
        # Create a dummy session with liked_games and vector_store
        class DummySession:
            liked_games = {"Game 1", "Game 2"}
            user_preferences = {"genres": ["action"]}
            vector_store = self.vector_store
        session = DummySession()
        recs = collaborative_filtering_with_fallback(
            user_id=123,
            filtered_games=self.df,
            session=session,
            ncf_model=self.ncf_model,
            user_encoder=self.user_encoder,
            game_encoder=self.game_encoder,
            top_n=3
        )
        self.assertIsInstance(
            recs, pd.DataFrame, "Collaborative filtering fallback should return a DataFrame")

    def test_get_user_embedding(self):
        from recommenders.collaborative import get_user_embedding
        embedding = get_user_embedding(123, self.ncf_model, self.user_encoder)
        self.assertIsNotNone(embedding, "User embedding should not be None")


if __name__ == '__main__':
    unittest.main()
