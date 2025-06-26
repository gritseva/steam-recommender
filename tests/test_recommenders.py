# tests/test_recommenders.py

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from recommenders.content_based import get_advanced_similar_games, apply_genre_filter
from recommenders.collaborative import collaborative_filtering_with_fallback, get_user_embedding


class DummyDoc:
    def __init__(self, app_id):
        self.metadata = {'app_id': app_id}


class DummyVectorStore:
    def similarity_search(self, query, k):
        # Return docs with app_id 1..k
        return [DummyDoc(i) for i in range(1, k+1)]


class DummySession:
    def __init__(self):
        self.liked_games = set([1, 2])
        self.user_preferences = {'genres': [
            'action'], 'release_year_filter': None}
        self.vector_store = DummyVectorStore()
        self.user_id = 123


class DummyNCFModel:
    def get_layer(self, layer_name):
        class Layer:
            def __call__(self, x):
                # return a numpy-like object with .numpy()
                arr = np.ones((len(x), 4))
                mock = MagicMock()
                mock.numpy.return_value = arr
                return mock

            def get_weights(self):
                return [np.ones((10, 4))]
        return Layer()


class DummyEncoder:
    def transform(self, ids):
        return np.array(ids).reshape(-1, 1)

    def inverse_transform(self, arr):
        return arr.flatten().tolist()


class TestRecommenders(unittest.TestCase):
    def setUp(self):
        # Create a simple DataFrame of 10 games
        self.df = pd.DataFrame({
            'app_id': list(range(1, 11)),
            'title': [f"Game {i}" for i in range(1, 11)],
            'tags': [['action', 'adventure'] for _ in range(10)],
            'date_release': pd.to_datetime(['2020-01-01']*10)
        })
        self.vector_store = DummyVectorStore()
        self.ncf_model = DummyNCFModel()
        self.user_encoder = DummyEncoder()
        self.game_encoder = DummyEncoder()
        self.session = DummySession()

    def test_apply_genre_filter(self):
        # Subset df to only those with 'action'
        filtered = apply_genre_filter(self.df.copy(), include_genres=[
                                      'action'], exclude_genres=None)
        self.assertTrue(all('action' in tags for tags in filtered['tags']))
        # No matches produces empty
        empty = apply_genre_filter(self.df.copy(), include_genres=[
                                   'nonexistent'], exclude_genres=None)
        self.assertTrue(empty.empty)

    def test_get_advanced_similar_games_basic(self):
        recs = get_advanced_similar_games(
            user_query='Game',
            combined_df=self.df,
            vector_store=self.vector_store,
            genres=['action'],
            release_year_filter={'comparator': 'after', 'year': 2019},
            k=5
        )
        # Should be a DataFrame up to k rows
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertLessEqual(len(recs), 5)
        self.assertIn('app_id', recs.columns)

    def test_collaborative_filtering_with_fallback(self):
        recs = collaborative_filtering_with_fallback(
            user_id=123,
            filtered_games=self.df,
            session=self.session,
            ncf_model=self.ncf_model,
            user_encoder=self.user_encoder,
            game_encoder=self.game_encoder,
            top_n=3
        )
        self.assertIsInstance(recs, pd.DataFrame)
        # Should not recommend already liked games
        self.assertFalse(
            any(a in self.session.liked_games for a in recs['app_id']))

    def test_get_user_embedding(self):
        emb = get_user_embedding(123, self.ncf_model, self.user_encoder)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (1, 4))


if __name__ == '__main__':
    unittest.main()
