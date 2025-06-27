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
        # Mock DataFrame with a variety of games
        self.df = pd.DataFrame({
            'app_id': [1, 2, 3, 4, 5, 6, 7],
            'title': [
                "Horror Game A", "Horror Game B", "Action Game C",
                "Horror Game A: DLC", "Horror Game A Remake", "Puzzle Game D", "Input Game SOMA"
            ],
            'tags': [
                ['horror', 'survival'], [
                    'horror', 'psychological'], ['action', 'shooter'],
                ['horror', 'survival'], ['horror', 'survival'], [
                    'puzzle'], ['horror', 'sci-fi']
            ],
            'genres': [
                ['Horror', 'Survival'], [
                    'Horror', 'Psychological'], ['Action', 'Shooter'],
                ['Horror', 'Survival'], ['Horror', 'Survival'], [
                    'Puzzle'], ['Horror', 'Sci-Fi']
            ],
            'date_release': pd.to_datetime([
                '2022-01-01', '2023-01-01', '2022-05-01',
                '2022-06-01', '2024-01-01', '2021-01-01', '2015-01-01'
            ]),
            'positive_ratio': [90, 85, 70, 60, 80, 75, 95]
        })
        self.DummyDoc = DummyDoc
        self.vector_store = MagicMock()
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

    def test_get_advanced_similar_games_happy_path(self):
        # similarity_search returns docs for app_ids 1, 2, 7 (the input)
        self.vector_store.similarity_search.return_value = [
            self.DummyDoc(1), self.DummyDoc(2), self.DummyDoc(7)]
        recs = get_advanced_similar_games(
            user_query="Input Game SOMA",
            combined_df=self.df,
            vector_store=self.vector_store,
            k=5
        )
        self.assertEqual(len(recs), 2)
        self.assertIn("Horror Game A", recs['title'].values)
        self.assertIn("Horror Game B", recs['title'].values)
        self.assertNotIn("Input Game SOMA", recs['title'].values)

    def test_get_advanced_similar_games_genre_filtering(self):
        self.vector_store.similarity_search.return_value = [
            self.DummyDoc(1), self.DummyDoc(2), self.DummyDoc(3)]
        recs = get_advanced_similar_games(
            user_query="Input Game SOMA",
            combined_df=self.df,
            vector_store=self.vector_store,
            genres=['horror'],
            k=5
        )
        self.assertIn("Horror Game A", recs['title'].values)
        self.assertIn("Horror Game B", recs['title'].values)
        self.assertNotIn("Action Game C", recs['title'].values)

    def test_get_advanced_similar_games_exclude_dlc_and_input(self):
        self.vector_store.similarity_search.return_value = [
            self.DummyDoc(1), self.DummyDoc(4), self.DummyDoc(7)]
        recs = get_advanced_similar_games(
            user_query=["Input Game SOMA"],
            combined_df=self.df,
            vector_store=self.vector_store,
            k=5
        )
        self.assertIn("Horror Game A", recs['title'].values)
        self.assertNotIn("Horror Game A: DLC", recs['title'].values)
        self.assertNotIn("Input Game SOMA", recs['title'].values)

    def test_get_advanced_similar_games_fuzzy_duplicate_removal(self):
        self.vector_store.similarity_search.return_value = [
            self.DummyDoc(1), self.DummyDoc(5)]
        recs = get_advanced_similar_games(
            user_query="Input Game SOMA",
            combined_df=self.df,
            vector_store=self.vector_store,
            k=5,
            similarity_threshold=90
        )
        # Only one of the two very similar titles should be present
        titles = recs['title'].values
        self.assertTrue(
            ("Horror Game A" in titles) != ("Horror Game A Remake" in titles)
        )

    def test_get_advanced_similar_games_no_results(self):
        self.vector_store.similarity_search.return_value = []
        recs = get_advanced_similar_games(
            user_query="Input Game SOMA",
            combined_df=self.df,
            vector_store=self.vector_store,
            k=5
        )
        self.assertTrue(recs.empty)

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
