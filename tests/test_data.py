# Test Data module
# tests/test_data.py
import unittest
import os
import pandas as pd
from config.config import GAME_CSV_PATH, BASE_DIR, COMBINED_DF_PATH
from data.data_loader import (
    load_games_csv,
    load_combined_df,
    load_json_data,
    merge_game_data,
    prepare_final_dataset,
)
from data.preprocess import preprocess_games_df, clean_game_descriptions, normalize_genres


class TestDataLoading(unittest.TestCase):
    def test_load_games_csv(self):
        df = load_games_csv()
        self.assertIsInstance(
            df, pd.DataFrame, "load_games_csv should return a DataFrame")
        # Check key columns exist
        expected_cols = {'app_id', 'title', 'genres', 'price'}
        self.assertTrue(expected_cols.issubset(set(df.columns)),
                        f"Games CSV should contain columns: {expected_cols}")

    def test_load_combined_df(self):
        df = load_combined_df()
        self.assertIsInstance(
            df, pd.DataFrame, "load_combined_df should return a DataFrame")
        # Combined DF should have both CSV and metadata fields
        self.assertIn('app_id', df.columns)
        self.assertIn('title', df.columns)

    def test_load_json_data(self):
        json_path = os.path.join(BASE_DIR, 'data', 'games_metadata.json')
        data = load_json_data(json_path)
        self.assertIsInstance(
            data, list, "load_json_data should return a list of dicts or entries")
        if data:
            self.assertIsInstance(
                data[0], (dict,), "Each JSON entry should be a dict")

    def test_merge_game_data(self):
        # Create dummy dataframes with overlapping and new columns
        df_main = pd.DataFrame({
            'app_id': [1, 2],
            'title': ['Game A', 'Game B']
        })
        df_new = pd.DataFrame({
            'app_id': [1, 2],
            'title': ['Game A', 'Game B Updated'],
            'extra_info': ['X', 'Y']
        })
        merged = merge_game_data(df_main, df_new)
        # After merge, title from df_new should overwrite df_main for app_id 2
        self.assertEqual(
            merged.loc[merged['app_id'] == 2, 'title'].iloc[0], 'Game B Updated')
        # extra_info column should be present
        self.assertIn('extra_info', merged.columns)
        # No columns ending with _new
        self.assertFalse(any(col.endswith('_new') for col in merged.columns),
                         "Redundant '_new' columns should be dropped")

    def test_prepare_final_dataset(self):
        # Using actual file paths; expects non-empty
        df = prepare_final_dataset(
            GAME_CSV_PATH,
            os.path.join(BASE_DIR, 'data', 'games_metadata.json'),
            os.path.join(BASE_DIR, 'data', 'cleaned_games_developers.csv')
        )
        self.assertIsInstance(
            df, pd.DataFrame, "prepare_final_dataset should return a DataFrame")
        self.assertFalse(df.empty, "Final dataset should not be empty")
        self.assertIn('title', df.columns,
                      "Final dataset should contain 'title' column")


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_games_df(self):
        df = pd.DataFrame({
            'app_id': ['10', '20', None],
            'date_release': ['2020-01-01', 'invalid', '2021-12-31'],
            'about_game': ['<p>Fun</p>', 'No HTML', '<div>Good</div>'],
        })
        pre = preprocess_games_df(df)
        # app_id should be integer where possible, invalid -> NaN dropped or coerced
        self.assertTrue(pre['app_id'].dtype in ('int64', 'float64'))
        # date_release should be datetime or NaT
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(pre['date_release']))

    def test_clean_game_descriptions(self):
        df = pd.DataFrame({
            'about_game': ['<p>Fun</p>', '<div>Exciting</div>', 'Plain text']
        })
        cleaned = clean_game_descriptions(df)
        for text in cleaned['about_game']:
            self.assertNotIn(
                '<', text, "HTML tags should be removed from descriptions")

    def test_normalize_genres(self):
        df = pd.DataFrame({
            'genres': [['Action', ' RPG '], 'Adventure', None, pd.Series(['Indie', None])]
        })
        norm = normalize_genres(df)
        # All non-null genre entries should be lists of stripped lowercase
        for entry in norm['genres'].dropna():
            if isinstance(entry, pd.Series):
                entry = entry.dropna().tolist()
            if entry is not None:
                self.assertIsInstance(entry, list)
                for g in entry:
                    self.assertTrue(isinstance(g, str)
                                    and g.islower() and g == g.strip())


if __name__ == '__main__':
    unittest.main()
