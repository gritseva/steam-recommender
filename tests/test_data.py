# Test Data module
# tests/test_data.py

import unittest
import os
import pandas as pd
from config.config import GAME_CSV_PATH, BASE_DIR, COMBINED_DF_PATH
from data.data_loader import load_games_csv, load_combined_df, load_json_data, merge_game_data, prepare_final_dataset
from data.preprocess import preprocess_games_df, clean_game_descriptions, normalize_genres


class TestDataLoading(unittest.TestCase):
    def test_load_games_csv(self):
        df = load_games_csv()
        self.assertIsInstance(
            df, pd.DataFrame, "load_games_csv should return a DataFrame")
        # Optionally check for expected columns if known
        self.assertTrue("app_id" in df.columns or "AppID" in df.columns,
                        "Games CSV should contain app_id column")

    def test_load_combined_df(self):
        df = load_combined_df()
        self.assertIsInstance(
            df, pd.DataFrame, "load_combined_df should return a DataFrame")

    def test_load_json_data(self):
        json_path = os.path.join(BASE_DIR, "data", "games_metadata.json")
        data = load_json_data(json_path)
        self.assertIsInstance(
            data, list, "load_json_data should return a list")

    def test_merge_game_data(self):
        # Create dummy dataframes
        df_main = pd.DataFrame({
            "app_id": [1, 2],
            "title": ["Game A", "Game B"]
        })
        df_new = pd.DataFrame({
            "app_id": [1, 2],
            "extra_info": ["Extra A", "Extra B"],
            "title": ["Game A", "Game B_new"]
        })
        merged = merge_game_data(df_main, df_new)
        self.assertIn("title", merged.columns,
                      "Merged DataFrame should have 'title' column")
        # Redundant columns should be dropped
        for col in merged.columns:
            self.assertFalse(col.endswith(
                "_new"), "Redundant columns ending with _new should be dropped")

    def test_prepare_final_dataset(self):
        # Using the real file paths from your config (assumes sample data exists)
        metadata_json_path = os.path.join(
            BASE_DIR, "data", "games_metadata.json")
        new_games_csv_path = os.path.join(
            BASE_DIR, "data", "cleaned_games_developers.csv")
        df = prepare_final_dataset(
            GAME_CSV_PATH, metadata_json_path, new_games_csv_path)
        self.assertIsInstance(
            df, pd.DataFrame, "prepare_final_dataset should return a DataFrame")
        # Check that at least the 'title' column is present after merging
        self.assertIn("title", df.columns,
                      "Final dataset should contain 'title' column")

    def test_preprocess_and_clean(self):
        # Create a dummy dataframe
        df = pd.DataFrame({
            "app_id": ["1", "2", "invalid"],
            "date_release": ["2020-01-01", "invalid-date", "2021-05-05"],
            "about_game": ["<p>Fun game</p>", "<div>Exciting</div>", "No HTML here"]
        })
        preprocessed = preprocess_games_df(df)
        self.assertTrue(preprocessed['app_id'].dtype == 'int64',
                        "app_id should be numeric after preprocessing")
        cleaned = clean_game_descriptions(preprocessed)
        for text in cleaned["about_game"]:
            self.assertNotIn(
                "<", text, "HTML tags should be removed from about_game")


if __name__ == '__main__':
    unittest.main()
