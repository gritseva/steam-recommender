import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from handlers.recommendation_handlers import handle_recommend_games


class TestRecommendationHandlers(unittest.TestCase):
    @patch("handlers.recommendation_handlers.get_user_session")
    @patch("handlers.recommendation_handlers.infer_user_preferences_with_llm")
    @patch("handlers.recommendation_handlers.extract_game_titles")
    @patch("handlers.recommendation_handlers.handle_translation")
    @patch("handlers.recommendation_handlers.collaborative_filtering_with_fallback")
    @patch("handlers.recommendation_handlers.generate_response")
    def test_handle_recommend_games_success(
        self, mock_generate_response, mock_collab, mock_translation, mock_extract, mock_infer, mock_get_session
    ):
        # Setup mocks
        mock_update = MagicMock()
        mock_context = MagicMock()
        mock_update.message.text = "I like Stardew Valley"
        mock_update.message.chat_id = 12345

        # Minimal DataFrame for games
        games_df = pd.DataFrame([{
            "app_id": 1,
            "title": "Stardew Valley",
            "tags": "farming,cozy",
            "genres": "Simulation",
            "themes": "",
            "description": "A farming game."
        }])
        mock_context.bot_data = {
            "games_complete_df": games_df,
            "ncf_model": MagicMock(),
            "user_encoder": MagicMock(),
            "game_encoder": MagicMock(),
            "tokenizer": MagicMock(),
            "transformer_model": MagicMock(),
        }

        # Session mock
        session = MagicMock()
        session.liked_games = ["Stardew Valley"]
        session.get_excluded_tags.return_value = []
        session.disliked_games = []
        session.user_id = 12345
        mock_get_session.return_value = session

        # Handler logic mocks
        mock_translation.return_value = ("I like Stardew Valley", "en")
        mock_extract.return_value = ["Stardew Valley"]
        mock_infer.return_value = {"liked_games": [
            "Stardew Valley"], "excluded_tags": []}
        mock_collab.return_value = games_df
        mock_generate_response.return_value = "Try Stardew Valley!"

        # Call handler
        handle_recommend_games(mock_update, mock_context)

        # Assert reply_text was called with a non-empty string
        mock_update.message.reply_text.assert_called()
        args, kwargs = mock_update.message.reply_text.call_args
        assert "Stardew Valley" in args[0]


if __name__ == "__main__":
    unittest.main()
