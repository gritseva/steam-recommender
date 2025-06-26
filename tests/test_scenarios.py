# tests/test_scenarios.py

import unittest
from unittest.mock import MagicMock, patch
from handlers.intent_router import route_message

# Each tuple: (incoming user message, expected intent key, expected keywords in reply)
SCENARIO_CASES = [
    ("Hi!", "additional_info", ["hello", "hi", "welcome"]),
    ("My steam ID is 76561198129676583", "additional_info",
     ["profile", "playtime", "favorite genres"]),
    ("Can you recommend me games like Outlast?",
     "recommend_games", ["Outlast", "recommend"]),
    ("What do you think about Signalis?", "opinion_request",
     ["Signalis", "feature", "pros", "cons"]),
    ("How to bake a cake?", "out_of_context",
     ["game", "recommendation", "gaming"]),
    ("Recommend top 10 action games",
     "top_games_request", ["action", "top", "games"]),
    ("I want to see the full walkthrough of Alan Wake 2",
     "video_search", ["Alan Wake 2", "youtube", "video"]),
    ("I have a prime time for Overwatch at 9:00 pm every evening, set a reminder",
     "game_session_reminder", ["reminder", "Overwatch", "9:00"]),
    ("Create my gaming profile based on my steam account",
     "user_gaming_profile", ["profile", "favorite genres", "most played"]),
    ("Порекомендуй игры, похожие на Undertale",
     "translation", ["Undertale", "recommend"]),
    ("Exclude games with mature content",
     "content_filter", ["excluded", "mature"]),
    ("I didn’t like your recommendations",
     "feedback", ["feedback", "improve"]),
    ("Compare Apex Legends to Overwatch", "game_comparison",
     ["Apex Legends", "Overwatch", "compare"]),
    ("I love Subnautica, The Last of Us, Horizon Zero Dawn, recommend games",
     "recommend_games", ["Subnautica", "The Last of Us", "Horizon Zero Dawn", "recommend"]),
]


class TestChatbotScenarios(unittest.TestCase):
    def setUp(self):
        # Create a fresh mock update and context for each test
        self.update = MagicMock()
        self.context = MagicMock()
        self.update.message.chat_id = 12345
        self.update.message.reply_text = MagicMock()

    @patch('handlers.intent_router.parse_user_intent')
    def test_scenarios_reply(self, mock_parse_intent):
        for user_message, expected_intent, expected_keywords in SCENARIO_CASES:
            with self.subTest(message=user_message, intent=expected_intent):
                # Arrange: mock the intent
                mock_parse_intent.return_value = expected_intent
                self.update.message.text = user_message

                # Act: route the message
                route_message(self.update, self.context)

                # Assert: handler sent a reply
                self.assertTrue(
                    self.update.message.reply_text.called,
                    f"No reply for message: '{user_message}'"
                )
                reply_args = self.update.message.reply_text.call_args[0]
                reply_text = reply_args[0] if reply_args else ""

                # Check that at least one expected keyword appears in the reply
                found = [kw for kw in expected_keywords if kw.lower()
                         in reply_text.lower()]
                self.assertTrue(
                    found,
                    f"Reply for '{user_message}' did not contain any of the expected keywords {expected_keywords}. Actual reply: '{reply_text}'"
                )

                # Reset mock for next iteration
                self.update.message.reply_text.reset_mock()


if __name__ == '__main__':
    unittest.main()
