# Test Sessions module
# tests/test_sessions.py
import unittest
from sessions.session_manager import get_user_session, clear_user_session


class TestSessions(unittest.TestCase):
    def setUp(self):
        # Ensure a clean state before each test
        self.user_id = 999
        clear_user_session(self.user_id)

    def test_session_creation_and_defaults(self):
        session = get_user_session(self.user_id)
        self.assertIsNotNone(session, "Session should be created")
        # Check default attributes
        self.assertEqual(session.user_id, self.user_id,
                         "User ID should be set correctly")
        self.assertIsInstance(session.liked_games, set,
                              "liked_games should be a set")
        self.assertIsInstance(session.disliked_games, set,
                              "disliked_games should be a set")
        self.assertIsInstance(session.user_preferences,
                              dict, "user_preferences should be a dict")

    def test_update_likes_dislikes(self):
        session = get_user_session(self.user_id)
        # Add likes
        session.update_likes(["Game X", "Game Y"])
        self.assertIn("Game X", session.liked_games,
                      "Game X should be in liked_games after update")
        self.assertIn("Game Y", session.liked_games,
                      "Game Y should be in liked_games after update")
        # Add dislikes
        session.update_dislikes(["Game Z"])
        self.assertIn("Game Z", session.disliked_games,
                      "Game Z should be in disliked_games after update")

    def test_update_preferences(self):
        session = get_user_session(self.user_id)
        # Update preferences
        prefs = {"genres": ["rpg", "strategy"], "release_year_filter": {
            "year": 2020, "comparator": "after"}}
        session.update_preferences(prefs)
        self.assertDictEqual(session.user_preferences, prefs,
                             "Preferences should match the provided dict")

    def test_clear_session(self):
        session = get_user_session(self.user_id)
        session.update_likes(["Game A"])
        # Clear
        clear_user_session(self.user_id)
        new_session = get_user_session(self.user_id)
        # New session should not preserve old likes
        self.assertNotIn("Game A", new_session.liked_games,
                         "Session should be cleared and not retain old liked_games")


if __name__ == '__main__':
    unittest.main()
