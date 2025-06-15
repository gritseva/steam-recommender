# Test Sessions module
# tests/test_sessions.py

import unittest
from sessions.session_manager import get_user_session, clear_user_session


class TestSessions(unittest.TestCase):
    def test_get_and_update_session(self):
        user_id = 999
        session = get_user_session(user_id)
        self.assertIsNotNone(session, "Session should be created")
        session.update_likes(["Game X"])
        self.assertIn("Game X", session.liked_games,
                      "Liked games should include 'Game X'")
        session.update_dislikes("Game Y")
        self.assertIn("Game Y", session.disliked_games,
                      "Disliked games should include 'Game Y'")
        session.set_user_id(user_id)
        self.assertEqual(session.user_id, user_id,
                         "User ID should be set correctly")
        session.update_preferences({"genres": ["RPG", "Action"]})
        self.assertIn("genres", session.user_preferences,
                      "Preferences should include genres")

    def test_clear_session(self):
        user_id = 1000
        session = get_user_session(user_id)
        session.update_likes(["Game Z"])
        clear_user_session(user_id)
        new_session = get_user_session(user_id)
        self.assertNotIn("Game Z", new_session.liked_games,
                         "Session should be cleared and not retain old data")


if __name__ == '__main__':
    unittest.main()
