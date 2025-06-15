# Session Manager module
import logging

logger = logging.getLogger(__name__)


class UserSession:
    """
    Represents a user's session, tracking their preferences, liked/disliked games,
    excluded tags, user ID, and any reminders.
    """

    def __init__(self):
        self.liked_games = set()
        self.disliked_games = set()
        self.excluded_tags = set()
        self.user_preferences = {}
        self.user_id = None
        self.reminders = []
        self.vector_store = None  # Optional: store a reference to the vector store if needed

    def update_likes(self, games):
        """Add games to the set of liked games."""
        if isinstance(games, (list, set)):
            self.liked_games.update(games)
        else:
            self.liked_games.add(games)
        logger.info(f"Updated liked games: {self.liked_games}")

    def update_dislikes(self, games):
        """Add games to the set of disliked games."""
        if isinstance(games, (list, set)):
            self.disliked_games.update(games)
        else:
            self.disliked_games.add(games)
        logger.info(f"Updated disliked games: {self.disliked_games}")

    def set_user_id(self, user_id):
        """Set the user's unique identifier."""
        self.user_id = user_id
        logger.info(f"User ID set to: {self.user_id}")

    def update_preferences(self, preferences: dict):
        """
        Update user preferences with the provided dictionary.
        Existing keys will be updated, and new ones added.
        """
        if isinstance(preferences, dict):
            self.user_preferences.update(preferences)
            logger.info(f"User preferences updated: {self.user_preferences}")
        else:
            logger.warning(
                "Preferences update failed: input is not a dictionary.")

    def set_excluded_tags(self, tags):
        """Add tags to be excluded from recommendations."""
        if isinstance(tags, (list, set)):
            self.excluded_tags.update(tags)
        else:
            self.excluded_tags.add(tags)
        logger.info(f"Excluded tags set to: {self.excluded_tags}")

    def get_excluded_tags(self):
        """Return the set of excluded tags."""
        return self.excluded_tags


# In-memory session storage (for production, consider persistent storage)
_sessions = {}


def get_user_session(user_id) -> UserSession:
    """
    Retrieve the session for the given user_id. If no session exists, create a new one.

    Args:
        user_id: The unique identifier for the user (e.g., Telegram chat ID).

    Returns:
        UserSession: The user's session object.
    """
    if user_id not in _sessions:
        _sessions[user_id] = UserSession()
        logger.info(f"Created new session for user_id: {user_id}")
    return _sessions[user_id]


def clear_user_session(user_id):
    """
    Clear the session for a given user, if needed.

    Args:
        user_id: The unique identifier for the user.
    """
    if user_id in _sessions:
        del _sessions[user_id]
        logger.info(f"Cleared session for user_id: {user_id}")
