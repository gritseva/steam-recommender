# Steam Game Recommender

A Telegram bot that recommends Steam games using advanced machine learning techniques, including collaborative filtering, content-based filtering, and hybrid approaches. The bot can also track game prices, set reminders, compare games, and provide video content, all via natural language commands.

## Features
- **Personalized Game Recommendations**: Uses collaborative filtering (Neural Collaborative Filtering model), content-based filtering (vector similarity, genre, and tag filters), and hybrid fallback strategies.
- **Telegram Bot Interface**: Interact with the recommender via Telegram commands and messages.
- **Game Comparison**: Compare two games for features, genres, and more.
- **Price Tracking**: Track price changes for specific games.
- **Reminders**: Set reminders for game sessions.
- **Video Search**: Find YouTube videos related to games.
- **Multilingual Support**: Detects and translates user messages.

## How It Works
- The bot loads and preprocesses Steam game data from CSV and JSON files.
- It uses a Neural Collaborative Filtering (NCF) model for collaborative recommendations based on user preferences and history.
- Content-based recommendations leverage vector similarity search and genre/tag filtering.
- A hybrid fallback system ensures recommendations are provided even for new users or sparse data.
- User sessions and preferences are managed per Telegram chat.

## Example Commands
```
/recommend I like puzzle games and Portal.
/compare Portal 2 and The Talos Principle
/price_tracker Hollow Knight
/reminder Play Hades tomorrow at 8pm
/video Celeste speedrun
/feedback The recommendations were great, but I want more indie games.
```

## Example Conversation
```
User: Hi!
Bot: Good evening! How can I help you find your next favorite Steam game?
User: Recommend me some co-op games.
Bot: Here are some top co-op games you might enjoy: ...
```

## Project Structure
- `main.py`: Entry point for the bot
- `handlers/`: Telegram command and message handlers
- `models/`: ML models and checkpoints
- `recommenders/`: Recommendation algorithms
- `data/`: Data loading and preprocessing
- `utils/`: Utility functions
- `sessions/`: User session management
- `tests/`: Unit and integration tests
